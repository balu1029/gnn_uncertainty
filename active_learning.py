from sklearn.model_selection import train_test_split
import torch
import numpy as np
import os
import shutil
import wandb
import csv
from PIL import Image, ImageDraw, ImageFont
import cairosvg

from ase.calculators.calculator import Calculator, all_properties
from ase.units import eV, fs, Angstrom, nm, kJ, mol
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.visualize import view

from datasets.md17_dataset import MD17Dataset
from uncertainty.ensemble import ModelEnsemble
from uncertainty.mve import MVE
from uncertainty.swag import SWAG
from uncertainty.evidential import EvidentialRegression
from gnn.egnn import EGNN
from datasets.helper import utils as qm9_utils
from datasets.helper.energy_calculation import OpenMMEnergyCalculation
from trainer import ModelTrainer
from datasets.helper.cv_visualizer import get_al_animation


class ALCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, model, max_uncertainty=10, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()  # Set the model to evaluation mode

        self.atom_numbers = [
            0,
            1,
            0,
            0,
            1,
            3,
            2,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            3,
            2,
            0,
            1,
            0,
            0,
            0,
        ]
        self.ground_charges = {-1: 0.0, 0: 1.0, 1: 6.0, 2: 7.0, 3: 8.0}
        self.number_to_type = {0: "H", 1: "C", 2: "N", 3: "O"}

        self.dtype = torch.float32
        self.charge_power = 2
        self.charge_scale = torch.tensor(max(self.ground_charges.values()) + 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_uncertainty = max_uncertainty
        self.max_force = 1000
        self.uncertainty_samples = []
        self.energy_unit = kJ / mol
        self.force_unit = self.energy_unit / Angstrom
        self.a_to_nm = 0.1
        self.energy_mean = self.force_mean = 0
        self.energy_std = self.force_std = 1

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_properties
    ):
        super().calculate(atoms, properties, system_changes)

        # Get atomic positions as a numpy array
        positions = atoms.get_positions()

        atom_numbers = [self.atom_numbers]
        atom_mask = torch.where(
            torch.tensor(atom_numbers) != -1, torch.tensor(1.0), torch.tensor(0.0)
        )
        atom_numbers = np.array(atom_numbers)
        coordinates = np.array(positions)

        num_atoms = atom_numbers.shape[1]
        num_snapshots = atom_numbers.shape[0]
        num_types = len(set(self.atom_numbers))

        one_hot = torch.eye(num_types, dtype=torch.float)[
            torch.tensor(atom_numbers.flatten())
        ].reshape(-1, num_atoms, num_types)
        charges = torch.tensor(
            [self.ground_charges[atom] for atom in atom_numbers.flatten()]
        ).reshape(-1, num_atoms)

        edge_mask = (
            torch.eye(num_atoms, dtype=torch.float)
            .unsqueeze(0)
            .expand(num_snapshots, -1, -1)
        )
        edge_mask = torch.where(edge_mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        edge_mask = edge_mask * atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        edge_mask = edge_mask.view(num_snapshots, -1)

        coordinates = torch.tensor(coordinates, requires_grad=True)
        coordinates = coordinates.unsqueeze(0)

        batch_size, n_nodes, _ = coordinates.size()
        atom_positions = (
            coordinates.view(batch_size * n_nodes, -1)
            .requires_grad_(True)
            .to(self.device, self.dtype)
        )
        atom_positions.retain_grad()
        atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(self.device, self.dtype)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, -1).to(
            self.device, self.dtype
        )
        one_hot = one_hot.to(self.device, self.dtype)
        charges = charges.to(self.device, self.dtype)
        nodes = qm9_utils.preprocess_input(
            one_hot, charges, self.charge_power, self.charge_scale, self.device
        )
        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)

        # Predict energy using the model
        energy, force, uncertainty = self.model.predict(
            x=atom_positions,
            h0=nodes,
            edges=edges,
            edge_attr=None,
            node_mask=atom_mask,
            edge_mask=edge_mask,
            n_nodes=n_nodes,
        )
        energy = energy * self.energy_std + self.energy_mean
        force = force * self.force_std
        uncertainty = uncertainty * self.force_std

        if uncertainty.item() > self.max_uncertainty:
            self.uncertainty_samples.append(positions * self.a_to_nm)

        # Convert energy to the appropriate ASE units (eV)
        self.results["energy"] = energy.item() * self.energy_unit

        # print("Uncertainty: ", uncertainty.item())
        # print("Energy:      ", energy.item())
        # print()
        # Store the forces in the results dictionary
        force = torch.clamp(force, -self.max_force, self.max_force)
        self.results["forces"] = force.cpu().detach().numpy() * self.force_unit

    def get_uncertainty_samples(self):
        return np.array(self.uncertainty_samples)

    def reset_uncertainty_samples(self):
        self.uncertainty_samples = []

    def drop_uncertainty_sample(self, index):
        self.uncertainty_samples.pop(index)

    def change_norm(self, norm_file):
        with open(norm_file, "r") as csvfile:
            reader = csv.reader(csvfile)
            data = {rows[0]: float(rows[1]) for rows in reader}

        self.energy_mean = data["mean_energy"]
        self.energy_std = data["std_energy"]
        self.force_mean = data["mean_forces"]
        self.force_std = data["std_forces"]


class ActiveLearning:

    def __init__(
        self,
        model,
        max_uncertainty=10,
        num_ensembles: int = 3,
        in_nf: int = 12,
        hidden_nf: int = 16,
        n_layers: int = 2,
        molecule_path: str = "al/base_data/start_pos.xyz",
        norm_file="al/base_data/norms_dataset.csv",
        lr=1e-4,
    ) -> None:
        """
        Initializes an instance of the ActiveLearning class.

        Parameters:
        - model_path (str): The path to the pre-trained model file. Default is "gnn/models/ala_converged_10000.pt".
        - num_ensembles (int): The number of ensembles in the model. Default is 3. HAS TO MATCH THE HYPERPARAMETERS OF THE MODEL IN MODEL_PATH.
        - in_nf (int): The number of input node features. Default is 12. HAS TO MATCH THE HYPERPARAMETERS OF THE MODEL IN MODEL_PATH.
        - hidden_nf (int): The number of hidden node features. Default is 16. HAS TO MATCH THE HYPERPARAMETERS OF THE MODEL IN MODEL_PATH.
        - n_layers (int): The number of layers in the model. Default is 2. HAS TO MATCH THE HYPERPARAMETERS OF THE MODEL IN MODEL_PATH.
        - molecule_path (str): The path to the molecule file. Default is "datasets/files/alaninedipeptide/xyz/ala_single.xyz".
        """
        self.atom_numbers = [
            0,
            1,
            0,
            0,
            1,
            3,
            2,
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            1,
            3,
            2,
            0,
            1,
            0,
            0,
            0,
        ]
        self.number_to_type = {0: "H", 1: "C", 2: "N", 3: "O"}

        self.cov_radii = {0: 0.32, 1: 0.75, 2: 0.71, 3: 0.63}

        self.bonds = [
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (4, 5),
            (4, 6),
            (6, 7),
            (6, 8),
            (8, 9),
            (8, 10),
            (8, 14),
            (10, 11),
            (10, 12),
            (10, 13),
            (14, 15),
            (14, 16),
            (16, 17),
            (16, 18),
            (18, 19),
            (18, 20),
            (18, 21),
        ]

        self.type_to_number = {v: k for k, v in self.number_to_type.items()}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.temperature = 300
        self.timestep = 0.5 * fs

        self.lr = lr

        # Instantiate the model
        self.num_ensembles = num_ensembles
        self.model = model.to(self.device)

        self.num_ensembles = num_ensembles

        self.calc = ALCalculator(model=model, max_uncertainty=max_uncertainty)
        if norm_file is not None:
            self.calc.change_norm(norm_file)
        self.atoms = read(molecule_path)
        self.atoms.calc = self.calc

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature)

        friction_coefficient = 0.01  # Friction coefficient in 1/fs
        self.dyn = Langevin(
            self.atoms,
            timestep=self.timestep,
            temperature_K=self.temperature,
            friction=friction_coefficient,
        )
        self.oracle = OpenMMEnergyCalculation()
        self.trainer = ModelTrainer(self.calc.model, self.num_ensembles)

    def sample_rand_pos(self, data_path) -> None:
        """
        Samples a random position from the dataset and sets the atoms object to this position.

        Parameters:
        - data_path (str): The path to the dataset folder.
        """
        # dataset = MD17Dataset(data_path, subtract_self_energies=False, in_unit="kj/mol", scale=False, determine_norm=True)
        coordinates = self._read_coordinates(data_path, last_n=3)
        np.random.seed(None)
        idx = np.random.randint(0, len(coordinates))
        self.atoms.positions = coordinates[idx]
        # Set velocities to zero
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature)

    def _read_coordinates(self, foldername: str, last_n: int = 1):
        all_atom_numbers = []
        all_coordinates = []
        files = [
            f
            for f in os.listdir(foldername)
            if f.endswith(".xyz") and f.startswith("data_")
        ]
        if len(files) > 0:
            files = [
                f
                for f in files
                if int(f.split("_")[1].split(".")[0]) >= len(files) - last_n
            ]
        if len(files) < last_n:
            files.append("train.xyz")
        for filename in files:
            with open(os.path.join(foldername, filename), "r") as file:
                while True:
                    line = file.readline()
                    if line == "":
                        break
                    num_atoms = int(line.strip())
                    _ = file.readline()  # Skip the energy line

                    # Initialize lists to store atom types and coordinates
                    atom_numbers = []

                    coordinates = []

                    # Read atom types and coordinates for each atom
                    for _ in range(num_atoms):
                        line = file.readline().split()
                        atom_type = line[0]
                        x, y, z = map(float, line[1:4])

                        atom_numbers.append(int(self.type_to_number[atom_type]))
                        coordinates.append([x, y, z])
                    all_coordinates.append(coordinates)
                    all_atom_numbers.append(atom_numbers)

        return np.array(all_coordinates)

    def run_simulation(self, steps: int, show_traj: bool = False) -> np.array:

        with Trajectory("ala.traj", "w", self.atoms) as traj:
            for i in range(steps):
                self.dyn.run(1)
                traj.write(self.atoms)

        if show_traj:
            traj = Trajectory("ala.traj")
            view(traj, block=True)

    def improve_model(
        self,
        num_iter,
        steps_per_iter,
        use_wandb=False,
        run_idx=1,
        model_path="gnn/models/ala_converged_1000000.pt",
        max_iterations=4000,
        epochs_per_iter=20,
        calibrate=False,
        force_uncertainty=False,
    ):
        model_out_path = f"al/run{run_idx}/models/"
        data_out_path = f"al/run{run_idx}/data/"
        train_out_path = f"al/run{run_idx}/data/train/"
        valid_out_path = f"al/run{run_idx}/data/valid/"
        if not os.path.exists(f"al/run{run_idx}"):
            os.makedirs(f"al/run{run_idx}")
            os.makedirs(model_out_path)
            os.makedirs(data_out_path)
            os.makedirs(train_out_path)
            os.makedirs(valid_out_path)
            os.makedirs(f"al/run{run_idx}/plots")
            os.makedirs(f"al/run{run_idx}/eval")
            # Copy the file to the directory
            shutil.copy2("al/base_data/train/train.xyz", train_out_path)
            shutil.copy2("al/base_data/valid/valid.xyz", valid_out_path)

        batch_size = 32
        val_batch_size = 256
        lr = self.lr

        log_interval = 100
        force_weight = 5
        energy_weight = 1

        optimizer = torch.optim.AdamW(self.calc.model.parameters(), lr=lr)
        criterion = torch.nn.L1Loss()
        if use_wandb:
            self.init_wandb(
                model=self.model,
                criterion=criterion,
                optimizer=optimizer,
                model_path=model_path,
                lr=lr,
                batch_size=batch_size,
                epochs_per_iter=epochs_per_iter,
                name=f"al_{run_idx}",
            )

        trainset = MD17Dataset(
            f"{train_out_path}",
            subtract_self_energies=False,
            in_unit="kj/mol",
            scale=True,
            determine_norm=True,
            store_norm_path=f"{data_out_path}norms_dataset.csv",
        )
        validset = MD17Dataset(
            f"{valid_out_path}",
            subtract_self_energies=False,
            in_unit="kj/mol",
            scale=True,
            load_norm_path=f"{data_out_path}norms_dataset.csv",
        )
        testset = MD17Dataset(
            f"al/base_data/test",
            subtract_self_energies=False,
            in_unit="kj/mol",
            scale=True,
            load_norm_path=f"{data_out_path}norms_dataset.csv",
        )
        self.calc.change_norm(f"{data_out_path}norms_dataset.csv")
        validloader = torch.utils.data.DataLoader(
            validset, batch_size=val_batch_size, shuffle=True
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=val_batch_size, shuffle=False
        )

        self.model.calibrate_uncertainty(
            validloader,
            device=self.device,
            dtype=self.dtype,
            path=f"al/run{run_idx}/plots/calibration_init.svg",
        )
        self.model.evaluate_all(
            validloader,
            device=self.device,
            dtype=torch.float32,
            plot_name=f"al/run{run_idx}/plots/plot_init",
            csv_path=f"al/run{run_idx}/eval/eval.csv",
            test_loader_out=testloader,
            best_model_available=False,
            use_energy_uncertainty=True,
            use_force_uncertainty=force_uncertainty,
        )
        self.model.valid_epoch(
            testloader,
            criterion,
            self.device,
            self.dtype,
            force_weight=force_weight,
            energy_weight=energy_weight,
            test=True,
        )
        self.model.valid_on_cv(
            validationloader=testloader,
            save_path=f"al/run{run_idx}/plots/heatmap_init",
            device=device,
            dtype=self.dtype,
            use_force_uncertainty=force_uncertainty,
        )
        self.model.epoch_summary(
            epoch=f"Initital validation",
            use_wandb=use_wandb,
            additional_logs={
                "dataset_size": len(trainset.coordinates),
                "max_uncertainty": self.calc.max_uncertainty,
            },
        )
        self.model.drop_metrics()

        if use_wandb:
            wandb.finish()
        for i in range(num_iter):
            # self.run_simulation(steps_per_iter, show_traj=False)
            num_uncertainty_samples = 0
            steps = {}
            for k in range(steps_per_iter):
                self.sample_rand_pos(train_out_path)
                for j in range(max_iterations):
                    if (
                        len(self.calc.get_uncertainty_samples())
                        > num_uncertainty_samples
                    ):
                        num_new_samples = (
                            len(self.calc.get_uncertainty_samples())
                            - num_uncertainty_samples
                        )
                        if num_new_samples > 1:
                            print(
                                f"Found {num_new_samples} uncertainty samples at step {j}. Dropping index -2"
                            )
                            # print(
                            #    "this should only happen at step 1 because setting resetting the position causes ase to do 2 energy calculations."
                            # )
                            self.calc.drop_uncertainty_sample(-2)

                        num_uncertainty_samples = len(
                            self.calc.get_uncertainty_samples()
                        )
                        print(
                            f"Found uncertainty sample {k} after {j} steps.", flush=True
                        )
                        steps[k] = j
                        break
                    if not self.check_atomic_distances():
                        print(
                            f"Atomic distances for sample {k} are too large or too small. Resetting this trajectory."
                        )
                        steps[k] = j
                        break
                    self.dyn.run(1)
                    if j == max_iterations - 1:
                        print(
                            f"Did not find any uncertainty samples for sample {k}.",
                            flush=True,
                        )
                        steps[k] = max_iterations

            self._log_md_steps(i, steps, f"al/run{run_idx}/eval/md_steps.csv")
            samples = self.calc.get_uncertainty_samples()
            self.calc.reset_uncertainty_samples()
            energies = self.oracle.calc_energy(samples)
            forces = self.oracle.calc_forces(samples)

            if len(samples) > 0:
                print(
                    f"Training model {i}. Added {len(samples)} samples to the dataset."
                )
                if len(samples) == 1:
                    valid_ratio = None
                    valid_file = None
                else:
                    valid_ratio = 0.1
                    valid_file = f"{valid_out_path}valid{i}.xyz"
                self.add_data(
                    samples,
                    energies,
                    forces,
                    f"{train_out_path}train{i}.xyz",
                    valid_ratio=valid_ratio,
                    valid_file=valid_file,
                )

                trainset = MD17Dataset(
                    f"{train_out_path}",
                    subtract_self_energies=False,
                    in_unit="kj/mol",
                    scale=True,
                    determine_norm=False,
                    load_norm_path=f"{data_out_path}norms_dataset.csv",
                )  # store_norm_path=f"{data_out_path}norms{i}.csv")
                validset = MD17Dataset(
                    f"{valid_out_path}",
                    subtract_self_energies=False,
                    in_unit="kj/mol",
                    scale=True,
                    load_norm_path=f"{data_out_path}norms_dataset.csv",
                )
                trainloader = torch.utils.data.DataLoader(
                    trainset, batch_size=32, shuffle=True
                )
                validloader = torch.utils.data.DataLoader(
                    validset, batch_size=val_batch_size, shuffle=True
                )
                self.model.set_wandb_name(f"al_{run_idx}_{i}")
                self.model.prepare_al_iteration()
                self.model.fit(
                    epochs=epochs_per_iter,
                    train_loader=trainloader,
                    valid_loader=validloader,
                    test_loader=testloader,
                    device=self.device,
                    dtype=self.dtype,
                    model_path=f"{model_out_path}model_{i}.pt",
                    use_wandb=use_wandb,
                    force_weight=force_weight,
                    energy_weight=energy_weight,
                    lr=lr,
                    additional_logs={
                        "dataset_size": len(trainset.coordinates),
                        "max_uncertainty": self.calc.max_uncertainty,
                    },
                )
                self.model.load_best_model()
                if calibrate:
                    self.model.calibrate_uncertainty(
                        validloader,
                        device=self.device,
                        dtype=self.dtype,
                        path=f"al/run{run_idx}/plots/calibration_{i}.svg",
                    )
                self.model.evaluate_all(
                    validloader,
                    device=self.device,
                    dtype=torch.float32,
                    plot_name=f"al/run{run_idx}/plots/plot_{i}",
                    csv_path=f"al/run{run_idx}/eval/eval.csv",
                    test_loader_out=testloader,
                    best_model_available=False,
                    use_energy_uncertainty=True,
                    use_force_uncertainty=force_uncertainty,
                )
                self.model.valid_on_cv(
                    validationloader=testloader,
                    save_path=f"al/run{run_idx}/plots/heatmap_{i}",
                    device=device,
                    dtype=self.dtype,
                    use_force_uncertainty=force_uncertainty,
                )

                torch.save(self.model.state_dict(), f"{model_out_path}model_{i}.pt")
                get_al_animation(f"al/run{run_idx}/data/train/")
            else:
                print(f"No uncertainty samples found in iteration {i}.")

    def add_data(
        self, samples, labels, forces, out_file, valid_ratio=None, valid_file=None
    ):
        samples = samples * 10

        if valid_ratio is not None:
            if valid_file is None:
                raise ValueError(
                    "If valid_ratio is not None, valid_file must be provided."
                )

            (
                train_samples,
                valid_samples,
                train_labels,
                valid_labels,
                train_forces,
                valid_forces,
            ) = train_test_split(
                samples, labels, forces, test_size=valid_ratio, random_state=42
            )
        else:
            train_samples = samples
            train_labels = labels
            train_forces = forces

        with open(out_file, "w") as file:
            for i in range(len(train_samples)):
                file.write(f"{len(train_samples[i])}\n")
                file.write(f"{train_labels[i]}\n")
                for j in range(len(train_samples[i])):
                    file.write(
                        f"{self.number_to_type[self.atom_numbers[j]]} {train_samples[i][j][0]} {train_samples[i][j][1]} {train_samples[i][j][2]} {train_forces[i][j][0]} {train_forces[i][j][1]} {train_forces[i][j][2]}\n"
                    )

        if valid_file is not None:
            with open(valid_file, "w") as file:
                for i in range(len(valid_samples)):
                    file.write(f"{len(valid_samples[i])}\n")
                    file.write(f"{valid_labels[i]}\n")
                    for j in range(len(valid_samples[i])):
                        file.write(
                            f"{self.number_to_type[self.atom_numbers[j]]} {valid_samples[i][j][0]} {valid_samples[i][j][1]} {valid_samples[i][j][2]} {valid_forces[i][j][0]} {valid_forces[i][j][1]} {valid_forces[i][j][2]}\n"
                        )

    def _log_md_steps(self, iteration: int, steps: dict, csv_path: str):
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["iteration", "sample", "MD steps"])
        with open(csv_path, "a") as csvfile:
            writer = csv.writer(csvfile)
            for sample in range(len(steps.keys())):
                writer.writerow([iteration, sample, steps[sample]])

    def init_wandb(
        self,
        model,
        criterion,
        optimizer,
        model_path,
        lr,
        batch_size,
        epochs_per_iter,
        name="al",
    ):
        wandb.init(
            # set the wandb project where this run will be logged
            project="ActiveLearning",
            name=name,
            # track hyperparameters and run metadata
            config={
                "name": "alaninedipeptide",
                "learning_rate_start": lr,
                "optimizer": type(optimizer).__name__,
                "batch_size": batch_size,
                "loss_fn": type(criterion).__name__,
                "model_checkpoint": model_path,
                "model": type(model).__name__,
                "epochs_per_iter": epochs_per_iter,
                "max_uncertainty": self.calc.max_uncertainty,
            },
        )

    def check_atomic_distances(self):
        coords = self.atoms.get_positions()
        atom_numbers = self.calc.atom_numbers
        distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        for i, j in self.bonds:
            dist = distances[i, j]
            cov_a = self.cov_radii[atom_numbers[i]]
            cov_b = self.cov_radii[atom_numbers[j]]

            max_dist = (cov_a + cov_b) * 1.5
            min_dist = (cov_a + cov_b) * 0.5

            if dist > max_dist:
                print(f"Distance between {i} and {j} is {dist} which is too large.")
                return False
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = distances[i, j]
                cov_a = self.cov_radii[atom_numbers[i]]
                cov_b = self.cov_radii[atom_numbers[j]]
                min_dist = (cov_a + cov_b) * 0.5

            if dist < min_dist:
                print(f"Distance between {i} and {j} is {dist} which is too small.")
                return False
        return True

    def _create_temp_intermediate_dataset(self, data_path, i):
        # loads the first n valid_i.xyz files in a temporary folder to create a dataset from this folder
        temp_folder = f"al/run{i}/data/temp/"
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        file = f"{data_path}valid.xyz"
        shutil.copy2(file, f"{temp_folder}valid.xyz")
        for j in range(i):
            file = f"{data_path}valid{j}.xyz"
            shutil.copy2(file, f"{temp_folder}valid{j}.xyz")
        return temp_folder

    def _remove_temp_intermediate_dataset(self, temp_folder):
        shutil.rmtree(temp_folder)

    def eval_on_cv(self, run_idx, model, device, dtype, use_force_uncertainty=False):
        model_out_path = f"al/run{run_idx}/models/"
        data_out_path = f"al/run{run_idx}/data/"
        val_batch_size = 256

        testset = MD17Dataset(
            f"al/base_data/test",
            subtract_self_energies=False,
            in_unit="kj/mol",
            scale=True,
            load_norm_path=f"{data_out_path}norms_dataset.csv",
        )
        self.calc.change_norm(f"{data_out_path}norms_dataset.csv")
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=val_batch_size, shuffle=False
        )

        for i in range(len(os.listdir(model_out_path))):
            tmp_path = self._create_temp_intermediate_dataset(
                f"al/run{run_idx}/data/valid/", i + 1
            )
            validset = MD17Dataset(
                tmp_path,
                subtract_self_energies=False,
                in_unit="kj/mol",
                scale=True,
                load_norm_path=f"{data_out_path}norms_dataset.csv",
            )
            validloader = torch.utils.data.DataLoader(
                validset, batch_size=val_batch_size, shuffle=True
            )
            self._remove_temp_intermediate_dataset(tmp_path)

            model_path = f"{model_out_path}model_{i}.pt"
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device(self.device))
            )

            model.eval()
            model.calibrate_uncertainty(
                validloader,
                device=device,
                dtype=dtype,
            )
            model.valid_on_cv(
                validationloader=testloader,
                save_path=f"al/run{run_idx}/plots/heatmap_{i}",
                device=device,
                dtype=dtype,
                use_force_uncertainty=use_force_uncertainty,
            )

    def svg_to_png(self, svg_path, png_path):
        """Convert an SVG file to a PNG file."""
        cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=200)

    def create_gif_from_svgs(
        self, run_idx, duration=1000, loop=0, img_type="energy_error"
    ):
        """
        Create a GIF from a list of SVG image paths.

        Parameters:
            svg_paths (list of str): List of file paths to SVG images.
            output_gif_path (str): Output file path for the GIF.
            duration (int): Duration for each frame in milliseconds.
            loop (int): Number of loops for the GIF (0 for infinite).
        """

        svg_paths = [
            f"al/run{run_idx}/plots/heatmap_{i}_{img_type}.svg"
            for i in range(
                len(
                    [
                        p
                        for p in os.listdir(f"al/run{run_idx}/plots/")
                        if img_type in p
                        and p.startswith("heatmap")
                        and p.endswith(".svg")
                    ]
                )
            )
        ]
        print(svg_paths[0], svg_paths[-1])
        png_files = []
        for i, svg_path in enumerate(svg_paths):
            # Convert each SVG to PNG
            png_path = f"temp_image_{i}.png"
            print(f"Converting {svg_path} to {png_path}")
            self.svg_to_png(svg_path, png_path)
            png_files.append(png_path)

        # Create GIF from PNG files
        output_path = f"al/run{run_idx}/plots/heatmap_{img_type}.gif"
        images = [Image.open(png) for png in png_files]

        # Use default Pillow font
        font = ImageFont.load_default(size=40)

        # Add centered text to each frame
        for idx, image in enumerate(images):
            draw = ImageDraw.Draw(image)
            text = f"Iteration {idx + 1}"  # Change text for each frame
            text_color = "black"

            # Calculate centered position
            image_width, image_height = image.size
            bbox = draw.textbbox((0, 0), text, font=font)  # Get bounding box
            text_width = bbox[2] - bbox[0]  # right - left
            text_x = (image_width - text_width) // 2
            text_y = 10

            # Draw text
            draw.text((text_x, text_y), text, font=font, fill=text_color)

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
        )

        # Clean up temporary PNG files
        for png_file in png_files:
            os.remove(png_file)


if __name__ == "__main__":
    # model_path = "gnn/models/ensemble3_20241106_095153/model_0.pt"
    # model_path = "gnn/models/mve_20241127_164156/model_3.pt"
    # model_path = "gnn/models/evi_20241129_140558/model_0.pt"
    model_path = "gnn/models/swag5_20241202_091422/model_2.pt"

    num_ensembles = 3
    in_nf = 12
    hidden_nf = 32
    n_layers = 4
    # model = MVE(EGNN, in_node_nf=in_nf, in_edge_nf=0, hidden_nf=hidden_nf, n_layers=n_layers, multi_dec=True)
    # model = ModelEnsemble(
    #     EGNN,
    #     num_models=num_ensembles,
    #     in_node_nf=in_nf,
    #     in_edge_nf=0,
    #     hidden_nf=hidden_nf,
    #     n_layers=n_layers,
    # )
    model = SWAG(
        EGNN,
        in_node_nf=in_nf,
        in_edge_nf=0,
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        sample_size=3,
    )
    # model = MVE(
    #     EGNN,
    #     in_node_nf=in_nf,
    #     in_edge_nf=0,
    #     hidden_nf=hidden_nf,
    #     n_layers=n_layers,
    #     multi_dec=True,
    # )
    # model = EvidentialRegression(
    #     EGNN,
    #     in_node_nf=in_nf,
    #     in_edge_nf=0,
    #     hidden_nf=hidden_nf,
    #     n_layers=n_layers,
    # )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    al = ActiveLearning(
        max_uncertainty=8,
        num_ensembles=num_ensembles,
        in_nf=in_nf,
        hidden_nf=hidden_nf,
        n_layers=n_layers,
        model=model,
        lr=1e-3,
    )
    # al.run_simulation(1000, show_traj=True)
    # print(len(al.calc.get_uncertainty_samples()))
    max_idx = np.array(
        [
            int(x.split("run")[1])
            for x in os.listdir("al/")
            if x.startswith("run") and not x.endswith("_")
        ]
    ).max()
    print(max_idx)

    al.improve_model(
        200,
        100,
        run_idx=max_idx + 1,
        use_wandb=True,
        model_path=model_path,
        epochs_per_iter=20,
        calibrate=True,
        force_uncertainty=False,
    )
    # al.eval_on_cv(59, model, al.device, al.dtype, use_force_uncertainty=False)
    # al.create_gif_from_svgs(54, duration=1000, loop=0, img_type="energy_error")
