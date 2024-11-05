import torch
import numpy as np
import os
import shutil
import wandb

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
from gnn.egnn import EGNN
from datasets.helper import utils as qm9_utils
from datasets.helper.energy_calculation import OpenMMEnergyCalculation
from trainer import ModelTrainer
import csv



class ALCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, max_uncertainty=10, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()  # Set the model to evaluation mode

        self.atom_numbers  = [0, 1, 0, 0, 1, 3, 2, 0, 1, 0, 1, 0, 0, 0, 1, 3, 2, 0, 1, 0, 0, 0]
        self.ground_charges  = {-1: 0.0,
                                0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}
        self.number_to_type = {0: "H",
                               1: "C",
                               2: "N",
                               3: "O"}
        
        self.dtype = torch.float32
        self.charge_power = 2
        self.charge_scale = torch.tensor(max(self.ground_charges.values())+1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_uncertainty = max_uncertainty
        self.max_force = 300
        self.uncertainty_samples = []
        self.energy_unit = kJ / mol
        self.force_unit = self.energy_unit / Angstrom
        self.a_to_nm = 0.1
        self.energy_mean = self.force_mean = 0
        self.energy_std = self.force_std = 1

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_properties):
        super().calculate(atoms, properties, system_changes)
        
        # Get atomic positions as a numpy array
        positions = atoms.get_positions()


        atom_numbers = [self.atom_numbers]
        atom_mask = torch.where(torch.tensor(atom_numbers) != -1, torch.tensor(1.0), torch.tensor(0.0))
        atom_numbers = np.array(atom_numbers)
        coordinates = np.array(positions)

        num_atoms = atom_numbers.shape[1]
        num_snapshots = atom_numbers.shape[0]
        num_types = len(set(self.atom_numbers))

        one_hot = torch.eye(num_types, dtype=torch.float)[torch.tensor(atom_numbers.flatten())].reshape(-1,num_atoms, num_types)
        charges = torch.tensor([self.ground_charges[atom] for atom in atom_numbers.flatten()]).reshape(-1, num_atoms)

        edge_mask = torch.eye(num_atoms, dtype=torch.float).unsqueeze(0).expand(num_snapshots, -1, -1)
        edge_mask = torch.where(edge_mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        edge_mask = edge_mask * atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        edge_mask = edge_mask.view(num_snapshots, -1)

        coordinates = torch.tensor(coordinates, requires_grad=True)
        coordinates = coordinates.unsqueeze(0)

        batch_size, n_nodes, _ = coordinates.size()
        atom_positions = coordinates.view(batch_size * n_nodes, -1).requires_grad_(True).to(self.device, self.dtype)
        atom_positions.retain_grad()
        atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(self.device, self.dtype)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, -1).to(self.device, self.dtype)
        one_hot = one_hot.to(self.device, self.dtype)
        charges = charges.to(self.device, self.dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, self.charge_power, self.charge_scale, self.device)
        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)

        # Predict energy using the model
        energy, force, uncertainty = self.model.predict(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        energy = energy * self.energy_std + self.energy_mean
        force = force * self.force_std + self.force_mean
        uncertainty = uncertainty * self.force_std



        if uncertainty.item() > self.max_uncertainty:
            self.uncertainty_samples.append(positions * self.a_to_nm)

        # Convert energy to the appropriate ASE units (eV)
        self.results['energy'] = energy.item() * self.energy_unit

        #print("Uncertainty: ", uncertainty.item())
        #print("Energy:      ", energy.item())
        #print()
        # Store the forces in the results dictionary
        force = torch.clamp(force, -self.max_force, self.max_force)
        self.results['forces'] = force.cpu().detach().numpy() * self.force_unit

    def get_uncertainty_samples(self):
        return np.array(self.uncertainty_samples)
    
    def reset_uncertainty_samples(self):
        self.uncertainty_samples = []   
    
    def change_norm(self, norm_file):
        with open(norm_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = {rows[0]: float(rows[1]) for rows in reader}

        self.energy_mean = data["mean_energy"]
        self.energy_std = data["std_energy"]
        self.force_mean = data["mean_forces"]
        self.force_std = data["std_forces"]


class ActiveLearning:
    
    def __init__(self, model, max_uncertainty=10, num_ensembles:int=3, in_nf:int=12, hidden_nf:int=16, n_layers:int=2, molecule_path:str="datasets/files/ala_converged_1000000/start_pos.xyz", norm_file="datasets/files/norm_values.csv") -> None:
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
        self.atom_numbers  = [0, 1, 0, 0, 1, 3, 2, 0, 1, 0, 1, 0, 0, 0, 1, 3, 2, 0, 1, 0, 0, 0]
        self.number_to_type = {0: "H",
                               1: "C",
                               2: "N",
                               3: "O"}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.temperature = 300
        self.timestep = 0.5 * fs

        # Instantiate the model
        self.num_ensembles = num_ensembles
        self.model = model.to(self.device)

        self.num_ensembles = num_ensembles

        self.calc = ALCalculator(model=model, max_uncertainty=max_uncertainty)
        if norm_file is not None:
            self.calc.change_norm(norm_file)
        self.atoms = read(molecule_path)
        self.atoms.set_calculator(self.calc)

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature)

        friction_coefficient = 0.01  # Friction coefficient in 1/fs
        self.dyn = Langevin(self.atoms, timestep=self.timestep, temperature_K=self.temperature, friction=friction_coefficient)
        self.oracle = OpenMMEnergyCalculation()
        self.trainer = ModelTrainer(self.calc.model, self.num_ensembles)

    def sample_rand_pos(self, data_path)->None:
        """
        Samples a random position from the dataset and sets the atoms object to this position.

        Parameters:
        - data_path (str): The path to the dataset file.
        """
        dataset = MD17Dataset(data_path, subtract_self_energies=False, in_unit="kj/mol", scale=False, determine_norm=True)
        np.random.seed(None)
        idx = np.random.randint(0, len(dataset))
        self.atoms.positions = np.array(dataset.coordinates[idx])

    def run_simulation(self, steps:int, show_traj:bool=False)->np.array:

        with Trajectory('ala.traj', 'w', self.atoms) as traj:
            for i in range(steps):
                self.dyn.run(1)
                traj.write(self.atoms)
       
        
        if show_traj:
            traj = Trajectory('ala.traj')
            view(traj, block=True)


    def improve_model(self, num_iter, steps_per_iter, use_wandb=False, run_idx=1, model_path="gnn/models/ala_converged_1000000.pt", max_iterations=5000):
        model_out_path = f"al/run{run_idx}/models/"
        data_out_path = f"al/run{run_idx}/data/"
        if not os.path.exists(f"al/run{run_idx}"):
            os.makedirs(f"al/run{run_idx}")
            os.makedirs(model_out_path)
            os.makedirs(data_out_path)
            # Copy the file to the directory
            shutil.copy2("datasets/files/train_in2/dataset.xyz", data_out_path)

        batch_size = 32
        lr=1e-3
        epochs_per_iter = 50

        log_interval = 100
        force_weight = 5
        energy_weight = 1
        
        optimizer = torch.optim.AdamW(self.calc.model.parameters(), lr=1e-3)
        criterion = torch.nn.L1Loss()
        if use_wandb:
            self.init_wandb(model=self.model, criterion=criterion, optimizer=optimizer, model_path=model_path, lr=lr, batch_size=batch_size, epochs_per_iter=epochs_per_iter)


        trainset = MD17Dataset(f"{data_out_path}", subtract_self_energies=False, in_unit="kj/mol", scale=True, determine_norm=True, store_norm_path=f"{data_out_path}norms_dataset.csv")
        validset = MD17Dataset(f"datasets/files/active_learning_validation2", subtract_self_energies=False, in_unit="kj/mol", scale=True, load_norm_path=f"{data_out_path}norms_dataset.csv")
        self.calc.change_norm(f"{data_out_path}norms_dataset.csv")
        validloader = torch.utils.data.DataLoader(validset, batch_size=512, shuffle=True)
        self.model.valid_epoch(validloader, criterion, self.device, self.dtype, force_weight=force_weight, energy_weight=energy_weight)
        self.model.epoch_summary(epoch=f"Initital validation", use_wandb=use_wandb, additional_logs={"dataset_size": len(trainset.coordinates)})                    
        self.model.drop_metrics()

        for i in range(num_iter):
            #self.run_simulation(steps_per_iter, show_traj=False)
            number_not_found = 0
            for k in range(steps_per_iter):
                self.sample_rand_pos(data_out_path)
                j = 0
                while len(self.calc.get_uncertainty_samples()) == k - number_not_found:
                    self.dyn.run(1)
                    j += 1
                    if j > max_iterations:
                        print("No more uncertainty samples found.")
                        number_not_found += 1
                        break
                
                if not j > max_iterations:
                    print(f"Found uncertainty sample after {j} steps.", flush=True)

            samples = self.calc.get_uncertainty_samples()
            self.calc.reset_uncertainty_samples()
            energies = self.oracle.calc_energy(samples)   
            forces = self.oracle.calc_forces(samples)   
            
            if len(samples) > 0:
                print(f"Training model {i}. Added {len(samples)} samples to the dataset.")

                self.add_data(samples, energies, forces, f"{data_out_path}data_{i}.xyz")

                trainset = MD17Dataset(f"{data_out_path}", subtract_self_energies=False, in_unit="kj/mol", scale=True, determine_norm=True, store_norm_path=f"{data_out_path}norms{i}.csv")
                validset = MD17Dataset(f"datasets/files/active_learning_validation2", subtract_self_energies=False, in_unit="kj/mol", scale=True, load_norm_path=f"{data_out_path}norms{i}.csv")
                self.calc.change_norm(f"{data_out_path}norms{i}.csv")
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
                validloader = torch.utils.data.DataLoader(validset, batch_size=512, shuffle=True)

                for epoch in range(epochs_per_iter):
                    self.model.train_epoch(trainloader, optimizer, criterion, epoch, self.device, self.dtype, force_weight=force_weight, energy_weight=energy_weight, log_interval=log_interval)
                    #self.trainer.train(num_epochs=2, learning_rate=1e-5, folder=data_out_path)
                    self.model.valid_epoch(validloader, criterion, self.device, self.dtype, force_weight=force_weight, energy_weight=energy_weight)
                    self.model.epoch_summary(epoch=f"Validation {i}_{epoch}", use_wandb=use_wandb, additional_logs={"dataset_size": len(trainset.coordinates)})                    
                    self.model.drop_metrics()
                
                torch.save(self.trainer.model.state_dict(), f"{model_out_path}model_{i}.pt")
                
        if use_wandb:
            wandb.finish()
                
    def add_data(self, samples, labels, forces, out_file):
        num_molecules = len(samples)
        samples = samples * 10
        with open(out_file, 'w') as file:
            for i in range(num_molecules):
                file.write(f"{len(samples[i])}\n")
                file.write(f"{labels[i]}\n")
                for j in range(len(samples[i])):
                    file.write(f"{self.number_to_type[self.atom_numbers[j]]} {samples[i][j][0]} {samples[i][j][1]} {samples[i][j][2]} {forces[i][j][0]} {forces[i][j][1]} {forces[i][j][2]}\n")

    def init_wandb(self, model, criterion, optimizer, model_path, lr, batch_size, epochs_per_iter):
        wandb.init(
                # set the wandb project where this run will be logged
                project="ActiveLearning",

                # track hyperparameters and run metadata
                config={
                "name": "alaninedipeptide",
                "learning_rate_start": lr,
                "optimizer": type(optimizer).__name__,
                "batch_size": batch_size,
                "loss_fn" : type(criterion).__name__,
                "model_checkpoint": model_path,
                "model" : type(model).__name__,
                "epochs_per_iter": epochs_per_iter,
                })
        


if __name__ == "__main__":
    model_path = "gnn/models/mve_2/model_0.pt"
    #model_path = "al/run10/models/model_19.pt"
    #model_path = "gnn/models/ala_converged_1000000_even_larger.pt"
    model_path = "gnn/models/ensemble3_6/model_0.pt"
    model_path = "gnn/models/ensemble3_20241018_101159/model_0.pt"
    
    num_ensembles = 3
    in_nf = 12
    hidden_nf = 32
    n_layers = 4
    #model = MVE(EGNN, in_node_nf=in_nf, in_edge_nf=0, hidden_nf=hidden_nf, n_layers=n_layers, multi_dec=True)
    model = ModelEnsemble(EGNN, num_models=num_ensembles, in_node_nf=in_nf, in_edge_nf=0, hidden_nf=hidden_nf, n_layers=n_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    al = ActiveLearning(max_uncertainty=6 ,num_ensembles=num_ensembles, in_nf=in_nf, hidden_nf=hidden_nf, n_layers=n_layers, model=model)
    #al.run_simulation(1000, show_traj=True)
    #print(len(al.calc.get_uncertainty_samples()))

    al.improve_model(100, 100,run_idx=28, use_wandb=True, model_path=model_path)
