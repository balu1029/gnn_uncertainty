import torch
import numpy as np
import os
import shutil

from ase.calculators.calculator import Calculator, all_properties
from ase.units import eV, fs, Angstrom, nm, kJ, mol
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.visualize import view

from uncertainty.ensemble import ModelEnsemble
from gnn.egnn import EGNN
from datasets.helper import utils as qm9_utils
from datasets.helper.energy_calculation import OpenMMEnergyCalculation
from trainer import ModelTrainer



class ALCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()  # Set the model to evaluation mode

        self.atom_numbers  = [0, 1, 0, 0, 1, 3, 2, 0, 1, 0, 1, 0, 0, 0, 1, 3, 2, 0, 1, 0, 0, 0]
        self.ground_charges  = {-1: 0.0,
                                0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}
        
        self.dtype = torch.float32
        self.charge_power = 2
        self.charge_scale = torch.tensor(max(self.ground_charges.values())+1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_uncertainty = 10
        self.uncertainty_samples = []
        self.energy_unit = kJ / mol
        self.force_unit = self.energy_unit / Angstrom
        self.a_to_nm = 0.1

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
        stacked_pred, energy, uncertainty = self.model(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        
        if uncertainty.item() > self.max_uncertainty:
            self.uncertainty_samples.append(positions * self.a_to_nm)

        # Convert energy to the appropriate ASE units (eV)
        self.results['energy'] = energy.item() * self.energy_unit

        # Compute forces by taking the gradient of energy w.r.t. positions
        energy.backward()
        forces = -atom_positions.grad.numpy()
        #print("Uncertainty: ", uncertainty.item())
        #print("Energy:      ", energy.item())
        #print()
        # Store the forces in the results dictionary
        self.results['forces'] = forces * self.force_unit

    def get_uncertainty_samples(self):
        return np.array(self.uncertainty_samples)
    
    def reset_uncertainty_samples(self):
        self.uncertainty_samples = []   


class ActiveLearning:
    
    def __init__(self, model_path:str="gnn/models/ala_converged_10000.pt", num_ensembles:int=3, in_nf:int=12, hidden_nf:int=16, n_layers:int=2, molecule_path:str="datasets/files/ala_converged_1000000/start_pos.xyz") -> None:
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.temperature = 300
        self.timestep = 0.5 * fs

        # Instantiate the model
        model = ModelEnsemble(EGNN, num_ensembles, in_node_nf=in_nf, in_edge_nf=0, hidden_nf=hidden_nf, n_layers=n_layers)
        model.load_state_dict(torch.load(model_path, self.device))

        self.num_ensembles = num_ensembles

        self.calc = ALCalculator(model=model)
        self.atoms = read(molecule_path)
        self.atoms.set_calculator(self.calc)

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature)

        friction_coefficient = 0.01  # Friction coefficient in 1/fs
        self.dyn = Langevin(self.atoms, timestep=self.timestep, temperature_K=self.temperature, friction=friction_coefficient)
        self.oracle = OpenMMEnergyCalculation()
        self.trainer = ModelTrainer(self.calc.model, self.num_ensembles)

    def run_simulation(self, steps:int, show_traj:bool=False)->np.array:

        with Trajectory('ala.traj', 'w', self.atoms) as traj:
            for i in range(steps):
                self.dyn.run(1)
                traj.write(self.atoms)
       
        
        if show_traj:
            traj = Trajectory('ala.traj')
            view(traj, block=True)


    def improve_model(self, num_iter, steps_per_iter, run_idx=1):
        model_out_path = f"al/run{run_idx}/models/"
        data_out_path = f"al/run{run_idx}/data/"
        if not os.path.exists(f"al/run{run_idx}"):
            os.makedirs(f"al/run{run_idx}")
            os.makedirs(model_out_path)
            os.makedirs(data_out_path)
            # Copy the file to the directory
            shutil.copy2("al/run1/data/base.xyz", data_out_path)

        for i in range(num_iter):
            self.run_simulation(steps_per_iter, show_traj=False)

            samples = self.calc.get_uncertainty_samples()
            self.calc.reset_uncertainty_samples()
            energies = self.oracle.calc_energy(samples)   
            forces = self.oracle.calc_forces(samples)         
            
            if len(samples) > 0:
                print(f"Training model {i}. Added {len(samples)} samples to the dataset.")

                self.trainer.add_data(samples, energies, forces, f"{data_out_path}data_{i}.txt")

                self.trainer.train(num_epochs=2, learning_rate=1e-5, folder=data_out_path)
                
                torch.save(self.trainer.model.state_dict(), f"{model_out_path}model_{i}.pt")
                self.calc.model.load_state_dict(torch.load(f"{model_out_path}model_{i}.pt", self.device))
        


if __name__ == "__main__":
    model_path = "gnn/models/ala_converged_10000_forces.pt"
    #model_path = "al/run2/models/model_21.pt"
    #model_path = "gnn/models/ala_converged_1000000_even_larger.pt"
    
    num_ensembles = 3
    in_nf = 12
    hidden_nf = 32
    n_layers = 4
    al = ActiveLearning(num_ensembles=num_ensembles, in_nf=in_nf, hidden_nf=hidden_nf, n_layers=n_layers, model_path=model_path)
    #al.run_simulation(1000, show_traj=True)
    #print(len(al.calc.get_uncertainty_samples()))

    al.improve_model(50, 200,run_idx=5)
