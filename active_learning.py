import torch
import numpy as np

from ase.calculators.calculator import Calculator, all_properties
from ase.units import eV, fs, Angstrom, nm, kJ, mol
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io.trajectory import Trajectory
from ase.io import read
from ase.visualize import view

from uncertainty.ensemble import ModelEnsemble
from gnn.egnn import EGNN
from datasets.helper import utils as qm9_utils
from datasets.helper.energy_calculation import OpenMMEnergyCalculation





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
        self.charge_scale = max(self.ground_charges.values())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_uncertainty = 10
        self.uncertainty_samples = []
        self.energy_unit = eV
        self.force_unit = eV / nm
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
        atom_positions = torch.tensor(coordinates.view(batch_size * n_nodes, -1),requires_grad=True).to(self.device, self.dtype)
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
        print("Uncertainty: ", uncertainty.item())
        print("Energy:      ", energy.item())
        print()
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
        model = ModelEnsemble(EGNN, num_ensembles, in_node_nf=12, in_edge_nf=0, hidden_nf=16, n_layers=2)
        model.load_state_dict(torch.load(model_path, self.device))

        self.calc = ALCalculator(model=model)
        self.atoms = read(molecule_path)
        self.atoms.set_calculator(self.calc)

        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature)

        self.dyn = VelocityVerlet(self.atoms, timestep=self.timestep)
        self.oracle = OpenMMEnergyCalculation()

    def run_simulation(self, steps:int, show_traj:bool=False)->np.array:

        with Trajectory('ala.traj', 'w', self.atoms) as traj:
            for i in range(steps):
                self.dyn.run(1)
                traj.write(self.atoms)
        samples = self.calc.get_uncertainty_samples()
        self.calc.reset_uncertainty_samples()

        print(self.oracle.calc_energy(samples))
        
        if show_traj:
            traj = Trajectory('ala.traj')
            view(traj)
        


if __name__ == "__main__":
    model_path = "gnn/models/ala_converged_1000000.pt"
    num_ensembles = 3
    in_nf = 12
    hidden_nf = 16
    n_layers = 2
    al = ActiveLearning(num_ensembles=num_ensembles, in_nf=in_nf, hidden_nf=hidden_nf, n_layers=n_layers, model_path=model_path)

    al.run_simulation(30,show_traj=True)
