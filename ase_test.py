import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_properties
from ase.units import eV, fs

from uncertainty.ensemble import ModelEnsemble
from gnn.egnn import EGNN
from datasets.helper import utils as qm9_utils

class MLCalculator(Calculator):
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

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_properties):
        super().calculate(atoms, properties, system_changes)
        
        # Get atomic positions as a numpy array
        positions = atoms.get_positions() * 0.1

        


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
        

        # Convert energy to the appropriate ASE units (eV)
        self.results['energy'] = energy.item() * eV

        # Compute forces by taking the gradient of energy w.r.t. positions
        energy.backward()
        forces = -atom_positions.grad.numpy()
        print("Uncertainty: ", uncertainty.item())
        print("Energy:      ", energy.item())
        print()
        # Store the forces in the results dictionary
        self.results['forces'] = forces


# Example of using the MLCalculator with an MD simulation
if __name__ == "__main__":
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase.io.trajectory import Trajectory
    from ase.io import read
    from ase.visualize import view
    

    # Define a simple atomic system
    # Read atoms from an XYZ file
    atoms = read("datasets/files/alaninedipeptide/xyz/ala_single.xyz")


    # Instantiate the model
    num_ensembles = 3
    model = ModelEnsemble(EGNN, num_ensembles, in_node_nf=12, in_edge_nf=0, hidden_nf=16, n_layers=2)
    model.load_state_dict(torch.load("gnn/models/ala_converged_10000.pt", map_location=torch.device('cpu')))

    # Create the custom calculator
    calc = MLCalculator(model=model)

    # Set the calculator to the atoms
    atoms.set_calculator(calc)

    # Set initial velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)

    # Define MD integrator
    dyn = VelocityVerlet(atoms, timestep=1.0 * fs)



    # Run the MD simulation

    
    with Trajectory('ala.traj', 'w', atoms) as traj:
        for i in range(100):
            dyn.run(1)  # Run for 1 step
            traj.write(atoms)
            #print(f"Step {i + 1}: Energy = {atoms.get_potential_energy()}")
    traj = Trajectory('ala.traj')
    #view(traj)
