from datasets.md17_dataset import MD17Dataset
from uncertainty.ensemble import ModelEnsemble
from gnn.egnn import EGNN
import torch
import numpy as np

from datasets.helper import utils as qm9_utils

from openmmtorch import TorchForce


class ModelEnsemble(torch.nn.Module):
    def __init__(self, base_model_class, num_models, in_node_nf, in_edge_nf, hidden_nf, n_layers, device):
        super(ModelEnsemble, self).__init__()
        self.num_models = num_models
        self.models = torch.nn.ModuleList([base_model_class(in_node_nf= in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device) for _ in range(num_models)])
        
    def forward(self, x, h0, edges, edge_attr, node_mask, edge_mask, n_nodes):
        # Collect the outputs from all models
        
        stacked_outputs = torch.stack([model( x, h0, edges, edge_attr, node_mask, edge_mask, n_nodes) for model in self.models])
        ensemble_output = torch.mean(stacked_outputs, dim=0)
        ensemble_uncertainty = torch.std(stacked_outputs, dim=0) * 3
        return stacked_outputs, ensemble_output, ensemble_uncertainty


class Model(torch.nn.Module):
    def __init__(self, path:str, num_ensembles:int, in_nf:int, hidden_nf:int, n_layers:int, device:str)->None:
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = ModelEnsemble(EGNN, num_ensembles, in_node_nf=in_nf, in_edge_nf=0, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.atom_numbers = ['0', '1', '0', '0', '1', '3', '2', '0', '1', '0', '1', '0', '0', '0', '1', '3', '2', '0', '1', '0', '0', '0']
        self.ground_charges  = {-1: 0.0,
                                0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}
        
        self.dtype = torch.float32
        self.charge_power = 2
        self.charge_scale = max(self.ground_charges.values())
        self.uncertainty_samples = []

    def forward(self, positions):
        """The forward method returns the energy computed from positions.

        Parameters
        ----------
        positions : torch.Tensor with shape (nparticles,3)
           positions[i,k] is the position (in nanometers) of spatial dimension k of particle i

        Returns
        -------
        potential : torch.Scalar
           The potential energy (in kJ/mol)
        """
        atom_numbers = self.atom_numbers * positions.shape[0]
        atom_mask = torch.where(torch.tensor(self.atom_numbers) != -1, torch.tensor(1.0), torch.tensor(0.0))
        atom_numbers = np.array(atom_numbers)
        coordinates = np.array(self.coordinates)

        num_atoms = atom_numbers.shape[1]
        num_snapshots = atom_numbers.shape[0]
        num_types = len(set(self.atom_numbers))

        one_hot = torch.eye(num_types, dtype=torch.float)[torch.tensor(atom_numbers.flatten())].reshape(-1,num_atoms, num_types)
        charges = torch.tensor([self.ground_charges[atom] for atom in atom_numbers.flatten()]).reshape(-1, num_atoms)

        edge_mask = torch.eye(num_atoms, dtype=torch.float).unsqueeze(0).expand(num_snapshots, -1, -1)
        edge_mask = torch.where(edge_mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        edge_mask = edge_mask * atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        edge_mask = edge_mask.view(num_snapshots, -1)

        self.coordinates = torch.tensor(self.coordinates)
        coordinates = coordinates.unsqueeze(0)

        batch_size, n_nodes, _ = coordinates.size()
        atom_positions = coordinates.view(batch_size * n_nodes, -1).to(self.device, self.dtype)
        atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(self.device, self.dtype)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, -1).to(self.device, self.dtype)
        one_hot = one_hot.to(self.device, self.dtype)
        charges = charges.to(self.device, self.dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, self.charge_power, self.charge_scale, self.device)
        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)


        stacked_pred, pred, uncertainty = self.model.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        uncertainty_threshold = 0.5
        if uncertainty[0] > uncertainty_threshold:
            self.uncertainty_samples.append(coordinates[0])
        return pred


class ActiveLearning:
    
    def __init__(self, model_path:str, num_ensembles:int, in_nf:int, hidden_nf:int, n_layers:int) -> None:

        self.atom_numbers = [0, 1, 0, 0, 1, 3, 2, 0, 1, 0, 1, 0, 0, 0, 1, 3, 2, 0, 1, 0, 0, 0]
        self.ground_charges  = {-1: 0.0,
                                0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

        self.model = Model(model_path, num_ensembles, in_nf=in_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)


        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.coordinates = positions
        print(self.atom_numbers)
        atom_numbers = [self.atom_numbers] * positions.shape[0]
        print(positions.shape[0])
        print(atom_numbers)
        atom_mask = torch.where(torch.tensor(self.atom_numbers) != -1, torch.tensor(1.0), torch.tensor(0.0))
        atom_numbers = np.array(atom_numbers)

        print(atom_numbers.shape)
        coordinates = np.array(self.coordinates)
        
        num_atoms = atom_numbers.shape[1]
        num_snapshots = atom_numbers.shape[0]
        num_types = len(set(self.atom_numbers))

        one_hot = torch.eye(num_types, dtype=torch.float)[torch.tensor(atom_numbers.flatten())].reshape(-1,num_atoms, num_types)
        charges = torch.tensor([self.ground_charges[atom] for atom in atom_numbers.flatten()]).reshape(-1, num_atoms)

        edge_mask = torch.eye(num_atoms, dtype=torch.float).unsqueeze(0).expand(num_snapshots, -1, -1)
        edge_mask = torch.where(edge_mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        edge_mask = edge_mask * atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        edge_mask = edge_mask.view(num_snapshots, -1)

        self.coordinates = torch.tensor(self.coordinates)
        coordinates = coordinates.unsqueeze(0)

        batch_size, n_nodes, _ = coordinates.size()
        atom_positions = coordinates.view(batch_size * n_nodes, -1).to(self.device, self.dtype)
        atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(self.device, self.dtype)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, -1).to(self.device, self.dtype)
        one_hot = one_hot.to(self.device, self.dtype)
        charges = charges.to(self.device, self.dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, self.charge_power, self.charge_scale, self.device)
        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)

        module = torch.jit.script(self.model)
        module.save('model.pt')

    def run_simulation(self, steps:int)->np.array:

        for i in range(steps):
            pass


if __name__ == "__main__":
    model_path = "best_model.pt"
    num_ensembles = 2
    in_nf = 12
    hidden_nf = 16
    n_layers = 2
    al = ActiveLearning(model_path, num_ensembles, in_nf, hidden_nf, n_layers)
