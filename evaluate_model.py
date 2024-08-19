import torch
from gnn.egnn import EGNN
from uncertainty.ensemble import ModelEnsemble
from datasets.md17_dataset import MD17Dataset

from datasets.helper import utils as qm9_utils


class ModelEvaluator:
    def __init__(self, model_path="gnn/models/ala_converged_1000000.pt", num_ensemble = 3, **kwargs):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.model = ModelEnsemble(EGNN, num_models=num_ensemble, **kwargs).to(self.device)
        self.model.load_state_dict(torch.load(model_path, self.device))
        self._create_dataset()
        self.ground_charges  = {-1: 0.0,
                                0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}

        self.charge_scale = torch.tensor(max(self.ground_charges.values()) + 1)
        self.charge_power = 2

    
    def _create_dataset(self, dataset_path="datasets/files/ala_converged_1000000", subtract_self_energies=False, in_unit="eV"):
        self.dataset = MD17Dataset(dataset_path, subtract_self_energies=subtract_self_energies, in_unit=in_unit)

    def evaluate(self):
        # Assuming dataset is a torch.utils.data.Dataset object
        data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=False)
        total_samples = len(self.dataset)
        correct_predictions = 0
        loss_fn = torch.nn.L1Loss()
        losses = []
        uncertainties = []
        with torch.no_grad():
            for data in data_loader:
                batch_size, n_nodes, _ = data['coordinates'].size()
                atom_positions = data['coordinates'].view(batch_size * n_nodes, -1).to(self.device, self.dtype)
                atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(self.device, self.dtype)
                edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, -1).to(self.device, self.dtype)
                one_hot = data['one_hot'].to(self.device, self.dtype)
                charges = data['charges'].to(self.device, self.dtype)
                nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, self.device)
                nodes = nodes.view(batch_size * n_nodes, -1)
                edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)
                label = (data["energies"]).to(self.device, self.dtype)
                stacked_pred, pred, uncertainty = self.model(inputs)    

                uncertainties.append(uncertainty)

                loss = loss_fn(pred, labels)
                losses.append(loss.item())
        loss = sum(losses) / len(losses)
        uncertainty = sum(uncertainties) / len(uncertainties)
        return loss, uncertainty
    
if __name__ == "__main__":
    model_path = "gnn/models/ala_converged_1000000.pt"
    num_ensemble = 3
    in_nf = 12
    in_edge_nf = 0
    hidden_nf = 16
    n_layers = 2
    evaluator = ModelEvaluator(num_ensemble=num_ensemble, in_node_nf=in_nf, in_edge_nf = in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, model_path=model_path)
    loss, uncertainty = evaluator.evaluate()
    print(f"Loss: {loss}, Uncertainty: {uncertainty}")