from uncertainty.mve import MVE
from uncertainty.ensemble import ModelEnsemble
from gnn.egnn import EGNN
from datasets.md17_dataset import MD17Dataset

import torch

num_ensembles = 3
in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mve = MVE(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers)
ens = ModelEnsemble(EGNN, num_ensembles, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers)

state_dict_path = "gnn/models/ala_converged_1000000_even_larger.pt"
#state_dict_path = "gnn/models/ala_converged_1000000_forces_mve.pt"
ens.load_state_dict(torch.load(state_dict_path, map_location=device))

test_dataset = MD17Dataset("datasets/files/ala_converged_validation",subtract_self_energies=False)

dataset = "datasets/files/ala_converged_forces_1000"
#test_dataset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=True, train_ratio=0.8)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

ens.evaluate_uncertainty(test_loader, device=device, dtype=torch.float32)