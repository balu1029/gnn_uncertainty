from uncertainty.mve import MVE
from uncertainty.ensemble import ModelEnsemble
from uncertainty.swag import SWAG
from gnn.egnn import EGNN
from datasets.md17_dataset import MD17Dataset

import torch

num_ensembles = 3
in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers)
mve = MVE(EGNN, multi_dec=True, out_features=1, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers)
ens = ModelEnsemble(EGNN, num_ensembles, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers)

state_dict_path = "gnn/models/swag.pt"
#state_dict_path = "al/run9/models/model_18.pt"
#state_dict_path = "gnn/models/ala_converged_1000000_forces_mve.pt"
weights = torch.load(state_dict_path, map_location=device)
swag.load_state_dict(torch.load(state_dict_path, map_location=device))
state_dict_path = "gnn/models/ala_converged_1000_forces_mve_no_warmup.pt"
mve.load_state_dict(torch.load(state_dict_path, map_location=device))
state_dict_path = "gnn/models/ensemble.pt"
ens.load_state_dict(torch.load(state_dict_path, map_location=device))



test_dataset = MD17Dataset("datasets/files/ala_converged_validation",subtract_self_energies=False)

#dataset = "datasets/files/ala_converged_forces_1000"
#test_dataset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=False, train_ratio=0.8)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

#swag.evaluate_uncertainty(test_loader, device=device, dtype=torch.float32)
#swag.evaluate_model(test_loader, device=device, dtype=torch.float32, save_path="gnn/swag_eval.png")
#mve.evaluate_model(test_loader, device=device, dtype=torch.float32, save_path="gnn/mve_eval.png")
ens.evaluate_model(test_loader, device=device, dtype=torch.float32, save_path="gnn/ensemble_eval.png")

#mve.evaluate_uncertainty(test_loader, device=device, dtype=torch.float32, save_path="gnn/mve_uncertainty.png")
#swag.evaluate_uncertainty(test_loader, device=device, dtype=torch.float32, save_path="gnn/swag_uncertainty.png")
ens.evaluate_uncertainty(test_loader, device=device, dtype=torch.float32, save_path="gnn/ensemble_uncertainty.png")
