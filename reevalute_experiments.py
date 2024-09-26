from uncertainty.swag import SWAG
from uncertainty.ensemble import ModelEnsemble
from uncertainty.mve import MVE
from uncertainty.evidential import EvidentialRegression
from gnn.egnn import EGNN
from datasets.md17_dataset import MD17Dataset

import torch
from torch import nn
import time
import argparse
import os



in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4

ensemble_size = 6
uncertainty_method = "ENS"
swag_sample_size = 5

model_dir = "gnn/models/ensemble6_1"
out_path = "logs/ensemble6_1_density_energy"

if not os.path.exists(out_path):
    os.makedirs(out_path)


batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"


testset_in = "datasets/files/validation_in"
testset_out = "datasets/files/validation_out"
testset_in = MD17Dataset(testset_in,subtract_self_energies=False, in_unit="kj/mol")
testset_out = MD17Dataset(testset_out,subtract_self_energies=False, in_unit="kj/mol")


# Create data loaders for train and validation sets
testloader_in = torch.utils.data.DataLoader(testset_in, batch_size=batch_size, shuffle=False)
testloader_out = torch.utils.data.DataLoader(testset_out, batch_size=batch_size, shuffle=False)


timestamp = time.strftime("%Y%m%d_%H%M%S")

for i, model_name in enumerate(os.listdir(model_dir)):
    model_path = f"{model_dir}/{model_name}"

    if uncertainty_method == "MVE":
        mve = MVE(EGNN, multi_dec=True, out_features=1, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        mve.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        mve.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name=f"{out_path}/plot_{i}", csv_path=f"{out_path}/eval.csv", test_loader_out=testloader_out, best_model_available=False, use_energy_uncertainty=True, use_force_uncertainty=False)

    if uncertainty_method == "SWAG":
    
        swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device, sample_size = swag_sample_size)
        swag.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        swag.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name=f"{out_path}/plot_{i}", csv_path=f"{out_path}/eval.csv", test_loader_out=testloader_out, best_model_available=False, use_energy_uncertainty=True, use_force_uncertainty=True)

    if uncertainty_method == "ENS":
        ens = ModelEnsemble(EGNN, num_models=ensemble_size, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        ens.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        ens.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name=f"{out_path}/plot_{i}", csv_path=f"{out_path}/eval.csv", test_loader_out=testloader_out, best_model_available=False, use_energy_uncertainty=True, use_force_uncertainty=True)

    if uncertainty_method == "EVI":
        
        evi = EvidentialRegression(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        evi.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        evi.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name=f"{out_path}/plot_{i}", csv_path=f"{out_path}/eval.csv", test_loader_out=testloader_out, best_model_available=False, use_energy_uncertainty=True, use_force_uncertainty=False)


