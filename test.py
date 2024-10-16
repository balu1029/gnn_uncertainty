from uncertainty.swag import SWAG
from uncertainty.ensemble import ModelEnsemble
from uncertainty.mve import MVE
from uncertainty.evidential import EvidentialRegression
from gnn.egnn import EGNN
from datasets.md17_dataset import MD17Dataset

import torch
from torch import nn
import time


in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4
force_weight = 10
energy_weight = 1

batch_size = 32
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = "datasets/files/train_in"
testset_in = "datasets/files/validation_in"
testset_out = "datasets/files/validation_out"
model_path = None#"./gnn/models/ala_converged_1000000_forces_mve.pt"
trainset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=True, train_ratio=0.8)
validset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=False, train_ratio=0.8)
testset_in = MD17Dataset(testset_in,subtract_self_energies=False, in_unit="kj/mol")
testset_out = MD17Dataset(testset_out,subtract_self_energies=False, in_unit="kj/mol")


# Create data loaders for train and validation sets

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
testloader_in = torch.utils.data.DataLoader(testset_in, batch_size=batch_size, shuffle=False)
testloader_out = torch.utils.data.DataLoader(testset_out, batch_size=batch_size, shuffle=False)


# Create data loaders for train and validation sets

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
ens = ModelEnsemble(EGNN, num_models=3, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
mve = MVE(EGNN, multi_dec=True, out_features=1, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
evi = EvidentialRegression(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)

#ens.fit(epochs=10000, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=False, patience=800, model_path=None)
#ens.evaluate_model(validloader, device=device, dtype=torch.float32, save_path=None)

swag.fit(epochs=10000, swag_start_epoch=7000, swag_freq=1,train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=True, patience=800, force_weight=force_weight, energy_weight=energy_weight)

#swag.load_state_dict(torch.load("gnn/models/swag.pt", map_location=torch.device(device)))
#swag.evaluate_uncertainty(validloader, device=device, dtype=torch.float32)
#swag.evaluate_model(validloader, device="cpu", dtype=torch.float32, save_path=None)

#mve.fit(epochs=100, warmup_steps=0, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=True, patience=800, model_path=None)
#mve.load_state_dict(torch.load("gnn/models/ala_converged_1000_forces_mve_no_warmup.pt", map_location=torch.device('cpu')))
#mve.evaluate_uncertainty(validloader, device="cpu", dtype=torch.float32, save_path=None)
#mve.evaluate_model(validloader, device="cpu", dtype=torch.float32, save_path=None)

#evi.fit(epochs=1000, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=False, patience=800, log_interval=5, force_weight=0)
#evi.load_state_dict(torch.load("gnn/models/evidential.pt", map_location=torch.device('cpu')))
#evi.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name="logs/evi/plot", csv_path="logs/evi/evi_eval.csv", test_loader_out=testloader_out)
