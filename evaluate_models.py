from uncertainty.swag import SWAG
from uncertainty.ensemble import ModelEnsemble
from uncertainty.mve import MVE
from gnn.egnn import EGNN
from datasets.md17_dataset import MD17Dataset

import torch
from torch import nn
import time
import argparse


parser = argparse.ArgumentParser(description='Script to evaluate models')
parser.add_argument('--uncertainty_method', type=str, default="ENS", help='MVE | ENS | SWAG')


args = parser.parse_args()
uncertainty_method = args.uncertainty_method

in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4
use_wandb = True

batch_size = 128
lr = 1e-3
epochs = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"

num_samples = 5

dataset = "datasets/files/ala_converged_1000_forces"
testset = "datsets/files/ala_converged_validation"
model_path = None#"./gnn/models/ala_converged_1000000_forces_mve.pt"
trainset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=True, train_ratio=0.8)
validset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=False, train_ratio=0.8)
testset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol")


# Create data loaders for train and validation sets

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
ens = ModelEnsemble(EGNN, num_models=3, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
mve = MVE(EGNN, multi_dec=True, out_features=1, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)


if uncertainty_method == "MVE":
    for i in range(num_samples):
        mve = MVE(EGNN, multi_dec=True, out_features=1, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        mve.fit(epochs=epochs, warmup_steps=0, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=800, model_path=None)
        mve.evaluate_uncertainty(validloader, device=device, dtype=torch.float32, plot_path=None, csv_path="uncertainty/mve_uncertainty_eval.csv")
        mve.evaluate_model(validloader, device=device, dtype=torch.float32, plot_path=None, csv_path="uncertainty/mve_model_eval.csv")

if uncertainty_method == "SWAG":
    for i in range(num_samples):
        swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        swag.fit(epochs=epochs, swag_start_epoch=7000, swag_freq=1, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=800, model_path=None)
        swag.evaluate_uncertainty(testloader, device=device, dtype=torch.float32, plot_path=None, csv_path="uncertainty/swag_uncertainty_eval.csv")
        swag.evaluate_model(testloader, device=device, dtype=torch.float32, plot_path=None, csv_path="uncertainty/swag_model_eval.csv")

if uncertainty_method == "ENS":
    for i in range(num_samples):
        ens = ModelEnsemble(EGNN, num_models=3, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        ens.fit(epochs=epochs, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=800, model_path=None)
        ens.evaluate_uncertainty(testloader, device=device, dtype=torch.float32, plot_path=None, csv_path="uncertainty/ensemble_uncertainty_eval.csv")
        ens.evaluate_model(testloader, device=device, dtype=torch.float32, plot_path=None, csv_path="uncertainty/ensemble_model_eval.csv")
