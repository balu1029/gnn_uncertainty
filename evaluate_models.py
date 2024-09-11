from uncertainty.swag import SWAG
from uncertainty.ensemble import ModelEnsemble
from uncertainty.mve import MVE
from gnn.egnn import EGNN
from datasets.md17_dataset import MD17Dataset

import torch
from torch import nn
import time
import argparse
import os

def setup_log_folder(name):
    i = 0
    while os.path.exists(f"logs/{name}_{i}"):
        i += 1
        name = f"logs/{name}_{i}"
        os.makedirs(name)
    return name

def setup_model_folder(name):
    i = 0
    while os.path.exists(f"gnn/models/{name}_{i}"):
        i += 1
        name = f"gnn/models/{name}_{i}"
        os.makedirs(name)
    return name


parser = argparse.ArgumentParser(description='Script to evaluate models')
parser.add_argument('--uncertainty_method', type=str, default="ENS", help='MVE | ENS | SWAG')
parser.add_argument('--num_samples', type=int, default=5, help='Number of independent training runs')
parser.add_argument('--swag_sample_size', type=int, default=5, help='Number of samples to evaluate SWAG')
parser.add_argument('--ensemble_size', type=int, default=3, help='Number of models to evaluate ENS')
parser.add_argument('--mve_warmup', type=int, default=0, help='Number of warmup steps for MVE')
parser.add_argument('--save_model', type=bool, default=False, help='Save model')


args = parser.parse_args()
uncertainty_method = args.uncertainty_method
num_samples = args.num_samples
swag_sample_size = args.swag_sample_size
ensemble_size = args.ensemble_size
warmup_steps = args.mve_warmup
save_model = args.save_model

in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4
use_wandb = True
force_weight = 10

batch_size = 128
lr = 1e-3
epochs = 10000
patience = 1500
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

swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
ens = ModelEnsemble(EGNN, num_models=ensemble_size, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
mve = MVE(EGNN, multi_dec=True, out_features=1, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)


model_path = None

if uncertainty_method == "MVE":
    name = "mve"
    log_path = setup_log_folder(name)
    model_path = setup_model_folder(name)
    path = f"logs/{name}/"
    for i in range(num_samples):
        if save_model:
            model_path = f"{model_path}/model_{i}.pt"
        mve = MVE(EGNN, multi_dec=True, out_features=1, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        mve.fit(epochs=epochs, warmup_steps=warmup_steps, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=patience, model_path=model_path, force_weight=force_weight)
        mve.evaluate_all(validloader, device=device, dtype=torch.float32, plot_name=f"{log_path}/plot_{i}", csv_path=f"{log_path}/eval.csv")

if uncertainty_method == "SWAG":
    name = f"swag{swag_sample_size}"
    log_path = setup_log_folder(name)
    model_path = setup_model_folder(name)
    path = f"logs/{name}/"
    for i in range(num_samples):
        if save_model:
            model_path = f"{model_path}/model_{i}.pt"
        swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device, sample_size = swag_sample_size)
        swag.fit(epochs=epochs, swag_start_epoch=7000, swag_freq=1, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=patience, model_path=model_path, force_weight=force_weight)
        swag.evaluate_all(validloader, device=device, dtype=torch.float32, plot_name=f"{log_path}/plot_{i}", csv_path=f"{log_path}/eval.csv")

if uncertainty_method == "ENS":
    name = f"ensemble{ensemble_size}"
    log_path = setup_log_folder(name)
    model_path = setup_model_folder(name)
    path = f"logs/{name}/"
    for i in range(num_samples):
        if save_model:
            model_path = f"{model_path}/model_{i}.pt"
        ens = ModelEnsemble(EGNN, num_models=ensemble_size, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        ens.fit(epochs=epochs, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=patience, model_path=model_path, force_weight=force_weight)
        ens.evaluate_all(validloader, device=device, dtype=torch.float32, plot_name=f"{log_path}/plot_{i}", csv_path=f"{log_path}/eval.csv")


