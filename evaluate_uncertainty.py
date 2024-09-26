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

def setup_log_folder(name, timestamp):

    path = f"logs/{name}_{timestamp}"
    os.makedirs(path)
    return path

def setup_model_folder(name, timestamp):

    path = f"gnn/models/{name}_{timestamp}"
    os.makedirs(path)
    return path


parser = argparse.ArgumentParser(description='Script to evaluate models')
parser.add_argument('--uncertainty_method', type=str, default="SWAG", help='MVE | ENS | SWAG | EVI')
parser.add_argument('--num_samples', type=int, default=5, help='Number of independent training runs')
parser.add_argument('--swag_sample_size', type=int, default=5, help='Number of samples to evaluate SWAG')
parser.add_argument('--ensemble_size', type=int, default=3, help='Number of models to evaluate ENS')
parser.add_argument('--mve_warmup', type=int, default=0, help='Number of warmup steps for MVE')
parser.add_argument('--save_model', type=bool, default=True, help='Save model')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
parser.add_argument('--swag_start_epoch', type=int, default=7000, help='Epoch to start SWAG sampling')
parser.add_argument('--force_weight', type=float, default=5, help='Factor to weight the force loss')
parser.add_argument('--energy_weight', type=float, default=1, help='Factor to weight the energy loss')
parser.add_argument('--use_wandb', type=bool, default=True, help="set to True to log information about training to wandb")


args = parser.parse_args()
uncertainty_method = args.uncertainty_method
num_samples = args.num_samples
swag_sample_size = args.swag_sample_size
ensemble_size = args.ensemble_size
warmup_steps = args.mve_warmup
save_model = args.save_model
epochs = args.epochs
swag_start_epoch = args.swag_start_epoch
force_weight = args.force_weight
energy_weight = args.energy_weight
use_wandb = args.use_wandb

in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4


batch_size = 128
lr = 1e-3
patience = 5000
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

timestamp = time.strftime("%Y%m%d_%H%M%S")

if uncertainty_method == "MVE":
    name = "mve"
    log_path = setup_log_folder(name, timestamp)
    if save_model:
        base_model_path = setup_model_folder(name, timestamp)
    path = f"logs/{name}"
    for i in range(num_samples):
        if save_model:
            model_path = f"{base_model_path}/model_{i}.pt"
        mve = MVE(EGNN, multi_dec=True, out_features=1, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        mve.fit(epochs=epochs, warmup_steps=warmup_steps, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=patience, model_path=model_path, force_weight=force_weight, energy_weight=energy_weight)
        mve.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name=f"{log_path}/plot_{i}", csv_path=f"{log_path}/eval.csv", test_loader_out=testloader_out)

if uncertainty_method == "SWAG":
    name = f"swag{swag_sample_size}"
    log_path = setup_log_folder(name, timestamp)
    if save_model:
        base_model_path = setup_model_folder(name, timestamp)
    path = f"logs/{name}"
    for i in range(num_samples):
        if save_model:
            model_path = f"{base_model_path}/model_{i}.pt"
        swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device, sample_size = swag_sample_size)
        swag.fit(epochs=epochs, swag_start_epoch=swag_start_epoch, swag_freq=1, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=patience, model_path=model_path, force_weight=force_weight, energy_weight=energy_weight)
        swag.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name=f"{log_path}/plot_{i}", csv_path=f"{log_path}/eval.csv", test_loader_out=testloader_out)

if uncertainty_method == "ENS":
    name = f"ensemble{ensemble_size}"
    log_path = setup_log_folder(name, timestamp)
    if save_model:
        base_model_path = setup_model_folder(name, timestamp)
    path = f"logs/{name}"
    for i in range(num_samples):
        if save_model:
            model_path = f"{base_model_path}/model_{i}.pt"
        ens = ModelEnsemble(EGNN, num_models=ensemble_size, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        ens.fit(epochs=epochs, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=patience, model_path=model_path, force_weight=force_weight, energy_weight=energy_weight)
        ens.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name=f"{log_path}/plot_{i}", csv_path=f"{log_path}/eval.csv", test_loader_out=testloader_out)

if uncertainty_method == "EVI":
    name = "evi"
    log_path = setup_log_folder(name, timestamp)
    if save_model:
        base_model_path = setup_model_folder(name, timestamp)
    path = f"logs/{name}"
    for i in range(num_samples):
        if save_model:
            model_path = f"{base_model_path}/model_{i}.pt"
        evi = EvidentialRegression(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers, device=device)
        evi.fit(epochs=epochs, train_loader=trainloader, valid_loader=validloader, device=device, dtype=torch.float32, use_wandb=use_wandb, patience=patience, model_path=model_path, force_weight=force_weight, energy_weight=energy_weight)
        evi.evaluate_all(testloader_in, device=device, dtype=torch.float32, plot_name=f"{log_path}/plot_{i}", csv_path=f"{log_path}/eval.csv", test_loader_out=testloader_out)



