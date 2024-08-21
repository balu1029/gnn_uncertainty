from uncertainty.ensemble import ModelEnsemble
from gnn.egnn import EGNN
from datasets.qm9 import QM9
from datasets.md_dataset import MDDataset
from datasets.md17_dataset import MD17Dataset
import torch
from torch import nn
from datasets.helper import utils as qm9_utils
import numpy as np
import time

from torch.utils.data import random_split

import wandb

from torchsummary import summary
from sklearn.model_selection import train_test_split


if __name__ == "__main__":


    use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device: " + str(device), flush=True)
    dtype = torch.float32

    epochs = 20
    batch_size = 256
    lr = 1e-3
    min_lr = 1e-7
    log_interval = 100
    
    force_weight = 1
    energy_weight = 1

    num_ensembles = 3
    in_node_nf = 12
    in_edge_nf = 0
    hidden_nf = 16
    n_layers = 3


    model = ModelEnsemble(EGNN, num_ensembles, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers).to(device)


    best_loss = np.inf


    start = time.time()
    dataset = "datasets/files/ala_converged_forces_1000"
    model_path = None#"./gnn/models/ala_converged_10000_forces_7_ensemble.pt"
    trainset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=True, train_ratio=0.8)
    validset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=False, train_ratio=0.8)

    print(f"Loaded dataset in: {time.time() - start} seconds", flush=True)

    # Create data loaders for train and validation sets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-16)
    
    factor = 0.1
    patience = 200
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

    total_params = sum(p.numel() for p in model.parameters())
    print("Number of trainable parameters: " + str(total_params))
    lr_before = 0
    lr_after = 0

    # start a new wandb run to track this script
    if use_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="GNN-Uncertainty",

            # track hyperparameters and run metadata
            config={
            "name": "alaninedipeptide",
            "learning_rate_start": lr,
            "layers": n_layers,
            "hidden_nf": hidden_nf,
            "scheduler": type(scheduler).__name__,
            "optimizer": type(optimizer).__name__,
            "patience": patience,
            "factor": factor,
            "dataset": dataset,
            "epochs": epochs,
            "num_ensembles": num_ensembles,
            "batch_size": batch_size,
            "in_node_nf" : in_node_nf,
            "in_edge_nf" : in_edge_nf,
            "loss_fn" : type(loss_fn).__name__,
            "model_checkpoint": model_path,
            }
        )

    for epoch in range(epochs):

        start = time.time()
        model.train_epoch(trainloader, optimizer, loss_fn, epoch, device, dtype, force_weight=force_weight, energy_weight=energy_weight, log_interval=log_interval, num_ensembles=num_ensembles)
        train_time = time.time() - start
        
        start = time.time()
        model.valid_epoch(validloader, loss_fn, device, dtype, force_weight=force_weight, energy_weight=energy_weight)
        val_time = time.time() - start

        train_losses_energy, train_losses_force, train_losses_total, train_uncertainties, valid_losses_energy, valid_losses_force, valid_losses_total, valid_uncertainties, num_in_interval, total_preds = model.pop_metrics()


        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step(np.array(valid_losses_total).mean())
        lr_after = optimizer.param_groups[0]['lr']

        if lr_before != lr_after:
            print(f"Learning rate changed to: {lr_after}", flush=True)

        if lr_after < min_lr:
            print(f"Learning rate is below minimum, stopping training")
            break
        
        if model_path is not None:
            if np.array(valid_losses_energy).mean() < best_loss:
                best_loss = np.array(valid_losses_energy).mean()
                torch.save(model.state_dict(), model_path)

        
        print("", flush=True)
        print(f"Training and Validation Results of Epoch {epoch}:", flush=True)
        print("================================")
        print(f"Training Loss Energy: {np.array(train_losses_energy).mean()}, Training Loss Force: {np.array(train_losses_force).mean()}, Training Uncertainty: {np.array(train_uncertainties).mean()}, time: {train_time}", flush=True)
        print(f"Validation Loss Energy: {np.array(valid_losses_energy).mean()}, Validation Loss Force: {np.array(valid_losses_force).mean()},Validation Uncertainty: {np.array(valid_uncertainties).mean()}, time: {val_time}", flush=True)
        print(f"Number of predictions within uncertainty interval: {num_in_interval}/{total_preds} ({num_in_interval/total_preds*100:.2f}%)", flush=True)
        print("", flush=True)

        if use_wandb:
            wandb.log({
                "train_loss_energy": np.array(train_losses_energy).mean(),
                "train_uncertainty": np.array(train_uncertainties).mean(),
                "train_loss_force": np.array(train_losses_force).mean(),
                "train_loss_total": np.array(train_losses_total).mean(),
                "valid_loss_energy": np.array(valid_losses_energy).mean(),
                "valid_uncertainty": np.array(valid_uncertainties).mean(),
                "valid_loss_force": np.array(valid_losses_force).mean(),
                "valid_loss_total": np.array(valid_losses_total).mean(),
                "in_interval": num_in_interval/total_preds*100,
                "lr" : lr_before 
            })
    if use_wandb:
        wandb.finish()
    
