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


    use_wandb = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print("Training on device: " + str(device), flush=True)
    dtype = torch.float32

    epochs = 2000
    batch_size = 256
    lr = 1e-3
    min_lr = 1e-7
    log_interval = 100#int(2000/batch_size)
    
    force_weight = 1
    energy_weight = 1

    num_ensembles = 3
    in_node_nf = 12
    in_edge_nf = 0
    hidden_nf = 32
    n_layers = 4
    model = ModelEnsemble(EGNN, num_ensembles, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers).to(device)


    best_loss = np.inf

    qm9 = QM9()
    qm9.create(1,0)
    start = time.time()
    dataset = "datasets/files/ala_converged_10000_forces"
    model_path = "./gnn/models/ala_converged_10000_forces_7_ensemble.pt"
    trainset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=True, train_ratio=0.8)
    validset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=False, train_ratio=0.8)
    # Split the dataset into train and validation sets
    #trainset, validset = random_split(trainset, [int(0.8*len(trainset)), len(trainset) - int(0.8*len(trainset))])
    #trainset, validset = train_test_split(trainset, test_size=0.2, random_state=42)

    print(f"Loaded dataset in: {time.time() - start} seconds", flush=True)

    # Create data loaders for train and validation sets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

    charge_scale = qm9.charge_scale
    charge_power = 2

    print(charge_scale.item())
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-16)
    
    #optimizer = torch.optim.SGD(model.parameters(),lr=lr)
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
        losses_energy = []
        losses_force = []
        total_losses = []
        uncertainties = []
        model.train()
        start = time.time()

        for i,data in enumerate(trainloader):
            batch_size, n_nodes, _ = data['coordinates'].size()
            atom_positions = data['coordinates'].view(batch_size * n_nodes, -1).requires_grad_(True).to(device, dtype)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
            edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype)

            nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, device)



            nodes = nodes.view(batch_size * n_nodes, -1)

            # nodes = torch.cat([one_hot, charges], dim=1)
            edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
            label_energy = (data["energies"]).to(device, dtype)
            label_forces = (data["forces"]).view(batch_size * n_nodes, -1).to(device, dtype)


            stacked_pred, pred, uncertainty = model.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes, leave_out=(i%num_ensembles))
            grad_outputs = torch.ones_like(pred)
            grad_atom_positions = -torch.autograd.grad(pred, atom_positions, grad_outputs=grad_outputs, create_graph=True)[0]
            stacked_label_energy = label_energy.repeat(len(stacked_pred),1)
            
            stacked_loss_energy = loss_fn(stacked_pred, stacked_label_energy)
            loss_energy = loss_fn(pred, label_energy)
            loss_force = loss_fn(grad_atom_positions, label_forces)
            total_loss = force_weight*loss_force + energy_weight*stacked_loss_energy

            optimizer.zero_grad()
            total_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            


            losses_energy.append(loss_energy.item())
            losses_force.append(loss_force.item())
            total_losses.append(total_loss.item())

            uncertainty = torch.mean(uncertainty)
            uncertainties.append(uncertainty.item())    
            
            

            if (i+1) % log_interval == 0:
                print(f"Epoch {epoch}, Batch {i+1}/{len(trainloader)}, Loss: {loss_energy.item()}, Uncertainty: {uncertainty.item()}", flush=True)

        train_time = time.time() - start
        model.eval()
        valid_losses_energy = []
        valid_losses_force = []
        valid_losses_total = []
        valid_uncertainties = []
        num_in_interval = 0
        total_preds = 0
        start = time.time()
        #with torch.no_grad():
        for i, data in enumerate(validloader):
            batch_size, n_nodes, _ = data['coordinates'].size()
            atom_positions = data['coordinates'].view(batch_size * n_nodes, -1).requires_grad_(True).to(device, dtype)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
            edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype)
            nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, device)
            nodes = nodes.view(batch_size * n_nodes, -1)
            edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
            label_energy = (data["energies"]).to(device, dtype)
            label_forces = (data["forces"]).view(batch_size * n_nodes, -1).to(device, dtype)


            stacked_pred, pred, uncertainty = model.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            grad_outputs = torch.ones_like(pred)
            grad_atom_positions = -torch.autograd.grad(pred, atom_positions, grad_outputs=grad_outputs, create_graph=True)[0]

            mean_loss_energy = loss_fn(pred, label_energy)
            mean_loss_force = loss_fn(grad_atom_positions, label_forces)
            total_loss = force_weight*mean_loss_force + energy_weight*mean_loss_energy

            valid_losses_energy.append(mean_loss_energy.item())
            valid_losses_force.append(mean_loss_force.item())
            valid_losses_total.append(total_loss.item())

            num_in_interval += torch.sum(torch.abs(pred - label_energy) <= uncertainty/2)
            total_preds += pred.size(0)
            uncertainty = torch.mean(uncertainty)
            valid_uncertainties.append(uncertainty.item())

        lr_before = optimizer.param_groups[0]['lr']
        scheduler.step(np.array(valid_losses_total).mean())
        lr_after = optimizer.param_groups[0]['lr']

        if lr_before != lr_after:
            print(f"Learning rate changed to: {lr_after}", flush=True)

        if lr_after < min_lr:
            print(f"Learning rate is below minimum, stopping training")
            break
        if np.array(valid_losses_energy).mean() < best_loss:
            best_loss = np.array(valid_losses_energy).mean()
            torch.save(model.state_dict(), model_path)

        val_time = time.time() - start
        print("", flush=True)
        print(f"Training and Validation Results of Epoch {epoch}:", flush=True)
        print("================================")
        print(f"Training Loss Energy: {np.array(losses_energy).mean()}, Training Loss Force: {np.array(losses_force).mean()}, Training Uncertainty: {np.array(uncertainties).mean()}, time: {train_time}", flush=True)
        print(f"Validation Loss Energy: {np.array(valid_losses_energy).mean()}, Validation Loss Force: {np.array(valid_losses_force).mean()},Validation Uncertainty: {np.array(valid_uncertainties).mean()}, time: {val_time}", flush=True)
        print(f"Number of predictions within uncertainty interval: {num_in_interval}/{total_preds} ({num_in_interval/total_preds*100:.2f}%)", flush=True)
        print("", flush=True)

        if use_wandb:
            wandb.log({
                "train_loss_energy": np.array(losses_energy).mean(),
                "train_uncertainty": np.array(uncertainties).mean(),
                "train_loss_force": np.array(losses_force).mean(),
                "train_loss_total": np.array(total_losses).mean(),
                "valid_loss_energy": np.array(valid_losses_energy).mean(),
                "valid_uncertainty": np.array(valid_uncertainties).mean(),
                "valid_loss_force": np.array(valid_losses_force).mean(),
                "valid_loss_total": np.array(valid_losses_total).mean(),
                "in_interval": num_in_interval/total_preds*100,
                "lr" : lr_after 
            })
    if use_wandb:
        wandb.finish()
    
