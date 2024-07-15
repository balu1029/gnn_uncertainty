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

import wandb

from torchsummary import summary
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    #device = torch.device("cpu")
    print("Training on device: " + str(device), flush=True)
    dtype = torch.float32

    epochs = 100
    batch_size = 1028
    lr = 1e-4
    min_lr = 1e-7
    log_interval = 100#int(2000/batch_size)

    num_ensembles = 10

    layers = 2
    hidden_nf = 64
    model = ModelEnsemble(EGNN, num_ensembles, in_node_nf=12, in_edge_nf=0, hidden_nf=64, n_layers=2).to(device)


    model_path = None #"./gnn/models/md17.pt"
    data_path = "./datasets/files/md17/"
    save_model = True
    save_path = "gnn/models/md17_ensemble.pt"

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    best_loss = np.inf

    qm9 = QM9()
    qm9.create(1,0)
    #trainset = MDDataset("datasets/files/alaninedipeptide")
    start = time.time()
    trainset = MD17Dataset(foldername=data_path, seed=42)
    # Split the dataset into train and validation sets
    trainset, validset = train_test_split(trainset, test_size=0.2, random_state=42)
    print(f"Loaded dataset in: {time.time() - start} seconds", flush=True)
    

    # Create data loaders for train and validation sets
    start = time.time()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
    print(f"Created data loaders in: {time.time() - start} seconds", flush=True)

    charge_scale = qm9.charge_scale
    charge_power = 2

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-16)
    #optimizer = torch.optim.SGD(model.parameters(),lr=lr)

    factor = 0.1
    patience = 5

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)


    wandb.init(
        # set the wandb project where this run will be logged
        project="GNN-Uncertainty",

        # track hyperparameters and run metadata
        config={
        "learning_rate_start": lr,
        "layers": layers,
        "hidden_nf": hidden_nf,
        "scheduler": type(scheduler).__name__,
        "optimizer": type(optimizer).__name__,
        "patience": patience,
        "factor": factor,
        "dataset": data_path,
        "epochs": epochs,
        }
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("Number of trainable parameters: " + str(total_params))
    lr_before = 0
    lr_after = 0
    for epoch in range(epochs):
        losses = []
        uncertainties = []
        model.train()
        start = time.time()
        for i,data in enumerate(trainloader):
            batch_size, n_nodes, _ = data['coordinates'].size()
            atom_positions = data['coordinates'].view(batch_size * n_nodes, -1).to(device, dtype)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
            edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype)

            nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, device)



            nodes = nodes.view(batch_size * n_nodes, -1)

            # nodes = torch.cat([one_hot, charges], dim=1)
            edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
            label = (data["energies"]).to(device, dtype)

            stacked_pred, pred, uncertainty = model.forward(leave_out=(i%num_ensembles), x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            stacked_label = label.repeat(stacked_pred.size(0), 1)
            loss = loss_fn(stacked_pred, stacked_label)
            loss.backward()
            mean_loss = loss_fn(pred, label)
            losses.append(mean_loss.item())

            uncertainty = torch.mean(uncertainty)
            uncertainties.append(uncertainty.item())    

            optimizer.step()
                        
            optimizer.zero_grad()

            

            if (i+1) % log_interval == 0:
                print(f"Epoch {epoch}, Batch {i+1}/{len(trainloader)}, Loss: {loss.item()}, Uncertainty: {uncertainty.item()}", flush=True)

        train_time = time.time() - start
        model.eval()
        valid_losses = []
        valid_uncertainties = []
        num_in_interval = 0
        total_preds = 0
        start = time.time()
        with torch.no_grad():
            for i, data in enumerate(validloader):
                batch_size, n_nodes, _ = data['coordinates'].size()
                atom_positions = data['coordinates'].view(batch_size * n_nodes, -1).to(device, dtype)
                atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
                edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = data['charges'].to(device, dtype)
                nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, device)
                nodes = nodes.view(batch_size * n_nodes, -1)
                edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
                label = (data["energies"]).to(device, dtype)


                stacked_pred, pred, uncertainty = model.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)


                stacked_label = label.repeat(stacked_pred.size(0), 1)
                loss = loss_fn(stacked_pred, stacked_label)
                mean_loss = loss_fn(pred, label)
                valid_losses.append(mean_loss.item())
                num_in_interval += torch.sum(torch.abs(pred - label) <= uncertainty/2)
                total_preds += pred.size(0)
                uncertainty = torch.mean(uncertainty)
                valid_uncertainties.append(uncertainty.item())

            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(np.array(valid_losses).mean())
            lr_after = optimizer.param_groups[0]['lr']

            if lr_before != lr_after:
                print(f"Learning rate changed to: {lr_after}", flush=True)

            if lr_after < min_lr:
                print(f"Learning rate is below minimum, stopping training")
                break
            if np.array(valid_losses).mean() < best_loss and save_model:
                best_loss = np.array(valid_losses).mean()
                torch.save(model.state_dict(), save_path)

        val_time = time.time() - start
        print("", flush=True)
        print(f"Training and Validation Results of Epoch {epoch}:", flush=True)
        print("================================")
        print(f"Training Loss: {np.array(losses).mean()}, Training Uncertainty: {np.array(uncertainties).mean()}, time: {train_time}", flush=True)
        print(f"Validation Loss: {np.array(valid_losses).mean()}, Validation Uncertainty: {np.array(valid_uncertainties).mean()}, time: {val_time}", flush=True)
        print(f"Number of predictions within uncertainty interval: {num_in_interval}/{total_preds} ({num_in_interval/total_preds*100:.2f}%)", flush=True)
        print("", flush=True)


        wandb.log({
            "train_loss": np.array(losses).mean(),
            "train_uncertainty": np.array(uncertainties).mean(),
            "valid_loss": np.array(valid_losses).mean(),
            "valid_uncertainty": np.array(valid_uncertainties).mean(),
            "in_interval": num_in_interval/total_preds*100,
        })

    wandb.finish()
    
