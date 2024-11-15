from gnn.egnn import EGNN
from uncertainty.base_uncertainty import BaseUncertainty
import torch
from torch import nn
from datasets.helper import utils as qm9_utils

import wandb
import numpy as np  
import time
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



class MVE(BaseUncertainty):
    def __init__(self, base_model_class, multi_dec = True, *args, **kwargs):
        super(MVE, self).__init__()
        self.model = base_model_class(*args, **kwargs, multi_dec=multi_dec)


        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_losses_total = []
        self.train_time = 0

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_losses_total = []
        self.valid_time = 0


    def fit(self, epochs, train_loader, valid_loader, device, dtype, model_path="gnn/models/mve.pt", use_wandb=False, warmup_steps=0, force_weight=1.0, energy_weight=1.0, log_interval=100, patience=200, factor=0.1, lr=1e-3, min_lr=1e-6, additional_logs=None, best_on_train=False): 

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-16)   
        criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)

        if warmup_steps > 0:
            self.warmup(train_loader, optimizer, criterion, device, dtype, epochs=warmup_steps, force_weight=force_weight, energy_weight=energy_weight, log_interval=log_interval)
        
        if use_wandb:
            self.init_wandb(scheduler,criterion,optimizer,model_path,train_loader,valid_loader,epochs,lr,patience,factor,force_weight,energy_weight)

        best_valid_loss = np.inf

        for epoch in range(epochs):
            self.train_epoch(train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch, device=device, dtype=dtype, force_weight=force_weight, energy_weight=energy_weight, log_interval=log_interval)
            self.valid_epoch(valid_loader=valid_loader, criterion=criterion, device=device, dtype=dtype, force_weight=force_weight, energy_weight=energy_weight)
            self.epoch_summary(epoch, use_wandb=use_wandb, lr=optimizer.param_groups[0]['lr'], additional_logs=additional_logs)

            if best_on_train:
                if np.array(self.train_losses_total).mean() < best_valid_loss:
                    best_valid_loss = np.array(self.train_losses_total).mean()
                    if model_path is not None:
                        torch.save(self.state_dict(), model_path)
                    self.best_model = self.state_dict()
            else:
                if np.array(self.valid_losses_total).mean() < best_valid_loss:
                    best_valid_loss = np.array(self.valid_losses_total).mean()
                    if model_path is not None:
                        torch.save(self.state_dict(), model_path)
                    self.best_model = self.state_dict()

            self.lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(np.array(self.valid_losses_total).mean())
            self.lr_after = optimizer.param_groups[0]['lr']
            self.drop_metrics()

        if use_wandb:
            wandb.finish()

    def warmup(self, train_loader, optimizer, criterion, device, dtype, epochs = 50, force_weight=1.0, energy_weight=1.0, log_interval=100, num_ensembles=3):
        print("",flush=True)
        print("Warmup phase started", flush=True)
        self.train()
        for epoch in range(epochs):
            start = time.time()
            for i,data in enumerate(train_loader):

                atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

                mean_energy, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
                
                loss_energy = criterion(mean_energy, label_energy)
                loss_force = criterion(mean_force, label_forces)
                total_loss = force_weight*loss_force + energy_weight*loss_energy

                optimizer.zero_grad()
                total_loss.backward()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                

                self.train_losses_energy.append(loss_energy.item()*train_loader.dataset.std_energy)
                self.train_losses_force.append(loss_force.item()*train_loader.dataset.std_energy)
                self.train_losses_total.append(total_loss.item())

                
                if (i+1) % log_interval == 0:
                    print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss_energy.item()}, Uncertainty: {uncertainty.item()}", flush=True) 
            self.train_time = time.time() - start
            lr = optimizer.param_groups[0]['lr']
            self.epoch_summary(f"Warmup-{epoch}", use_wandb=False, lr=lr)
        print("Warmup phase finished", flush=True)

    def train_epoch(self, train_loader, optimizer, criterion, epoch, device, dtype, force_weight=1.0, energy_weight=1.0, log_interval=100, num_ensembles=3):
        start = time.time()
        self.train()
        for i,data in enumerate(train_loader):

            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            mean_energy, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            
            loss_energy = criterion(mean_energy, label_energy)
            loss_force = criterion(mean_force, label_forces)
            total_loss = 0.5 * torch.mean(torch.log(uncertainty) + (loss_force*force_weight)/uncertainty) + loss_energy*energy_weight

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
            optimizer.step()
            

            self.train_losses_energy.append(loss_energy.item()*train_loader.dataset.std_energy)
            self.train_losses_force.append(loss_force.item()*train_loader.dataset.std_energy)
            self.train_losses_total.append(total_loss.item())
            
            if (i+1) % log_interval == 0:
                print(f"Epoch {epoch}, Batch {i+1}/{len(train_loader)}, Loss: {loss_energy.item()}, Uncertainty: {torch.mean(uncertainty).item()}", flush=True)
        
        self.train_time = time.time() - start


    def valid_epoch(self, valid_loader, criterion, device, dtype, force_weight=1.0, energy_weight=1.0):
        start = time.time()
        self.eval()
        for i,data in enumerate(valid_loader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            mean_energy, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)

 
            loss_energy = criterion(mean_energy, label_energy)
            loss_force = criterion(mean_force, label_forces)
            total_loss = force_weight * 0.5 * torch.mean(torch.log(uncertainty) + loss_force/uncertainty) + loss_energy*energy_weight

            self.valid_losses_energy.append(loss_energy.item()*valid_loader.dataset.std_energy)
            self.valid_losses_force.append(loss_force.item()*valid_loader.dataset.std_energy)
            self.valid_losses_total.append(total_loss.item())

        self.valid_time = time.time() - start


    def predict(self, x, use_force_uncertainty=False, *args, **kwargs):
        self.eval()
        energy, forces, uncertainty = self.forward(x=x, *args, **kwargs)
        return energy, forces, uncertainty*self.uncertainty_slope + self.uncertainty_bias


    def forward(self, x, *args, **kwargs):
        output = self.model.forward(x=x, *args, **kwargs)
        energy = output[:,0].squeeze()
        variance = output[:,1].squeeze()
        grad_output = torch.ones_like(energy)
        force = -torch.autograd.grad(outputs=energy, inputs=x, grad_outputs=grad_output, create_graph=True)[0]
        return energy, force, variance
    

    def drop_metrics(self):
        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_losses_total = []
        self.train_time = 0

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_losses_total = []
        self.valid_time = 0

    
    def epoch_summary(self, epoch, additional_logs=None, use_wandb=False, lr=None):
        attributes = [
            'train_losses_energy',
            'train_losses_force',
            'train_losses_total',
            'valid_losses_energy',
            'valid_losses_force',
            'valid_losses_total'
        ]

        for attr in attributes:
            if getattr(self, attr) == []:
                setattr(self, attr, [0])
                
        print("", flush=True)
        print(f"Training and Validation Results of Epoch {epoch}:", flush=True)
        print("================================")
        print(f"Training Loss Energy: {np.array(self.train_losses_energy).mean()}, Training Loss Force: {np.array(self.train_losses_force).mean()}, time: {self.train_time}", flush=True)
        if len(self.valid_losses_energy) > 0:
            print(f"Validation Loss Energy: {np.array(self.valid_losses_energy).mean()}, Validation Loss Force: {np.array(self.valid_losses_force).mean()}, time: {self.valid_time}", flush=True)
        print("", flush=True)

        logs = {"train_loss_energy": np.array(self.train_losses_energy).mean(),
                "train_loss_force": np.array(self.train_losses_force).mean(),
                "train_loss_total": np.array(self.train_losses_total).mean(),
                "valid_loss_energy": np.array(self.valid_losses_energy).mean(),
                "valid_loss_force": np.array(self.valid_losses_force).mean(),
                "valid_loss_total": np.array(self.valid_losses_total).mean(),
                "lr" : lr}
        
        if additional_logs is not None:
            logs.update(additional_logs)

        if use_wandb:
            wandb.log(logs)

    def init_wandb(self, scheduler, criterion, optimizer, model_path, train_loader, valid_loader, epochs, lr, patience, factor, force_weight, energy_weight):
        wandb.init(
                # set the wandb project where this run will be logged
                project="GNN-Uncertainty-MVE",
                name=self.wandb_name,

                # track hyperparameters and run metadata
                config={
                "name": "alaninedipeptide",
                "learning_rate_start": lr,
                "layers": self.model.n_layers,
                "hidden_nf": self.model.hidden_nf,
                "scheduler": type(scheduler).__name__,
                "optimizer": type(optimizer).__name__,
                "patience": patience,
                "factor": factor,
                "dataset": len(train_loader.dataset)+len(valid_loader.dataset),
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "in_node_nf" : self.model.in_node_nf,
                "in_edge_nf" : self.model.in_edge_nf,
                "loss_fn" : type(criterion).__name__,
                "model_checkpoint": model_path,
                "force_weight": force_weight,
                "energy_weight": energy_weight
                })


if __name__ == "__main__":
    mve = MVE()