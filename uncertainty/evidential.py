from uncertainty.base_uncertainty import BaseUncertainty

import torch
import torch.nn as nn
import numpy as np
import wandb
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EvidentialRegressionLoss(nn.Module):
    def __init__(self, coeff=1.0, omega=0.01, reduce=True, kl=False):
        """
        Initialize the loss function with optional regularization coefficient (coeff),
        omega for controlling the regularization term, and other settings.
        """
        super(EvidentialRegressionLoss, self).__init__()
        self.coeff = coeff
        self.omega = omega
        self.reduce = reduce
        self.kl = kl

    def NIG_NLL(self, y, gamma, v, alpha, beta):
        """
        Compute the Negative Log-Likelihood (NLL) for Normal Inverse Gamma (NIG) distribution.
        """
        twoBlambda = 2 * beta * (1 + v)
        
        nll = 0.5 * torch.log(np.pi / v) \
            - alpha * torch.log(twoBlambda) \
            + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + twoBlambda) \
            + torch.lgamma(alpha) \
            - torch.lgamma(alpha + 0.5)
        
        return torch.mean(nll) if self.reduce else nll

    def KL_NIG(self, mu1, v1, a1, b1, mu2, v2, a2, b2):
        """
        Compute the Kullback-Leibler (KL) divergence between two NIG distributions.
        """
        KL = 0.5 * (a1 - 1) / b1 * (v2 * torch.square(mu2 - mu1)) \
            + 0.5 * v2 / v1 \
            - 0.5 * torch.log(torch.abs(v2) / torch.abs(v1)) \
            - 0.5 + a2 * torch.log(b1 / b2) \
            - (torch.lgamma(a1) - torch.lgamma(a2)) \
            + (a1 - a2) * torch.digamma(a1) \
            - (b1 - b2) * a1 / b1
        return KL

    def NIG_Reg(self, y, gamma, v, alpha, beta):
        """
        Compute the regularization term based on error and evidence.
        """
        error = torch.abs(y - gamma)
        
        if self.kl:
            kl = self.KL_NIG(gamma, v, alpha, beta, gamma, self.omega, 1 + self.omega, beta)
            reg = error * kl
        else:
            evi = 2 * v + alpha
            reg = error * evi

        return torch.mean(reg) if self.reduce else reg

    def forward(self, y_true, evidential_output):
        """
        Forward pass to compute the total loss which includes both NLL and regularization.
        """
        gamma, v, alpha, beta = torch.split(evidential_output, 4, dim=-1)
        loss_nll = self.NIG_NLL(y_true, gamma, v, alpha, beta)
        loss_reg = self.NIG_Reg(y_true, gamma, v, alpha, beta)
        return loss_nll + self.coeff * loss_reg


class EvidentialRegression(BaseUncertainty):
    def __init__(self, base_model_class, *args, **kwargs):
        super(EvidentialRegression, self).__init__()
        self.model = base_model_class(*args, **kwargs, out_features=4)


        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_losses_total = []
        self.train_time = 0

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_losses_total = []
        self.valid_time = 0

    def fit(self, epochs, train_loader, valid_loader, device, dtype, model_path="gnn/models/evidential.pt", use_wandb=False, warmup_steps=0, force_weight=1.0, energy_weight=1.0, log_interval=100, patience=200, factor=0.1, lr=1e-3, min_lr=1e-6): 

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-16)   
        criterion = EvidentialRegressionLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
        
        if use_wandb:
            self.init_wandb(scheduler,criterion,optimizer,model_path,train_loader,valid_loader,epochs,lr,patience,factor)

        best_valid_loss = np.inf

        for epoch in range(epochs):
            self.train_epoch(train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch, device=device, dtype=dtype, force_weight=force_weight, energy_weight=energy_weight, log_interval=log_interval)
            self.valid_epoch(valid_loader=valid_loader, criterion=criterion, device=device, dtype=dtype, force_weight=force_weight, energy_weight=energy_weight)
            self.epoch_summary(epoch, use_wandb=use_wandb, lr=optimizer.param_groups[0]['lr'])

            if np.array(self.valid_losses_total).mean() < best_valid_loss and model_path is not None:
                best_valid_loss = np.array(self.valid_losses_total).mean()
                torch.save(self.state_dict(), model_path)

            self.lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(np.array(self.valid_losses_total).mean())
            self.lr_after = optimizer.param_groups[0]['lr']
            self.drop_metrics()

        if use_wandb:
            wandb.finish()


    def train_epoch(self, train_loader, optimizer, criterion, epoch, device, dtype, force_weight=1.0, energy_weight=1.0, log_interval=100):
        start = time.time()
        self.train()
        for i,data in enumerate(train_loader):

            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            mean_energy, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            
            loss_energy = criterion(mean_energy, label_energy)
            loss_force = criterion(mean_force, label_forces)
            total_loss = force_weight*loss_force + energy_weight*loss_energy
            total_loss /= force_weight + energy_weight
            total_loss = 0.5 * torch.mean(torch.log(uncertainty) + total_loss/uncertainty)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
            optimizer.step()
            

            self.train_losses_energy.append(loss_energy.item())
            self.train_losses_force.append(loss_force.item())
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
            total_loss = force_weight*loss_force + energy_weight*loss_energy
            total_loss /= force_weight + energy_weight
            total_loss = 0.5 * torch.mean(torch.log(uncertainty) + total_loss/uncertainty)

            self.valid_losses_energy.append(loss_energy.item())
            self.valid_losses_force.append(loss_force.item())
            self.valid_losses_total.append(total_loss.item())

        self.valid_time = time.time() - start


    def predict(self, x, *args, **kwargs):
        self.eval()
        return self.forward(x=x, *args, **kwargs)


    def forward(self, x, *args, **kwargs):
        output = self.model.forward(x=x, *args, **kwargs)
        mu, v, alpha, beta = torch.split(output, 1, dim=-1)
        energy = mu
        variance = np.sqrt(beta / (v * (alpha - 1)))
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

    
    def epoch_summary(self, epoch, use_wandb=False, lr=None):
        print("", flush=True)
        print(f"Training and Validation Results of Epoch {epoch}:", flush=True)
        print("================================")
        print(f"Training Loss Energy: {np.array(self.train_losses_energy).mean()}, Training Loss Force: {np.array(self.train_losses_force).mean()}, time: {self.train_time}", flush=True)
        if len(self.valid_losses_energy) > 0:
            print(f"Validation Loss Energy: {np.array(self.valid_losses_energy).mean()}, Validation Loss Force: {np.array(self.valid_losses_force).mean()}, time: {self.valid_time}", flush=True)
        print("", flush=True)

        if use_wandb:
            wandb.log({
                "train_loss_energy": np.array(self.train_losses_energy).mean(),
                "train_loss_force": np.array(self.train_losses_force).mean(),
                "train_loss_total": np.array(self.train_losses_total).mean(),
                "valid_loss_energy": np.array(self.valid_losses_energy).mean(),
                "valid_loss_force": np.array(self.valid_losses_force).mean(),
                "valid_loss_total": np.array(self.valid_losses_total).mean(),
                "lr" : lr 
            })

    def init_wandb(self, scheduler, criterion, optimizer, model_path, train_loader, valid_loader, epochs, lr, patience, factor):
        wandb.init(
                # set the wandb project where this run will be logged
                project="GNN-Uncertainty-Evidential",

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
                })
