from gnn.egnn import EGNN
import torch
from torch import nn
from datasets.helper import utils as qm9_utils

import wandb
import numpy as np  
import time
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



class MVE(nn.Module):
    def __init__(self, base_model_class, multi_dec = True, *args, **kwargs):
        super(MVE, self).__init__()
        self.model = base_model_class(*args, **kwargs, multi_dec=multi_dec)


        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_losses_total = []
        self.train_uncertainties = []
        self.train_time = 0

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_losses_total = []
        self.valid_uncertainties = []
        self.num_in_interval = 0
        self.total_preds = 0
        self.valid_time = 0

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
                

                self.train_losses_energy.append(loss_energy.item())
                self.train_losses_force.append(loss_force.item())
                self.train_losses_total.append(total_loss.item())

                self.train_uncertainties.append(torch.mean(uncertainty).item())
                
                if (i+1) % log_interval == 0:
                    print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss_energy.item()}, Uncertainty: {uncertainty.item()}", flush=True) 
            self.train_time = time.time() - start
            lr = optimizer.param_groups[0]['lr']
            self.epoch_summary(f"Warmup-{epoch}", False, lr)
        print("Warmup phase finished", flush=True)

    def train_epoch(self, train_loader, optimizer, criterion, epoch, device, dtype, force_weight=1.0, energy_weight=1.0, log_interval=100, num_ensembles=3):
        start = time.time()
        self.train()
        for i,data in enumerate(train_loader):

            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            mean_energy, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            
            loss_energy = criterion(mean_energy, label_energy)
            loss_force = criterion(mean_force, label_forces)
            total_loss = force_weight*loss_force + energy_weight*loss_energy
            total_loss = 0.5 * torch.mean(torch.log(uncertainty) + total_loss/uncertainty)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
            optimizer.step()
            

            self.train_losses_energy.append(loss_energy.item())
            self.train_losses_force.append(loss_force.item())
            self.train_losses_total.append(total_loss.item())

            self.train_uncertainties.append(torch.mean(uncertainty).item())
            
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
            total_loss = 0.5 * torch.mean(torch.log(uncertainty) + total_loss/uncertainty)

            self.valid_losses_energy.append(loss_energy.item())
            self.valid_losses_force.append(loss_force.item())
            self.valid_losses_total.append(total_loss.item())

            self.valid_uncertainties.append(torch.mean(uncertainty).item())

            self.num_in_interval += torch.sum(torch.abs(mean_energy - label_energy) <= uncertainty/2)
            self.total_preds += mean_energy.size(0)

        self.valid_time = time.time() - start

    def prepare_data(self, data, device, dtype):
        batch_size, n_nodes, _ = data['coordinates'].size()
        atom_positions = data['coordinates'].view(batch_size * n_nodes, -1).requires_grad_(True).to(device, dtype)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        charge_scale = data['charge_scale'][0]
        charge_power = data['charge_power'][0]

        nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, device)

        nodes = nodes.view(batch_size * n_nodes, -1)

        # nodes = torch.cat([one_hot, charges], dim=1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        label_energy = (data["energies"]).to(device, dtype)
        label_forces = (data["forces"]).view(batch_size * n_nodes, -1).to(device, dtype)

        return atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes

        
    def forward(self, x, *args, **kwargs):
        output = self.model.forward(x=x, *args, **kwargs)
        energy = output[:,0].squeeze()
        variance = output[:,1].squeeze()
        grad_output = torch.ones_like(energy)
        force = -torch.autograd.grad(outputs=energy, inputs=x, grad_outputs=grad_output, create_graph=True)[0]
        return energy, force, variance
    
    

    def pop_metrics(self):
        train_losses_energy = self.train_losses_energy
        train_losses_force = self.train_losses_force
        train_total_losses = self.train_losses_total
        train_uncertainties = self.train_uncertainties
        train_time = self.train_time

        valid_losses_energy = self.valid_losses_energy
        valid_losses_force = self.valid_losses_force
        valid_total_losses = self.valid_losses_total
        num_in_interval = self.num_in_interval
        total_preds = self.total_preds
        valid_uncertainties = self.valid_uncertainties
        valid_time = self.valid_time

        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_losses_total = []
        self.train_uncertainties = []
        self.train_time = 0

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_losses_total = []
        self.valid_uncertainties = []
        self.num_in_interval = 0
        self.total_preds = 0
        self.valid_time = 0

        return train_losses_energy, train_losses_force, train_total_losses, train_uncertainties, valid_losses_energy, valid_losses_force, valid_total_losses, valid_uncertainties, num_in_interval, total_preds, train_time, valid_time
    
    def epoch_summary(self, epoch, use_wandb=False, lr=None):
        print("", flush=True)
        print(f"Training and Validation Results of Epoch {epoch}:", flush=True)
        print("================================")
        print(f"Training Loss Energy: {np.array(self.train_losses_energy).mean()}, Training Loss Force: {np.array(self.train_losses_force).mean()}, Training Uncertainty: {np.array(self.train_uncertainties).mean()}, time: {self.train_time}", flush=True)
        if len(self.valid_losses_energy) > 0:
            print(f"Validation Loss Energy: {np.array(self.valid_losses_energy).mean()}, Validation Loss Force: {np.array(self.valid_losses_force).mean()},Validation Uncertainty: {np.array(self.valid_uncertainties).mean()}, time: {self.valid_time}", flush=True)
            print(f"Number of predictions within uncertainty interval: {self.num_in_interval}/{self.total_preds} ({self.num_in_interval/self.total_preds*100:.2f}%)", flush=True)
        print("", flush=True)

        if use_wandb:
            wandb.log({
                "train_loss_energy": np.array(self.train_losses_energy).mean(),
                "train_uncertainty": np.array(self.train_uncertainties).mean(),
                "train_loss_force": np.array(self.train_losses_force).mean(),
                "train_loss_total": np.array(self.train_losses_total).mean(),
                "valid_loss_energy": np.array(self.valid_losses_energy).mean(),
                "valid_uncertainty": np.array(self.valid_uncertainties).mean(),
                "valid_loss_force": np.array(self.valid_losses_force).mean(),
                "valid_loss_total": np.array(self.valid_losses_total).mean(),
                "in_interval": self.num_in_interval/self.total_preds*100,
                "lr" : lr 
            })

    def evaluate_uncertainty(self, test_loader, device, dtype, save_path=None):
        criterion = nn.L1Loss(reduction='none')
        energy_losses = torch.Tensor()
        uncertainties = torch.Tensor()
        self.eval()
        for i,data in enumerate(test_loader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            mean_energy, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            energy_losses = torch.cat((energy_losses, criterion(mean_energy.detach(), label_energy)), dim=0)
            uncertainties = torch.cat((uncertainties, uncertainty.detach()), dim=0)

            self.total_preds += mean_energy.size(0)
            atom_positions.detach()
        
        energy_losses = energy_losses.cpu().detach().numpy()
        uncertainties = uncertainties.cpu().detach().numpy()


        correlation = np.corrcoef(energy_losses, uncertainties)[0, 1]
        self._scatter_plot(energy_losses, uncertainties, 'MVE', 'Energy Losses', 'Uncertainties', text=f"Correlation: {correlation}", save_path=save_path)
        print(f"Correlation: {correlation}")

    def evaluate_model(self, test_loader, device, dtype, save_path=None):
        criterion = nn.L1Loss(reduction='none')
        predictions_energy = torch.Tensor()
        ground_truths_energy = torch.Tensor()
        self.eval()
        for i,data in enumerate(test_loader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            mean_energy, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            predictions_energy = torch.cat((predictions_energy, mean_energy.detach()), dim=0)
            ground_truths_energy = torch.cat((ground_truths_energy, label_energy.detach()), dim=0)

            self.total_preds += mean_energy.size(0)
            atom_positions.detach()
        # Calculate R2 scores for energy and forces
        energy_r2 = r2_score(ground_truths_energy.cpu().detach().numpy(), predictions_energy.cpu().detach().numpy())

        self._scatter_plot(ground_truths_energy.cpu().detach().numpy(), predictions_energy.cpu().detach().numpy(), 'MVE', 'Ground Truth Energy', 'Predicted Energy', text=f"Energy R2 Score: {energy_r2}", save_path=save_path)


    def _scatter_plot(self, x, y, title, xlabel, ylabel, text="", save_path=None):
        plt.scatter(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot([min(x), max(x)], [min(x), max(x)], color='red', linestyle='--')
        plt.text(0.1, 0.9, text, transform=plt.gca().transAxes)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()


if __name__ == "__main__":
    mve = MVE()