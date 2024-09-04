from gnn.egnn import EGNN
import torch
from torch import nn
from datasets.helper import utils as qm9_utils

import wandb
import numpy as np  
import time
import matplotlib.pyplot as plt


class SWAG(nn.Module):
    def __init__(self, base_model_class, sample_size = 5, low_rank = False,  *args, **kwargs):
        super(SWAG, self).__init__()
        self.model = base_model_class(*args, **kwargs)

        self.num_samples = 0
        self.distribution = None

        self.ignore_weight_list = ["first_moment", "second_moment", "cov_matrix"]
        new_weights = torch.cat([param.view(-1) for param in self.parameters()])

        self.register_buffer("first_moment", torch.zeros_like(new_weights))
        self.register_buffer("second_moment", torch.zeros_like(new_weights))
        #self.register_buffer("cov_matrix", torch.zeros(size=(new_weights.shape[0], new_weights.shape[0])))

        

        self.low_rank = low_rank

        self.sample_size = sample_size

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
    
    def fit(self, epochs, swag_start_epoch, swag_freq, train_loader, valid_loader, device, dtype, model_path="gnn/models/swag.pt" , force_weight=1.0, energy_weight=1.0, log_interval=100):

        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-16)   
        criterion = nn.L1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)

        best_valid_loss = np.inf

        for epoch in range(epochs):
            self.train_epoch(train_loader=train_loader, optimizer=optimizer, criterion=criterion, epoch=epoch, device=device, dtype=dtype, force_weight=force_weight, energy_weight=energy_weight, log_interval=log_interval)
            self.valid_epoch(valid_loader=valid_loader, criterion=criterion, device=device, dtype=dtype, force_weight=force_weight, energy_weight=energy_weight)
            self.epoch_summary(epoch)

            if np.array(self.valid_losses_total).mean() < best_valid_loss:
                best_valid_loss = np.array(self.valid_losses_total).mean()
                torch.save(self.state_dict(), model_path)

            self.pop_metrics()
            

            if epoch >= swag_start_epoch and (epoch-swag_start_epoch) % swag_freq == 0:
                self._sample_swag_moments()
            else:                                                   # do not change learning rate after starting to sample weights
                self.lr_before = optimizer.param_groups[0]['lr']
                scheduler.step(np.array(self.valid_losses_total).mean())
                self.lr_after = optimizer.param_groups[0]['lr']
                self.pop_metrics()
        
    def predict(self, x, *args, **kwargs):
        self.eval()
        coord_shape = x.shape
        forces = []
        energies = []
        
        for i in range(self.sample_size):
            self._load_sample_swag_weights()
            energy, force = self.forward(x, *args, **kwargs)     
            energies.append(energy.detach().unsqueeze(0))
            forces.append(force.detach().unsqueeze(0))
        energies = torch.cat(energies, dim=0)
        forces = torch.cat(forces, dim=0)
        uncertainty = torch.std(energies,dim=0)
        energy = torch.mean(energies, dim=0)
        force = torch.mean(forces, dim=0)
        
        return energy, force, uncertainty
            
                

    def train_epoch(self, train_loader, optimizer, criterion, epoch, device, dtype, force_weight=1.0, energy_weight=1.0, log_interval=100, num_ensembles=3):
        start = time.time()
        self.train()
        for i,data in enumerate(train_loader):

            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            energy, force = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            
            loss_energy = criterion(energy, label_energy)
            loss_force = criterion(force, label_forces)
            total_loss = force_weight*loss_force + energy_weight*loss_energy

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
            optimizer.step()
            

            self.train_losses_energy.append(loss_energy.item())
            self.train_losses_force.append(loss_force.item())
            self.train_losses_total.append(total_loss.item())

            
            if (i+1) % log_interval == 0:
                print(f"Epoch {epoch}, Batch {i+1}/{len(train_loader)}, Loss: {loss_energy.item()}", flush=True)
        
        self.train_time = time.time() - start


    def valid_epoch(self, valid_loader, criterion, device, dtype, force_weight=1.0, energy_weight=1.0):
        start = time.time()
        self.eval()
        for i,data in enumerate(valid_loader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            energy, force = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)

 
            loss_energy = criterion(energy, label_energy)
            loss_force = criterion(force, label_forces)
            total_loss = force_weight*loss_force + energy_weight*loss_energy

            self.valid_losses_energy.append(loss_energy.item())
            self.valid_losses_force.append(loss_force.item())
            self.valid_losses_total.append(total_loss.item())

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
        energy = output.squeeze()
        grad_output = torch.ones_like(energy)
        force = -torch.autograd.grad(outputs=energy, inputs=x, grad_outputs=grad_output, create_graph=True)[0]
        return energy, force
    
    
    def _sample_swag_moments(self):
        params = self.state_dict()
        
        
        new_weights = torch.cat([param.view(-1) for param in self.parameters()])

        
        if self.num_samples == 0:
            self.first_moment = torch.zeros_like(new_weights)
            self.second_moment = torch.zeros_like(new_weights)

        self.first_moment = (self.first_moment * self.num_samples + new_weights) / (self.num_samples + 1)
        self.second_moment = (self.second_moment * self.num_samples + new_weights**2) / (self.num_samples + 1)
        self.num_samples += 1


    def _load_sample_swag_weights(self):
        if not self.low_rank:
            var = torch.clamp(self.second_moment - self.first_moment**2, 1e-30)
            var_sample = var.sqrt() * torch.randn_like(var)
            weights = self.first_moment + var_sample
        else:
            return NotImplementedError
            L = torch.linalg.cholesky(self.cov_matrix)
            z = torch.randn_like(self.first_moment)
            weights = self.first_moment + L @ z
        self.load_flattened_weights_into_model(weights)
        
    def load_flattened_weights_into_model(self, weights):
        current_index = 0
        #flattened_weights = torch.cat([param.view(-1) for param in weights])
        flattened_weights = weights

        for param in self.parameters():
            num_param_elements = param.numel()
            param_flat = flattened_weights[current_index:current_index + num_param_elements]
            param.data = param_flat.view_as(param).data
            current_index += num_param_elements

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
        print(f"Training Loss Energy: {np.array(self.train_losses_energy).mean()}, Training Loss Force: {np.array(self.train_losses_force).mean()}, time: {self.train_time}", flush=True)
        if len(self.valid_losses_energy) > 0:
            print(f"Validation Loss Energy: {np.array(self.valid_losses_energy).mean()}, Validation Loss Force: {np.array(self.valid_losses_force).mean()}, time: {self.valid_time}", flush=True)

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

    def evaluate_uncertainty(self, test_loader, device, dtype):
        criterion = nn.L1Loss(reduction='none')
        energy_losses = torch.Tensor()
        uncertainties = torch.Tensor()
        self.eval()
        for i,data in enumerate(test_loader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            mean_energy, mean_force, uncertainty = self.predict(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            energy_losses = torch.cat((energy_losses, criterion(mean_energy.detach(), label_energy)), dim=0)
            uncertainties = torch.cat((uncertainties, uncertainty.detach()), dim=0)

            self.total_preds += mean_energy.size(0)
            atom_positions.detach()
        
        energy_losses = energy_losses.cpu().detach().numpy()
        uncertainties = uncertainties.cpu().detach().numpy()

        plt.scatter(energy_losses, uncertainties)
        plt.plot([0, max(energy_losses)], [0, max(energy_losses)], color='red', linestyle='--')
        plt.xlabel('Energy Losses')
        plt.ylabel('Uncertainties')
        plt.title('Scatter Plot of Energy Losses vs Uncertainties')
        plt.show()

        correlation = np.corrcoef(energy_losses, uncertainties)[0, 1]
        print(f"Correlation: {correlation}")


if __name__ == "__main__":
    swag = SWAG(EGNN, in_node_nf=12, in_edge_nf=0, hidden_nf=32, n_layers=4)