from gnn.egnn import EGNN
import torch
from torch import nn
from datasets.helper import utils as qm9_utils


class ModelEnsemble(nn.Module):
    def __init__(self, base_model_class, num_models, *args, **kwargs):
        super(ModelEnsemble, self).__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([base_model_class(*args, **kwargs) for _ in range(num_models)])
        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_total_losses = []
        self.train_uncertainties = []

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_total_losses = []
        self.valid_uncertainties = []
        self.num_in_interval = 0
        self.total_preds = 0

    def train_epoch(self, train_loader, optimizer, criterion, epoch, device, dtype, force_weight=1.0, energy_weight=1.0, log_interval=100, num_ensembles=3):
         for i,data in enumerate(train_loader):

            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            energies, mean_energy, forces, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes, leave_out=(i%num_ensembles))
            stacked_label_energy = label_energy.repeat(len(energies),1)
            stacked_label_force = label_forces.repeat(len(forces),1,1)
            
            stacked_loss_energy = criterion(energies, stacked_label_energy)
            loss_energy = criterion(mean_energy, label_energy)
            stacked_loss_force = criterion(forces, stacked_label_force)
            loss_force = criterion(mean_force, label_forces)
            total_loss = force_weight*stacked_loss_force + energy_weight*stacked_loss_energy

            optimizer.zero_grad()
            total_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            

            self.train_losses_energy.append(loss_energy.item())
            self.train_losses_force.append(loss_force.item())
            self.train_total_losses.append(total_loss.item())

            uncertainty = torch.mean(uncertainty)
            self.train_uncertainties.append(uncertainty.item())    
            
            if (i+1) % log_interval == 0:
                print(f"Epoch {epoch}, Batch {i+1}/{len(train_loader)}, Loss: {loss_energy.item()}, Uncertainty: {uncertainty.item()}", flush=True)

    def valid_epoch(self, valid_loader, criterion, device, dtype, force_weight=1.0, energy_weight=1.0):
        
        for i,data in enumerate(valid_loader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            energies, mean_energy, forces, mean_force, uncertainty = self.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            stacked_label_energy = label_energy.repeat(len(energies),1)
            stacked_label_force = label_forces.repeat(len(forces),1,1)
            
            stacked_loss_energy = criterion(energies, stacked_label_energy)
            loss_energy = criterion(mean_energy, label_energy)
            stacked_loss_force = criterion(forces, stacked_label_force)
            loss_force = criterion(mean_force, label_forces)
            total_loss = force_weight*stacked_loss_force + energy_weight*stacked_loss_energy

            self.valid_losses_energy.append(loss_energy.item())
            self.valid_losses_force.append(loss_force.item())
            self.valid_total_losses.append(total_loss.item())

            uncertainty = torch.mean(uncertainty)
            self.valid_uncertainties.append(uncertainty.item())

            self.num_in_interval += torch.sum(torch.abs(mean_energy - label_energy) <= uncertainty/2)
            self.total_preds += mean_energy.size(0)

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

        
    def forward(self, x, leave_out=None, *args, **kwargs):
        # Collect the outputs from all models
        if leave_out is not None:
            energies = torch.stack([model(x=x, *args,**kwargs) for i, model in enumerate(self.models) if i != leave_out])
        else:
            energies = torch.stack([model(x=x,*args,**kwargs) for model in self.models])
        mean_energy = torch.mean(energies, dim=0)
        grad_outputs = torch.ones_like(mean_energy)
        forces = torch.stack([-torch.autograd.grad(outputs=e, inputs=x, grad_outputs=grad_outputs, create_graph=True)[0] for e in energies])
        mean_force = torch.mean(forces, dim=0) 
        ensemble_uncertainty = (torch.max(energies, dim=0).values - torch.min(energies, dim=0).values) * 1
        return energies, mean_energy, forces, mean_force, ensemble_uncertainty
    
    

    def pop_metrics(self):
        train_losses_energy = self.train_losses_energy
        train_losses_force = self.train_losses_force
        train_total_losses = self.train_total_losses
        train_uncertainties = self.train_uncertainties

        valid_losses_energy = self.valid_losses_energy
        valid_losses_force = self.valid_losses_force
        valid_total_losses = self.valid_total_losses
        num_in_interval = self.num_in_interval
        total_preds = self.total_preds
        valid_uncertainties = self.valid_uncertainties

        self.train_losses_energy = []
        self.train_losses_force = []
        self.train_total_losses = []
        self.train_uncertainties = []

        self.valid_losses_energy = []
        self.valid_losses_force = []
        self.valid_total_losses = []
        self.valid_uncertainties = []
        self.num_in_interval = 0
        self.total_preds = 0

        return train_losses_energy, train_losses_force, train_total_losses, train_uncertainties, valid_losses_energy, valid_losses_force, valid_total_losses, valid_uncertainties, num_in_interval, total_preds