import torch
from torch import nn

from abc import abstractmethod

import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

from datasets.helper import utils as qm9_utils
import csv
import os

class BaseUncertainty(nn.Module):

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, epochs, train_loader, valid_loader, device, dtype, model_path, use_wandb=False, force_weight=1.0, energy_weight=1.0, log_interval=100, patience=200, factor=0.1, lr=1e-3, min_lr=1e-6):
        pass

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

        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
        label_energy = (data["energies"]).to(device, dtype)
        label_forces = (data["forces"]).view(batch_size * n_nodes, -1).to(device, dtype)

        return atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes
    
    
    def evaluate_uncertainty(self, test_loader, device, dtype, plot_path=None, csv_path=None, show_plot=False):
        criterion = nn.L1Loss(reduction='none')
        energy_losses = torch.Tensor()
        uncertainties = torch.Tensor()
        self.eval()
        for i,data in enumerate(test_loader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            energy, forces, uncertainty = self.predict(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            energy_losses = torch.cat((energy_losses, criterion(energy.cpu().detach(), label_energy.cpu())), dim=0)
            uncertainties = torch.cat((uncertainties, uncertainty.cpu().detach()), dim=0)

            atom_positions.detach()
        
        energy_losses = energy_losses.cpu().detach().numpy()
        uncertainties = uncertainties.cpu().detach().numpy()


        correlation = np.corrcoef(energy_losses, uncertainties)[0, 1]
        self._scatter_plot(energy_losses, uncertainties, self.__class__.__name__, 'Energy Losses', 'Uncertainties', text=f"Correlation: {correlation}", save_path=plot_path, show_plot=show_plot)
        print(f"Correlation: {correlation}")
        if csv_path:
            if os.path.exists(csv_path):
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.__class__.__name__, correlation])
            else:
                with open(csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Method', 'Correlation Coefficient'])
                    writer.writerow([self.__class__.__name__,  correlation])

    def evaluate_model(self, test_loader, device, dtype, plot_path=None, csv_path=None, show_plot=False):
        criterion = nn.L1Loss(reduction='none')
        predictions_energy = torch.Tensor()
        ground_truths_energy = torch.Tensor()
        self.eval()
        for i,data in enumerate(test_loader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            energy, forces, uncertainty = self.predict(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            predictions_energy = torch.cat((predictions_energy, energy.cpu().detach()), dim=0)
            ground_truths_energy = torch.cat((ground_truths_energy, label_energy.cpu().detach()), dim=0)

            atom_positions.detach()
        
        ground_truths_energy = ground_truths_energy.cpu().detach().numpy()
        predictions_energy = predictions_energy.cpu().detach().numpy()

        energy_r2 = r2_score(ground_truths_energy, predictions_energy)

        self._scatter_plot(ground_truths_energy, predictions_energy, self.__class__.__name__, 'Ground Truth Energy', 'Predicted Energy', text=f"Energy R2 Score: {energy_r2}", save_path=plot_path, show_plot=show_plot)
        if csv_path:
            if os.path.exists(csv_path):
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.__class__.__name__, energy_r2])
            else:
                with open(csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Method', 'Energy R2 Score'])
                    writer.writerow([self.__class__.__name__, energy_r2])

    def evaluate_all(self, test_loader_in, device, dtype, test_loader_out=None, plot_name=None, csv_path=None, show_plot=None):

        energy_r2_in, forces_r2_in, correlation_in, energy_losses_in, forces_losses_in = self._eval_all(test_loader_in, device, dtype, plot_path=f"{plot_name}_in", plot_title=self.__class__.__name__+' In Distribution')

        if test_loader_out:
            energy_r2_out, forces_r2_out, correlation_out, energy_losses_out, forces_losses_out = self._eval_all(test_loader_out, device, dtype, plot_path=f"{plot_name}_out", plot_title=self.__class__.__name__+' Out Distribution')
        
        else:
            energy_r2_out, forces_r2_out, correlation_out, energy_losses_out, forces_losses_out = 0, 0, 0, np.array([0]), np.array([0])

        if csv_path:
            if os.path.exists(csv_path):
                with open(csv_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([self.__class__.__name__, energy_r2_in, forces_r2_in, correlation_in, np.mean(energy_losses_in), np.mean(forces_losses_in), energy_r2_out, forces_r2_out, correlation_out, np.mean(energy_losses_out), np.mean(forces_losses_out)])
            else:
                with open(csv_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Method', 'Energy R2 Score In Distribution', 'Forces R2 Score In Distribution', 'Correlation In Distribution', 'Energy Losses In Distribution', 'Forces Losses In Distribution', 'Energy R2 Score Out Distribution', 'Forces R2 Score Out Distribution', 'Correlation Out Distribution', 'Energy Losses Out Distribution', 'Forces Losses Out Distribution'])
                    writer.writerow([self.__class__.__name__, energy_r2_in, forces_r2_in, correlation_in, np.mean(energy_losses_in), np.mean(forces_losses_in), energy_r2_out, forces_r2_out, correlation_out, np.mean(energy_losses_out), np.mean(forces_losses_out)])

    def _eval_all(self, dataloader, device, dtype, plot_path=None, plot_title=None):
        criterion = nn.L1Loss(reduction='none')
        predictions_energy = torch.Tensor()
        ground_truths_energy = torch.Tensor()
        predictions_forces = torch.Tensor()
        ground_truths_forces = torch.Tensor()
        energy_losses = torch.Tensor()
        forces_losses = torch.Tensor()
        uncertainties = torch.Tensor()
        self.eval()
        for i,data in enumerate(dataloader):
            atom_positions, nodes, edges, atom_mask, edge_mask, label_energy, label_forces, n_nodes = self.prepare_data(data, device, dtype)    

            energy, forces, uncertainty = self.predict(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            predictions_energy = torch.cat((predictions_energy, energy.cpu().detach()), dim=0)
            ground_truths_energy = torch.cat((ground_truths_energy, label_energy.cpu().detach()), dim=0)
            predictions_forces = torch.cat((predictions_forces, forces.cpu().detach()), dim=0)
            ground_truths_forces = torch.cat((ground_truths_forces, label_forces.cpu().detach()), dim=0)
            energy_losses = torch.cat((energy_losses, criterion(energy.cpu().detach(), label_energy.cpu())), dim=0)
            forces_losses = torch.cat((forces_losses, criterion(forces.cpu().detach(), label_forces.cpu())), dim=0)
            uncertainties = torch.cat((uncertainties, uncertainty.cpu().detach()), dim=0)

            atom_positions.detach()
        ground_truths_energy = ground_truths_energy.cpu().detach().numpy()
        predictions_energy = predictions_energy.cpu().detach().numpy()
        ground_truths_forces = ground_truths_forces.cpu().detach().numpy()
        predictions_forces = predictions_forces.cpu().detach().numpy()
        energy_losses = energy_losses.cpu().detach().numpy()
        forces_losses = forces_losses.cpu().detach().numpy()
        uncertainties = uncertainties.cpu().detach().numpy()
        energy_r2 = r2_score(ground_truths_energy, predictions_energy)
        forces_r2 = r2_score(ground_truths_forces, predictions_forces)
        correlation = np.corrcoef(energy_losses, uncertainties)[0, 1]

        if plot_path:
            self._scatter_plot(ground_truths_energy, predictions_energy, plot_title, 'Ground Truth Energy', 'Predicted Energy', text=f"Energy R2 Score: {energy_r2}", save_path=plot_path + "_energy.png", show_plot=False)
            self._scatter_plot(energy_losses, uncertainties, plot_title, 'Energy Losses', 'Uncertainties', text=f"Correlation: {correlation}", save_path=plot_path + "_uncertainty.png", show_plot=False)

        return energy_r2, forces_r2, correlation, energy_losses, forces_losses


    def _scatter_plot(self, x, y, title, xlabel, ylabel, text="", save_path=None, show_plot=False):
        plt.scatter(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.plot([min(x), max(x)], [min(x), max(x)], color='red', linestyle='--')
        plt.text(0.1, 0.9, text, transform=plt.gca().transAxes)
        if save_path is not None:
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        plt.close()