import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from datasets.helper import utils as qm9_utils
from datasets.md17_dataset import MD17Dataset

class ModelTrainer:
    def __init__(self, model, num_ensembles=3):
        self.model = model
        self.atom_numbers  = [0, 1, 0, 0, 1, 3, 2, 0, 1, 0, 1, 0, 0, 0, 1, 3, 2, 0, 1, 0, 0, 0]
        self.ground_charges  = {-1: 0.0,
                                0: 1.0,
                                1: 6.0,
                                2: 7.0,
                                3: 8.0}
        self.number_to_type = {0: "H",
                               1: "C",
                               2: "N",
                               3: "O"}
        self.charge_power = 2
        self.charge_scale = torch.tensor(max(self.ground_charges.values())+1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.num_ensembles = num_ensembles


    def train(self, num_epochs=10, learning_rate=1e-5, folder="al/run1/data"):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        loss_fn = nn.L1Loss()

        self.prepare_data(folder)
        self.model.train()

        for epoch in range(num_epochs):
            losses = []
            uncertainties = []
            log_interval = 200

            for i,data in enumerate(self.dataloader):
                batch_size, n_nodes, _ = data['coordinates'].size()
                atom_positions = data['coordinates'].view(batch_size * n_nodes, -1).to(self.device, self.dtype)
                atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(self.device, self.dtype)
                edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, -1).to(self.device, self.dtype)
                one_hot = data['one_hot'].to(self.device, self.dtype)
                charges = data['charges'].to(self.device, self.dtype)

                nodes = qm9_utils.preprocess_input(one_hot, charges, self.charge_power, self.charge_scale, self.device)



                nodes = nodes.view(batch_size * n_nodes, -1)

                # nodes = torch.cat([one_hot, charges], dim=1)
                edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)
                label = (data["energies"]).to(self.device, self.dtype)


                stacked_pred, pred, uncertainty = self.model.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes, leave_out=(i%self.num_ensembles))
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
                    print(f"Epoch {epoch}, Batch {i+1}/{len(self.dataloader)}, Loss: {loss.item()}, Uncertainty: {uncertainty.item()}", flush=True)
            
            print(f"Epoch {epoch}, Loss: {sum(losses) / len(losses)}, Uncertainty: {sum(uncertainties) / len(uncertainties)}")

        self.model.eval()
        return self.model
    
    def add_data(self, samples, labels, out_file):
         num_molecules = len(samples)
         samples = samples * 10
         with open(out_file, 'w') as file:
            for i in range(num_molecules):
                file.write(f"{len(samples[i])}\n")
                file.write(f"{labels[i]}\n")
                for j in range(len(samples[i])):
                    file.write(f"{self.number_to_type[self.atom_numbers[j]]} {samples[i][j][0]} {samples[i][j][1]} {samples[i][j][2]}\n")
    
    def prepare_data(self, folder):
        dataset = MD17Dataset(folder,seed=42, subtract_self_energies=False, in_unit="eV")
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)


    def prepare_data_old(self, samples, labels):
        positions = samples 
        atom_numbers = [self.atom_numbers] * samples.shape[0]
        atom_mask = torch.where(torch.tensor(atom_numbers) != -1, torch.tensor(1.0), torch.tensor(0.0))
        atom_numbers = np.array(atom_numbers)
        coordinates = np.array(positions)

        num_atoms = atom_numbers.shape[1]
        num_snapshots = atom_numbers.shape[0]
        num_types = len(set(self.atom_numbers))

        one_hot = torch.eye(num_types, dtype=torch.float)[torch.tensor(atom_numbers.flatten())].reshape(-1,num_atoms, num_types)
        charges = torch.tensor([self.ground_charges[atom] for atom in atom_numbers.flatten()]).reshape(-1, num_atoms)

        edge_mask = torch.eye(num_atoms, dtype=torch.float).unsqueeze(0).expand(num_snapshots, -1, -1)
        edge_mask = torch.where(edge_mask == 1, torch.tensor(0.0), torch.tensor(1.0))
        edge_mask = edge_mask * atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        edge_mask = edge_mask.view(num_snapshots, -1)

        coordinates = torch.tensor(coordinates, requires_grad=True)
        #coordinates = coordinates.unsqueeze(0)

        batch_size, n_nodes, _ = coordinates.size()
        atom_positions = coordinates.view(batch_size * n_nodes, -1).requires_grad_(True).to(self.device, self.dtype)
        atom_positions.retain_grad()
        atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(self.device, self.dtype)
        edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, -1).to(self.device, self.dtype)
        one_hot = one_hot.to(self.device, self.dtype)
        charges = charges.to(self.device, self.dtype)
        nodes = qm9_utils.preprocess_input(one_hot, charges, self.charge_power, self.charge_scale, self.device)
        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, self.device)

        self.data = {"nodes" : nodes,
                "atom_positions" : atom_positions,
                "energies" : torch.tensor(labels),
                "atom_mask": atom_mask,
                "edge_mask": edge_mask,
                "edges": edges,
                "n_nodes": n_nodes}