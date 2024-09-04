from uncertainty.swag import SWAG
from gnn.egnn import EGNN
from datasets.md17_dataset import MD17Dataset

import torch
from torch import nn
import time


in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4

batch_size = 32
lr = 1e-3

dataset = "datasets/files/ala_converged_forces_1000"
model_path = None#"./gnn/models/ala_converged_1000000_forces_mve.pt"
trainset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=True, train_ratio=0.2)
validset = MD17Dataset(dataset,subtract_self_energies=False, in_unit="kj/mol",train=False, train_ratio=0.8)


# Create data loaders for train and validation sets

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)

swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers)

swag.fit(epochs=2, swag_start_epoch=0, swag_freq=1,train_loader=trainloader, valid_loader=validloader, device="cpu", dtype=torch.float32)

#swag.load_state_dict(torch.load("swag.pt"))
#swag.evaluate_uncertainty(validloader, device="cpu", dtype=torch.float32)

