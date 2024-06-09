from gnn.egnn import EGNN
from datasets.qm9 import QM9
import torch
from torch import nn
from datasets.helper import utils as qm9_utils

from torchsummary import summary

if __name__ == "__main__":

    device = torch.device("cpu")
    dtype = torch.float32

    epochs = 1

    model = EGNN(15,0,64,n_layers=3)

    qm9 = QM9()
    qm9.create(128,0)
    trainloader = qm9.dataloaders["train"]
    charge_scale = qm9.charge_scale
    charge_power = 2
    self_energies = -6.5369

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-16)
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    #print(total_params)
    for epoch in range(epochs):
        for data in trainloader:
            batch_size, n_nodes, _ = data['positions'].size()
            atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, dtype)
            atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, dtype)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype)
            nodes = qm9_utils.preprocess_input(one_hot, charges, charge_power, charge_scale, device)


            nodes = nodes.view(batch_size * n_nodes, -1)
            # nodes = torch.cat([one_hot, charges], dim=1)
            edges = qm9_utils.get_adj_matrix(n_nodes, batch_size, device)
            label = (data["homo"] - self_energies).to(device, dtype)

            pred = model.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            loss = loss_fn(pred,label)
            print(loss.item())

            loss.backward()
            optimizer.step()
