from gnn.egnn import EGNN
from datasets.qm9 import QM9
from datasets.md_dataset import MDDataset
import torch
from torch import nn
from datasets.helper import utils as qm9_utils
import numpy as np



def make_global_adjacency_matrix(n_nodes):
    device = "cpu"
    row = (
        torch.arange(0, n_nodes, dtype=torch.long)
        .reshape(1, -1, 1)
        .repeat(1, 1, n_nodes)
        .to(device=device)
    )
    col = (
        torch.arange(0, n_nodes, dtype=torch.long)
        .reshape(1, 1, -1)
        .repeat(1, n_nodes, 1)
        .to(device=device)
    )
    full_adj = torch.concat([row, col], dim=0)
    diag_bool = torch.eye(n_nodes, dtype=torch.bool).to(device=device)
    return full_adj, diag_bool

if __name__ == "__main__":

    device = torch.device("cpu")
    dtype = torch.float32

    epochs = 2000


    model = EGNN(12,0,128,n_layers=16)

    qm9 = QM9()
    qm9.create(1,0)
    trainset = MDDataset("datasets/files/alaninedipeptide")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    charge_scale = qm9.charge_scale
    charge_power = 2

    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-16)
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100, verbose=True)

    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    #print(total_params)

    for epoch in range(epochs):
        losses = []
        for data in trainloader:
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

            pred = model.forward(x=atom_positions, h0=nodes, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
            loss = loss_fn(pred,label)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            scheduler.step(loss.item())        
            optimizer.zero_grad()
        print(np.array(losses).mean())
    
