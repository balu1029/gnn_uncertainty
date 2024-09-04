import torch
from torch import nn
from gnn.architecture.egnn_architecture import *
from gnn.base_gnn import BaseGNN


class EGNN(nn.Module, BaseGNN):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1, out_features=1, multi_dec=False):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.in_node_nf = in_node_nf   
        self.in_edge_nf = in_edge_nf
        self.multi_dec = multi_dec

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))
        if self.multi_dec:
            graph_dec_energy = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                        act_fn,
                                        nn.Linear(self.hidden_nf, out_features)).to(self.device)
            graph_dec_variance = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                        act_fn,
                                        nn.Linear(self.hidden_nf, out_features)).to(self.device)
            self.graph_dec = nn.ModuleList([graph_dec_energy, graph_dec_variance])
        else:
            self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                        act_fn,
                                        nn.Linear(self.hidden_nf, out_features))
            
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        if self.multi_dec:
            pred = torch.stack([self.graph_dec[0](h), self.graph_dec[1](h)], dim=1)
        else:   
            pred = self.graph_dec(h)
        return pred.squeeze(1)



