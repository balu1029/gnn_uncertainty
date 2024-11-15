import torch
from torch import nn
from gnn.architecture.egnn_architecture import *
from gnn.base_gnn import BaseGNN

import torch.nn.functional as F

class DenseNormalGamma(nn.Module):
    def __init__(self, in_features, out_features):
        super(DenseNormalGamma, self).__init__()
        self.units = int(out_features)
        self.dense = nn.Linear(in_features=in_features, out_features=4 * self.units)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = torch.chunk(output, 4, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return torch.cat([mu, v, alpha, beta], dim=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4 * self.units)

    def get_config(self):
        return {'units': self.units}


class EGNN(nn.Module, BaseGNN):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1, out_features=1, multi_dec=False, evidential=False):
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
                                        nn.Linear(self.hidden_nf, out_features),
                                        nn.Softplus()).to(self.device)
            self.graph_dec = nn.ModuleList([graph_dec_energy, graph_dec_variance])

        elif evidential:
            self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                        act_fn,
                                        DenseNormalGamma(in_features=self.hidden_nf, out_features=1))
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



