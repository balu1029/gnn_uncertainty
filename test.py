from uncertainty.mve import MVE
from gnn.egnn import EGNN

num_ensembles = 3
in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4


mve = MVE(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers)