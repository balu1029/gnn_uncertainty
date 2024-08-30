from uncertainty.swag import SWAG
from gnn.egnn import EGNN


in_node_nf = 12
in_edge_nf = 0
hidden_nf = 32
n_layers = 4


swag = SWAG(EGNN, in_node_nf=in_node_nf, in_edge_nf=in_edge_nf, hidden_nf=hidden_nf, n_layers=n_layers)

swag._sample_swag_moments()