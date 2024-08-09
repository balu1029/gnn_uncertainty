from datasets.md17_dataset import MD17Dataset
from uncertainty.ensemble import ModelEnsemble
from gnn.egnn import EGNN
import torch
import numpy as np


class ActiveLearning:
    
    def __init__(self, model_path:str, num_ensembles:int, in_nf:int, hidden_nf:int, n_layers:int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model = ModelEnsemble(EGNN, num_ensembles, in_node_nf=12, in_edge_nf=0, hidden_nf=16, n_layers=2).to(self.device)
        self.model.load_state_dict(torch.load(model_path))

    def run_simulation(self, steps:int)->np.array:

        for i in range(steps):
            pass

