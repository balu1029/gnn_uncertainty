from gnn.egnn import EGNN
import torch
from torch import nn


class ModelEnsemble(nn.Module):
    def __init__(self, base_model_class, num_models, *args, **kwargs):
        super(ModelEnsemble, self).__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([base_model_class(*args, **kwargs) for _ in range(num_models)])
        
    def forward(self,leave_out=None, *args, **kwargs):
        # Collect the outputs from all models
        if leave_out is not None:
            stacked_outputs = torch.stack([model(*args,**kwargs) for i, model in enumerate(self.models) if i != leave_out])
        else:
            stacked_outputs = torch.stack([model(*args,**kwargs) for model in self.models])
        ensemble_output = torch.mean(stacked_outputs, dim=0)
        ensemble_uncertainty = torch.std(stacked_outputs, dim=0) * 3
        return stacked_outputs, ensemble_output, ensemble_uncertainty