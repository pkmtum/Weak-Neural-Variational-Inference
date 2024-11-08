import torch
import torch.nn as nn
import pyro.distributions as dist
import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class DeltaPosteriorY(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        # means
        self.y_0 = nn.Parameter(make_torch_tensor(options["mean_y_0"], torch.Size([options["dim_y"]])))

    def sample(self, num_samples=None):
        if num_samples==None:
            y = dist.Delta(self.y_0, event_dim=1).rsample()
        else:
            y = dist.Delta(self.y_0, event_dim=1).rsample(torch.Size((num_samples,)))
        return y

    def log_prob(self, y):
        # calculate log_prob
        log_prob_y = dist.Delta(self.y_0, event_dim=1).log_prob(y)
        return log_prob_y
    
    def entropy(self):
        # calculate entropy
        entropy_y = torch.tensor(0.0)
        return entropy_y

    @property
    def mean_y(self):
        return self.y_0
