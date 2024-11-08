import torch
import torch.nn as nn
import pyro.distributions as dist
import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class DeltaPosteriorX(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        # means
        self.x_0 = nn.Parameter(make_torch_tensor(options["mean_x_0"], torch.Size([options["dim_x"]])))

    def sample(self, num_samples=None):
        if num_samples==None:
            x = dist.Delta(self.x_0, event_dim=1).rsample()
        else:
            x = dist.Delta(self.x_0, event_dim=1).rsample(torch.Size((num_samples,)))
        return x

    def log_prob(self, x):
        # calculate log_prob
        log_prob_x = dist.Delta(self.x_0, event_dim=1).log_prob(x)
        return log_prob_x
    
    def entropy(self):
        # calculate entropy
        entropy_x = torch.tensor(0.0)
        return entropy_x

    @property
    def mean_x(self):
        return self.x_0
