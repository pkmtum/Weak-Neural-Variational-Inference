import torch
import torch.nn as nn
import pyro.distributions as dist
import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class DeltaPosteriorY_given_X(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        # means
        self.y_0 = nn.Parameter(make_torch_tensor(options["mean_y_0"], torch.Size([options["dim_y"]])))

    def sample(self, x, num_samples=None):
        if num_samples==None:
            y = dist.Delta(self.y_0, event_dim=1).rsample()
        else:
            y = dist.Delta(self.y_0, event_dim=1).rsample(torch.Size((num_samples,)))
        return y

    def log_prob(self, x, y):
        # calculate log_prob
        log_prob_y = dist.Delta(self.y_0, event_dim=1).log_prob(y)
        return log_prob_y
    
    def entropy(self, x):
        # calculate entropy
        entropy_y = torch.tensor(0.0)
        return entropy_y

    @property
    def mean_y(self, x=None):
        return self.y_0

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def set_parameter(self, parameter_name, value):
        if parameter_name == "mean_y_0":
            self.y_0 = nn.Parameter(make_torch_tensor(value, torch.Size([self.options["dim_y"]])))
        else:
            raise ValueError("Parameter name not recognized")