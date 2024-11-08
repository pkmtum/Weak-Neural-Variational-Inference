import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class DiagMVNPosterior_Y(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)
        
        # dim of the MVN
        self.dim_y = options["dim_y"]

        # means & coveriance
        self.y_0 = nn.Parameter(make_torch_tensor(options["mean_y_0"], torch.Size([self.dim_y])))
        self.cov_y_parameters = nn.Parameter(make_torch_tensor(torch.log(torch.tensor(options["cov_y_0"])), torch.Size([self.dim_y])))

    def sample(self, num_samples=None):
        if num_samples==None:
            y = dist.normal.Normal(self.y_0, self.calc_scale_tril()).rsample()
        else:
            y = dist.normal.Normal(self.y_0, self.calc_scale_tril()).rsample(torch.Size((num_samples,)))
        return y

    def log_prob(self, y):
        # calculate log_prob
        log_prob_y = dist.normal.Normal(self.y_0,self.calc_scale_tril()).log_prob(y)
        return log_prob_y
    
    def entropy(self):
        # calculate entropy
        entropy_y = dist.normal.Normal(self.y_0, self.calc_scale_tril()).entropy()
        entropy_y = torch.sum(entropy_y)
        return entropy_y

    @property
    def mean_x(self):
        return None

    @property
    def mean_y(self):
        return self.y_0

    def calc_scale_tril(self):
        # compute torche covariance matrix
        # diagonal elements are exponentiated
        L = torch.exp(self.cov_y_parameters)
        # return torche covariance matrix
        return L