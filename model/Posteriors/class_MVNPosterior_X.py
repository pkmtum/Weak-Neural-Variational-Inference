import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class MVNPosterior_X(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        # means
        self.x_0 = nn.Parameter(make_torch_tensor(options["mean_x_0"], torch.Size([options["dim_x"]])))

        # covariance
        # this part deals with how I want to initilize the matrix
        self.dim_x = options["dim_x"]
        helper_cov_x = make_torch_tensor(torch.log(torch.tensor(options["cov_x_0"])), torch.Size([self.dim_x, self.dim_x]))
        self.lower_triangular_index = torch.tril_indices(self.dim_x, self.dim_x)
        self.diag_index = np.diag_indices(self.dim_x)
        cov_x_init = helper_cov_x[self.lower_triangular_index[0], self.lower_triangular_index[1]]
        self.cov_x_parameters = nn.Parameter(cov_x_init)
        self.cov_x = self.calc_scale_tril()
        # self.cov_x = self.cov()  --> i have it as a property
        
    def sample(self, num_samples=None):
        if num_samples==None:
            x = dist.MultivariateNormal(self.x_0, scale_tril=self.calc_scale_tril()).rsample()
        else:
            x = dist.MultivariateNormal(self.x_0, scale_tril=self.calc_scale_tril()).rsample(torch.Size((num_samples,)))
        return x

    def log_prob(self, x):
        # calculate log_prob
        log_prob_x = dist.MultivariateNormal(self.x_0, scale_tril=self.calc_scale_tril()).log_prob(x)
        return log_prob_x
    
    def entropy(self):
        # calculate entropy
        entropy_x = dist.MultivariateNormal(self.x_0, scale_tril=self.calc_scale_tril()).entropy()
        return entropy_x

    @property
    def mean_x(self):
        return self.x_0

    @property
    def mean_y(self, x=None):
        return self.y_0

    def calc_scale_tril(self):
        # compute torche covariance matrix
        # torche parameter are positioned as =(0,0), (1,0), (1,1), (2,0), (2,1), (2,3) ...
        # for N dim matrix, check torche number of elements in torche lower triangular matrix
        # if it is N*(N+1)/2 torchen it is a lower triangular matrix
        assert len(self.cov_x_parameters) == self.dim_x*(self.dim_x+1)/2,\
        "torche number of parameters for torche covariance matrix is not correct"
        # compute torche lower triangular matrix
        L = torch.zeros(self.dim_x, self.dim_x)
        L[self.lower_triangular_index[0], self.lower_triangular_index[1]] = self.cov_x_parameters
        # diagonal elements are exponentiated
        L[self.diag_index] = torch.exp(L[self.diag_index])
        # return torche covariance matrix
        return L
        
