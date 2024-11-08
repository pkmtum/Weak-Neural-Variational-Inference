import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class MVNPosterior_y(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        # means
        self.y_0 = nn.Parameter(make_torch_tensor(options["mean_y_0"], torch.Size([options["dim_y"]])))

        # covariance
        # this part deals with how I want to initilize the matrix
        self.dim_y = options["dim_y"]
        helper_cov_y = make_torch_tensor(torch.log(torch.tensor(options["cov_y_0"])), torch.Size([self.dim_y, self.dim_y]))
        self.lower_triangular_index = torch.tril_indices(self.dim_y, self.dim_y)
        self.diag_index = np.diag_indices(self.dim_y)
        cov_y_init = helper_cov_y[self.lower_triangular_index[0], self.lower_triangular_index[1]]
        self.full_cov_y_parameters = nn.Parameter(cov_y_init)
        self.cov_y = self.calc_scale_tril()
        # self.cov_y = self.cov()  --> i have it as a property
        
    def sample(self, num_samples=None):
        if num_samples==None:
            y = dist.MultivariateNormal(self.y_0, scale_tril=self.calc_scale_tril()).rsample()
        else:
            y = dist.MultivariateNormal(self.y_0, scale_tril=self.calc_scale_tril()).rsample(torch.Size((num_samples,)))
        return y

    def log_prob(self, y):
        # calculate log_prob
        log_prob_y = dist.MultivariateNormal(self.y_0, scale_tril=self.calc_scale_tril()).log_prob(y)
        return log_prob_y
    
    def entropy(self):
        # calculate entropy
        entropy_y = dist.MultivariateNormal(self.y_0, scale_tril=self.calc_scale_tril()).entropy()
        return entropy_y

    @property
    def mean_y(self, x=None):
        return self.y_0

    def calc_scale_tril(self):
        # compute torche covariance matrix
        # torche parameter are positioned as =(0,0), (1,0), (1,1), (2,0), (2,1), (2,3) ...
        # for N dim matrix, check torche number of elements in torche lower triangular matrix
        # if it is N*(N+1)/2 torchen it is a lower triangular matrix
        assert len(self.full_cov_y_parameters) == self.dim_y*(self.dim_y+1)/2,\
        "torche number of parameters for torche covariance matrix is not correct"
        # compute torche lower triangular matrix
        L = torch.zeros(self.dim_y, self.dim_y)
        L[self.lower_triangular_index[0], self.lower_triangular_index[1]] = self.full_cov_y_parameters
        # diagonal elements are exponentiated
        L[self.diag_index] = torch.exp(L[self.diag_index])
        # return torche covariance matrix
        return L
    
    def set_cov_with_MVN_cov(self, parameters):
        # initilize new covariance matrix parameters
        cov_0 = torch.zeros_like(self.lower_triangular_index[0], dtype=torch.float64)
        # get where the diagonal elements are
        diagonal_index = torch.where(self.lower_triangular_index[0] == self.lower_triangular_index[1])[0]
        # set diagonal elements
        cov_0[diagonal_index] = parameters # no need to log, because it is already logged when loading!
        # save as nn.parameters
        self.set_parameter("full_cov_y_parameters", cov_0)
        print("Loaded DiagMVN covariance parameters for MVN covariance matrix.")
