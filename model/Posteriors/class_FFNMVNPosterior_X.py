import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.nn_funcs.function_FFNBuilder import build_ffn
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class FFNMVNPosteriorX(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        self.dim_y = options["dim_y"]
        self.dim_x = options["dim_x"]
        self.hidden_layers = options["hidden_layers"]
        self.activation_func_name = options["activation_func_name"]
        self.learn_cov_seperatly = options["learn_cov_seperatly"]

        self.NN = build_ffn(self.dim_y, self.dim_x, self.activation_func_name, self.hidden_layers)
        if self.learn_cov_seperatly:
            helper_cov_x = make_torch_tensor(torch.log(torch.tensor(options["cov_x_0"])), torch.Size([self.dim_x, self.dim_x]))
            self.lower_triangular_index = torch.tril_indices(self.dim_x, self.dim_x)
            self.diag_index = np.diag_indices(self.dim_x)
            cov_x_init = helper_cov_x[self.lower_triangular_index[0], self.lower_triangular_index[1]]
            self.cov_x_parameters = nn.Parameter(cov_x_init)
        else:
            raise NotImplementedError("This is not implemented yet. Please select learn_cov_seperatly=True.")
        
        # intermetdiate variables
        self.mu_x = None
        self.cov_x = None

    def sample(self, y, num_samples=None):
        # calculate mean via NN
        self.mu_x = self.NN.forward(y)

        # depends on if we learn the covariance with a NN or not
        if self.learn_cov_seperatly:
            L = self.calc_scale_tril()
        else:
            raise NotImplementedError("This is not implemented yet. Please select learn_cov_seperatly=True.")

        # sample from distribution
        # num_samples is not used here because the dimensions of y gives the number of samples.
        # In theory, we can sample from the distribution num_samples times given one y.
        # This, however, would mess up my structure of the code.
        x = dist.MultivariateNormal(self.mu_x, scale_tril=L).rsample()
        return x
    
    def log_prob(self, x, y):
        # depends on if we learn the covariance with a NN or not
        if self.learn_cov_seperatly:
            L = self.calc_scale_tril()
        else:
            raise NotImplementedError("This is not implemented yet. Please select learn_cov_seperatly=True.")
        # calculate log_prob (ignore y and use self.mu_x and self.cov_x to not double calculations)
        log_prob_x = dist.MultivariateNormal(self.mu_x, scale_tril=L).log_prob(x)
        return log_prob_x
    
    def entropy(self, y):
        # depends on if we learn the covariance with a NN or not
        if self.learn_cov_seperatly:
            L = self.calc_scale_tril()
        else:
            raise NotImplementedError("This is not implemented yet. Please select learn_cov_seperatly=True.")
        # calculate entropy
        entropy_x = dist.MultivariateNormal(self.mu_x, scale_tril=L).entropy()
        return entropy_x

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
        