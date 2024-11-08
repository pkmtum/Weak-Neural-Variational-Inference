import torch
import torch.nn as nn
# import torch.distributions as dist
# import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
# from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class ReducedMVNPosterior_x(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        self.dim_x = options["dim_x"]
        self.reduced_dim_x = options["reduced_dim_x"]

        # means
        self.x_0 = nn.Parameter(make_torch_tensor(options["mean_x_0"], torch.Size([options["dim_x"]])))

        self.sigma_x = nn.Parameter(make_torch_tensor(torch.log(torch.tensor(options["cov_x_0"])), torch.Size([options["dim_x"]])))
        V_0 = torch.zeros(options["dim_x"], options["reduced_dim_x"])
        self.V_x = nn.Parameter(V_0)
        
    def sample(self, num_samples=None):
        # reparameterization trick for sampling (but split into two parts)
        x = self.x_0 + torch.einsum('i,ji->ji',torch.exp(self.sigma_x), torch.randn(torch.Size((num_samples, self.dim_x)))) + torch.einsum('ik,lk->li', self.V_x, torch.randn(torch.Size((num_samples, self.reduced_dim_x))))
        return x

    def log_prob(self, x):
        # calculate log_prob
        raise NotImplementedError("I should not need this, only the entropy.")

    def entropy(self):
        # calculate entropy: 0.5 * log(det(V * V^T + sigma_x^2 * I)) + D/2 * ln(2*pi) with
        # - "D/2 * ln(2*pi)" is a constant and is therefore not considered
        # - "log(det(...))": can be calculated in one step by torch.logdet()
        # entropy_x = 0.5 * torch.logdet(self.V_x @ self.V_x.t() + torch.diag_embed(torch.pow(torch.exp(self.sigma_x), 2)))

        # Forget what was written above. This is the real shit: https://en.wikipedia.org/wiki/Matrix_determinant_lemma#Generalization
        diag_A = torch.pow(torch.exp(self.sigma_x), 2)
        logdet_A = torch.sum(torch.log(diag_A)) # det(diag(x)) is prod(x_i)
        A_inv = torch.diag_embed(1/diag_A)
        I_m = torch.eye(self.reduced_dim_x)
        Vt_A_inv_U = self.V_x.t() @ A_inv @ self.V_x # this is [red_x, red_x] !
        logdet_Vt_A_inv_U = torch.logdet(Vt_A_inv_U + I_m) # cheap cause low dimensional
        logdet = logdet_Vt_A_inv_U + logdet_A
        entropy_x = 0.5 * logdet
        return entropy_x

    @property
    def mean_x(self, x=None):
        return self.x_0

    # def calc_scale_tril(self):
    #     # compute torche covariance matrix
    #     # torche parameter are positioned as =(0,0), (1,0), (1,1), (2,0), (2,1), (2,3) ...
    #     # for N dim matrix, check torche number of elements in torche lower triangular matrix
    #     # if it is N*(N+1)/2 torchen it is a lower triangular matrix
    #     assert len(self.full_cov_y_parameters) == self.dim_y*(self.dim_y+1)/2,\
    #     "torche number of parameters for torche covariance matrix is not correct"
    #     # compute torche lower triangular matrix
    #     L = torch.zeros(self.dim_y, self.dim_y)
    #     L[self.lower_triangular_index[0], self.lower_triangular_index[1]] = self.full_cov_y_parameters
    #     # diagonal elements are exponentiated
    #     L[self.diag_index] = torch.exp(L[self.diag_index])
    #     # return torche covariance matrix
    #     return L
    
    # def set_cov_with_MVN_cov(self, parameters):
    #     # initilize new covariance matrix parameters
    #     cov_0 = torch.zeros_like(self.lower_triangular_index[0], dtype=torch.float64)
    #     # get where the diagonal elements are
    #     diagonal_index = torch.where(self.lower_triangular_index[0] == self.lower_triangular_index[1])[0]
    #     # set diagonal elements
    #     cov_0[diagonal_index] = parameters # no need to log, because it is already logged when loading!
    #     # save as nn.parameters
    #     self.set_parameter("full_cov_y_parameters", cov_0)
    #     print("Loaded DiagMVN covariance parameters for MVN covariance matrix.")