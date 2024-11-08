import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor
import ParentClasses.class_approximatePosterior as approximatePosterior


# TODO: fit this more to the general structure, test it, put in automatic selection
class MVN_FFNN_posterior(approximatePosterior):
    def __init__(self, options):
        """
        Form of options:
        Dict with keys: "NN_mu", "NN_cholesky", "dim_y"
        where each key strating with "NN" is a dict with keys "dim_in", "dim_out", "dim_hidden", "num_hidden_layers", "activation"
        """
        super().__init__(options)

        # indecies for cholesky matrix
        self.dim_y = options["dim_y"]
        self.lower_triangular_index = torch.tril_indices(self.dim_y, self.dim_y)
        self.diag_index = np.diag_indices(self.dim_y)

        NN = options["NN_mu"]
        hidden_sizes1 = torch.tensor([int(NN["dim_hidden"]) for _ in range(NN["num_hidden_layers"])])
        self.network1 = self._build_network(NN["dim_in"], hidden_sizes1, NN["dim_out"], NN["activation"])

        NN = options["NN_cholesky"]
        hidden_sizes2 = torch.tensor([int(NN["dim_hidden"]) for _ in range(NN["num_hidden_layers"])])
        self.network2 = self._build_network(NN["dim_in"], hidden_sizes2, NN["dim_out"], NN["activation"])
        
    def _build_network(self, input_size, hidden_sizes, output_size, activation):
        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            layers.append(self._get_activation(activation))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    def _mu_and_cholesky(self, x):
        mu = self.network1(x)
        cholesky = self.network2(x)
        return mu, cholesky

    def sample(self, x,  num_samples=None):
        mu, cholesky = self._mu_and_cholesky(x)
        if num_samples==None:
            y = dist.MultivariateNormal(mu, scale_tril=self.calc_scale_tril(cholesky)).rsample()
        else:
            y = dist.MultivariateNormal(mu, scale_tril=self.calc_scale_tril(cholesky)).rsample(torch.Size((num_samples,)))
        return y

    def log_prob(self, x, y):
        mu, cholesky = self._mu_and_cholesky(x)
        # calculate log_prob
        log_prob_x = dist.MultivariateNormal(mu, scale_tril=self.calc_scale_tril(cholesky)).log_prob(y)
        return log_prob_x
    
    def entropy(self, x):
        mu, cholesky = self._mu_and_cholesky(x)
        # calculate entropy
        entropy_x = dist.MultivariateNormal(mu, scale_tril=self.calc_scale_tril(cholesky)).entropy()
        return entropy_x

    @property
    def mean_y(self, x):
        mu, cholesky = self._mu_and_cholesky(x)
        return mu

    def calc_scale_tril(self, parameters):
        # compute torche covariance matrix
        # torche parameter are positioned as =(0,0), (1,0), (1,1), (2,0), (2,1), (2,3) ...
        # for N dim matrix, check torche number of elements in torche lower triangular matrix
        # if it is N*(N+1)/2 torchen it is a lower triangular matrix
        assert len(parameters) == self.dim_y*(self.dim_y+1)/2,\
        "torche number of parameters for torche covariance matrix is not correct"
        # compute torche lower triangular matrix
        L = torch.zeros(self.dim_y, self.dim_y)
        L[self.lower_triangular_index[0], self.lower_triangular_index[1]] = parameters
        # diagonal elements are exponentiated
        L[self.diag_index] = torch.exp(L[self.diag_index])
        # return torche covariance matrix
        return L
    

options = {
    "NN_mu": {
        "dim_in": 10,
        "dim_out": 5,
        "dim_hidden": 20,
        "num_hidden_layers": 2,
        "activation": "relu"
    },
    "NN_cholesky": {
        "dim_in": 10,
        "dim_out": 5,
        "dim_hidden": 20,
        "num_hidden_layers": 2,
        "activation": "relu"
    },
    "dim_y": 5
}
A =  MVN_FFNN_posterior(options)