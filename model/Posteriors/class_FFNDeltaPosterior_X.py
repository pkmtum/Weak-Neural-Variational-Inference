import torch
import torch.nn as nn
import warnings

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.nn_funcs.function_FFNBuilder import build_ffn


class FFNDeltaPosteriorX(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        self.dim_y = options["dim_y"]
        self.dim_x = options["dim_x"]
        self.hidden_layers = options["hidden_layers"]
        self.activation_func_name = options["activation_func_name"]

        self.rescale_input_flag = options.get("rescale_input_flag", False)
        self.rescale_input_mean = options.get("rescale_input_mean")
        self.rescale_input_std = options.get("rescale_input_std")

        self.NN = build_ffn(self.dim_y, self.dim_x, self.activation_func_name, self.hidden_layers)

    def sample(self, y, num_samples=None):
        #rescaling the input <3
        if self.rescale_input_flag:
            y = (y - self.rescale_input_mean) / self.rescale_input_std
        # sample from the posterior (it's a delta)
        return self.NN.forward(y)

    def log_prob(self, x, y):
        # calculate log_prob
        # I should never need this? 
        return torch.tensor(0.0)
    
    def entropy(self, y):
        # calculate entropy
        entropy_x = torch.tensor(0.0)
        return entropy_x
    
    def rescale_input_normal(self, mean, std):
        self.rescale_input_flag = True
        if self.rescale_input_mean is not None:
            warnings.warn("Rescaling input mean is already set. Overwriting it.")
        self.rescale_input_mean = mean
        if self.rescale_input_std is not None:
            warnings.warn("Rescaling input std is already set. Overwriting it.")
        if torch.allclose(std, torch.tensor(0.0)):
            warnings.warn("Rescaling input std is 0. Setting it to 1.")
            std = 1
        self.rescale_input_std = std