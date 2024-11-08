import torch
import torch.nn as nn

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.nn_funcs.function_CNNBuilder import build_cnn
from utils.nn_funcs.function_FFNBuilder import build_ffn

class CNNDeltaPosteriorX(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)

        self.YToField = options["YToField"]
        if "output_trafo" in options:
            self.output_trafo = options["output_trafo"]
        else:
            self.output_trafo = None
        
        # define the neural network
        self.CNN = build_cnn(options["input_channels"], 
                             options["activation_func_name"], 
                             options["feature_dim_layers"],
                             average_pooling_layers=options["average_pooling_layers"])
        if options["average_pooling_layers"] is bool:
            pic_size_reduction = 2**(len(options["feature_dim_layers"])*options["average_pooling_layers"])
        else:
            pic_size_reduction = 2**sum(options["average_pooling_layers"])
        self.NN = build_ffn(int((options["pic_size"] / pic_size_reduction)**2 * options["feature_dim_layers"][-1]), 
                            options["output_size"], 
                            options["activation_func_name"], 
                            options["hidden_layers"])

    def sample(self, y, num_samples=None):
        # sample from the posterior
        u = self.YToField.eval(y)
        x = self.CNN(u) # CNN
        x = x.view(x.size(0), -1) # flatten
        x = self.NN(x) # FFNN
        if self.output_trafo is not None:
            x = torch.einsum('ij,kj->ki', self.output_trafo, x) # self.output_trafo @ x 
        return x

    def log_prob(self, x, y):
        # calculate log_prob
        # I should never need this? 
        return torch.tensor(0.0)
    
    def entropy(self, y):
        # calculate entropy
        entropy_x = torch.tensor(0.0)
        return entropy_x

    @property
    def mean_x(self):
        return self.x_0
