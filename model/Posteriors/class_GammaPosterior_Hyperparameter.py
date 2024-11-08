from typing import Any, Mapping
import torch
from model.ParentClasses.class_approximatePosterior import approximatePosterior

class PosteriorHyperparameterGamma(approximatePosterior):
    def __init__(self, posterior_hyperparameter_options, prior_hyperparameter_options):
        # inherit from PriorX
        super().__init__(posterior_hyperparameter_options)
        # extract options
        self.a = self.options["a"]
        self.b = self.options["b"]
        self.a_0 = prior_hyperparameter_options["a_0"]
        self.b_0 = prior_hyperparameter_options["b_0"]

    def sample(self, num_samples):
        return torch.distributions.Gamma(self.a, self.b).sample(num_samples)
    
    def expectation(self):
        return torch.div(self.a, self.b)
    
    def log_prob(self, Phi):
        return torch.distributions.Gamma(self.a, self.b).log_prob(Phi).sum(dim=-1)
    
    def entropy(self):
        return torch.distributions.Gamma(self.a, self.b).entropy().sum(dim=-1)
    
    def closedFormUpdate(self, jumps):
        jumps_mean_pow_2 = jumps.pow(2).mean(dim=-2)
        self.a = self.a_0 + 1 / 2
        self.b = self.b_0 + jumps_mean_pow_2 / 2

    # This is overloaded from the (grand-)parent class
    # Note that this only behaves like the real state_dict because it is fake. Not as flexible as the real one.
    def state_dict(self):
        return {"a": self.a,
                "b": self.b}
    
    # This is overloaded from the (grand-)parent class
    # Note that this only behaves like the real state_dict because it is fake. Not as flexible as the real one.
    def load_state_dict(self, state_dict, strict=True):
        if strict:
            assert state_dict.keys() == {"a", "b"}
        self.a = state_dict["a"]
        self.b = state_dict["b"]