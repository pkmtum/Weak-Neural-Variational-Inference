import torch
from model.ParentClasses.class_ParentPriors import PriorX

class PriorHyperparameterGamma(PriorX):
    def __init__(self, prior_hyperparameter_options):
        # inherit from PriorX
        super().__init__(prior_hyperparameter_options)
        # extract options
        self.a_0 = self.options["a_0"]
        self.b_0 = self.options["b_0"]

    def sample(self, num_samples):
        return torch.distributions.Gamma(self.a_0, self.b_0).sample(num_samples)
    
    def log_prob(self, jumpPenalty):
        return torch.distributions.Gamma(self.a_0, self.b_0).log_prob(jumpPenalty).sum(dim=-1)

