import torch
import torch.distributions as dist

from model.ParentClasses.class_LambdaPosterior import LambdaPosterior


class LambdaConstant_posterior(LambdaPosterior):
    def __init__(self, lambda_posterior_options):
        # inherent stuff
        super().__init__(lambda_posterior_options)

        # Constant value
        self.constant = torch.tensor(self.options["value"])

    def guide(self, theta=None):
        Lambda = torch.sample("Lambda", dist.Delta(self.mean))

    def update_parameters(self, E_r_max_squared):
        # Nothing to update
        pass
    
    def set_value(self, value):
        self.constant = value

    @property
    def mean(self):
        return self.constant
        return self.constant
    def set_value(self, value):
        self.constant = value
        
    def sample(self):
        return self.constant
    
    def log_prob(self, Lambda):
        return torch.tensor(0.0)
    
    def entropy(self):
        return torch.tensor(0.0)