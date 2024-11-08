import torch
import torch.distributions as dist
from model.ParentClasses.class_ParentPriors import PriorLambda


class PriorLambdaMVN(PriorLambda):
    def __init__(self, options):
        super().__init__(options)

        self.mean = torch.tensor(options["value"])
        self.cov = torch.tensor(options["cov"])  # some big number for uninformative prior

    def sample(self):
        Lambda = torch.sample("Lambda", dist.Normal(self.mean, self.cov))
        return Lambda
    
    def log_prob(self, Lambda):
        return dist.Normal(self.mean, self.cov).log_prob(Lambda)
    