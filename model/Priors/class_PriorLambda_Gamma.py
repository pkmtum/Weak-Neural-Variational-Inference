import torch
import torch.distributions as dist
from model.ParentClasses.class_ParentPriors import PriorLambda


class PriorLambdaGamma(PriorLambda):
    def __init__(self, options):
        super().__init__(options)

        self.a_0 = options["a_0"]
        self.b_0 = options["b_0"]

    def sample(self):
        Lambda = torch.sample("Lambda", dist.Gamma(self.a_0, self.b_0))
        return Lambda
    