import torch
import torch.distributions as dist

from model.ParentClasses.class_LambdaPosterior import LambdaPosterior


class LambdaGamma_posterior(LambdaPosterior):
    def __init__(self, lambda_posterior_options):
        # inherent stuff
        super().__init__(lambda_posterior_options)

        # this is important for the closed form update
        self.a_prior = self.options["a_0"]
        self.b_prior = self.options["b_0"]
        self.N = self.options["N"]

        # these are the actual gamma parameters
        self.a = self.options["a_0"]
        self.b = self.options["b_0"]

        # initialize the parameters
        self.update_parameters(torch.tensor(10**10))  # some big number as the initial error

    def guide(self, theta):
        Lambda = torch.sample("Lambda", dist.Delta(self.mean))

    def update_parameters(self, E_r_max_squared):
        self.a = self.N / 2 + self.a_prior
        self.b = self.b_prior + self.N * E_r_max_squared / 2

    @property
    def mean(self):
        return self.N * torch.tensor(self.a) / self.b
