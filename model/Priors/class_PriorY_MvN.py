import torch
import torch.distributions as dist

# my imports
from model.ParentClasses.class_ParentPriors import PriorY
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class PriorYMvN(PriorY):
    def __init__(self, options):
        super().__init__(options)

        dim = options["dim"]
        mean_y = options["mean"]
        sigma_y = options["sigma"]

        # make sure mean_y is a vector
        self.mean_y = make_torch_tensor(mean_y, torch.Size([dim]))
        # make sure sigma_y is a tensor
        self.sigma_y = make_torch_tensor(sigma_y, torch.Size([dim, dim]))
        self.cholesky = torch.linalg.cholesky(self.sigma_y)

        # check if the sizes are really correct now 
        # (someone might fuck up their manual tensor input or provide wrong dim)
        assert self.sigma_y.size() == torch.Size([dim, dim])
        assert self.mean_y.size() == torch.Size([dim])

    def sample(self, x=None):
        y = torch.sample("y", dist.MultivariateNormal(loc=self.mean_y, scale_tril=self.cholesky))
        return y

    def log_prob(self, y, x=None):
        return dist.MultivariateNormal(loc=self.mean_y, scale_tril=self.cholesky).log_prob(y)