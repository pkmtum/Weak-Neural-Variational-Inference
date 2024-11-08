import torch
import torch.distributions as dist

# my imports
from model.ParentClasses.class_ParentPriors import PriorX
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class PriorXMvN(PriorX):
    def __init__(self, options):
        """
        This is a multivariate normal x prior

        :param options: dict of options.

        Form in input-file:
        dict(mean=0,
            sigma=1,
            )
        Additionally: dim: number of unknowns
        """

        super().__init__(options)

        # unpack options
        dim = options["dim"]
        mean_x = options["mean"]
        sigma_x = options["sigma"]

        # make sure mean_x is a vector
        self.mean_x = make_torch_tensor(mean_x, torch.Size([dim]))
        # make sure sigma_x is a tensor
        self.sigma_x = make_torch_tensor(sigma_x, torch.Size([dim, dim]))
        self.cholesky = torch.linalg.cholesky(self.sigma_x)

        # double-check if the sizes are really correct now
        # (someone might fuck up their manual tensor input or provide wrong dim)
        assert self.sigma_x.size() == torch.Size([dim, dim])
        assert self.mean_x.size() == torch.Size([dim])

    def sample(self):
        x = torch.sample("x", dist.MultivariateNormal(loc=self.mean_x, scale_tril=self.cholesky))
        return x

    def log_prob(self, x):
        return dist.MultivariateNormal(loc=self.mean_x, scale_tril=self.cholesky).log_prob(x)