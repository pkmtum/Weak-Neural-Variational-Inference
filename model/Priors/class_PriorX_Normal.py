import torch
import torch.distributions as dist

# my imports
from model.ParentClasses.class_ParentPriors import PriorX
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class PriorXNormal(PriorX):
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
        # dim_s = options["dim_s_grid"]
        mean_x = options["mean"]
        sigma_x = options["sigma"]
        self.XToField = options["X1ToField"]

        # make sure mean_x is a vector
        self.mean_x = make_torch_tensor(mean_x, torch.Size([dim]))
        # make sure sigma_x is a tensor
        self.sigma_x = make_torch_tensor(sigma_x, torch.Size([dim]))
        # self.cholesky = torch.linalg.cholesky(self.sigma_x)

        # double-check if the sizes are really correct now
        # (someone might fuck up their manual tensor input or provide wrong dim)
        assert self.sigma_x.size() == torch.Size([dim])
        assert self.mean_x.size() == torch.Size([dim])

    def sample(self):
        x = torch.sample("x", dist.normal.Normal(loc=self.mean_x, scale=self.sigma_x))
        return x

    def log_prob(self, x):
        if x.size()[-1] != self.mean_x.size()[-1]:
            raise KeyError("This is the wrong prior for this. Consider FieldNormal prior.")
            # x = self.XToField.eval(x)
            # x = torch.flatten(x, start_dim=-2, end_dim=-1)
        return torch.sum(dist.normal.Normal(loc=self.mean_x, scale=self.sigma_x).log_prob(x), dim=-1)

            