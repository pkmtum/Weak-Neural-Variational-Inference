import torch
import torch.distributions as dist

# my imports
from model.ParentClasses.class_ParentPriors import PriorX
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor
from utils.function_part_observation import PartObservation


class PriorXFieldNormal(PriorX):
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
        # dim = options["dim"]
        s_grid1 = options["dim_s_grid_1"]
        s_grid2 = options["dim_s_grid_2"]
        mean_x = options["mean"]
        sigma_x = options["sigma"]
        self.XToField = options["X1ToField"]
        bf_grid_size1 = options["dim_bf_grid1"]
        bf_grid_size2 = options["dim_bf_grid2"]
        self.obs = PartObservation()
        self.obs.get_regular_index(torch.zeros(s_grid1, s_grid2), bf_grid_size1, bf_grid_size2, obs_on_boundary=True)

        # make sure mean_x is a vector
        self.mean_x = make_torch_tensor(mean_x, torch.Size([bf_grid_size1*bf_grid_size2]))
        # make sure sigma_x is a tensor
        self.sigma_x = make_torch_tensor(sigma_x, torch.Size([bf_grid_size1*bf_grid_size2]))
        # self.cholesky = torch.linalg.cholesky(self.sigma_x)

        # double-check if the sizes are really correct now
        # (someone might fuck up their manual tensor input or provide wrong dim)
        assert self.sigma_x.size() == torch.Size([bf_grid_size1*bf_grid_size2])
        assert self.mean_x.size() == torch.Size([bf_grid_size1*bf_grid_size2])

    def sample(self):
        x = torch.sample("x", dist.normal.Normal(loc=self.mean_x, scale=self.sigma_x))
        return x

    def log_prob(self, x):
        MF = self.XToField.eval(x)
        MF_points = self.obs.filter(MF)
        MF_points = torch.flatten(MF_points, start_dim=-2, end_dim=-1)
        return torch.sum(dist.normal.Normal(loc=self.mean_x, scale=self.sigma_x).log_prob(MF_points), dim=-1)

