import torch
import torch.nn as nn
# import pyro
# import pyro.distributions as dist
import torch.distributions as dist
import math

class Likelihood(nn.Module):
    def __init__(self, PDE, val_lambda, theta_0, ThetaToField=None, WF_options=None) -> None:
        super().__init__()
        self.sigma_r = torch.sqrt(torch.tensor(1/val_lambda))
        self.working_theta = nn.Parameter(theta_0) # this is the one we want to optimize (requires_grad=True)
        self.theta = torch.tensor([]) # this is the set we already have (requires_grad=False)
        self.PDE = PDE
        self.dim_theta = theta_0.size()[0]
        self.full_theta = torch.eye(self.dim_theta)

        self.ThetaToField = ThetaToField
        self.min_rad = torch.tensor(WF_options.get("min_rad", 0.0))
        self.max_rad = torch.tensor(WF_options.get("max_rad", 1.0))
        self.radius_type = WF_options.get("radius_type", "linear")
        if self.radius_type not in ["linear", "log"]:
            raise ValueError("Radius type must be 'linear' or 'log'")

    def add_theta(self, theta):
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        if self.num_theta == 0:
            self.theta = theta
        else:
            self.theta = torch.concat((self.theta, theta), axis=0)

    def get_theta(self):
        if self.theta.numel() == 0:
            return list(self.parameters())[0].clone().detach().unsqueeze(0)
        else:
            return self.theta
    
    def reset_theta(self):
        self.theta = torch.tensor([])

    def reset_working_theta(self, theta_0):
        self.working_theta = nn.Parameter(theta_0)

    def set_full_theta(self):
        self.theta = torch.eye(self.working_theta.clone().detach().size()[0])

    def get_random_theta(self, num_samples=1, mean=0, std=1):
        return torch.randn(num_samples, self.working_theta.clone().detach().size()[0]) * std + mean

    def get_random_full_theta(self, num_samples=1):
        num = torch.randint(0, self.dim_theta, (num_samples,))
        return self.full_theta[num]
    
    def get_full_theta(self):
        return self.full_theta

    def get_circular_theta(self, num_samples=1):
        # create a weight function with a random center and a radius
        bfs_per_row = self.ThetaToField.bf_grid.size()[1]
        bfs_per_col = self.ThetaToField.bf_grid.size()[2]
        bf_grid = self.ThetaToField.bf_grid
        n_bfs = self.ThetaToField.n_bfs
        BC_mask = self.ThetaToField.mask_given_values
        # random center
        x, y = torch.randint(0, bfs_per_row, (num_samples,)), torch.randint(0, bfs_per_col, (num_samples,))
        center = bf_grid[:, x, y]
        # radius
        if self.radius_type == "linear":
            radius = torch.rand(num_samples) * (self.max_rad - self.min_rad) + self.min_rad
        elif self.radius_type == "log":
            radius = torch.exp(torch.rand(num_samples) * (torch.log(self.max_rad) - torch.log(self.min_rad)) + torch.log(self.min_rad))
        # calculate squared distance to center
        dist_center = torch.sub(bf_grid.repeat(num_samples,1,1,1).permute(2,3,1,0), center)
        sq_dist_center = torch.pow(dist_center, 2)
        d = sq_dist_center.sum(dim=2)
        # which points are in the circle
        mask = d <= radius ** 2
        points = mask.flatten(start_dim=0, end_dim=1)
        # flip the coin if this is a "x" or "y" - dimension theta
        flip = torch.randint(0, 2, (num_samples,)).to(torch.bool)
        # create theta
        theta = torch.zeros(n_bfs, num_samples, dtype=torch.bool)
        theta[n_bfs//2:, flip] = points[:, flip]
        theta[:n_bfs//2, ~flip] = points[:, ~flip]
        # delete elements where we have Dirichlet BCs
        theta = theta[~BC_mask, :]
        # to float and correct format
        theta = theta.to(torch.double)
        theta = theta.permute(1,0)
        return theta
        
    def log_prob(self, y_i, x_i):
        # fail for no theta
        if self.num_theta == 0:
            # Wikipedia: "Given no event (no data), the likelihood is 1; any non-trivial event will have a lower likelihood."
            # --> Thus log(1) = 0
            return torch.tensor(0.)
        
        # residual calculation for mulitple (or one) theta
        else:
            # get residual
            res = self.PDE.forward(y_i, x_i, self.theta)
        
            # calculate log_prob
            constant_term = self.num_theta * (- math.log(self.sigma_r) - math.log(math.sqrt(2 * math.pi)))  # dim = 1
            res_terms = torch.sum(torch.pow(res, 2) / (2 * torch.pow(self.sigma_r, 2)), dim=-1)  # dim = [num_samples, ..., num_theta, 1]
            return constant_term + res_terms  # dim = [num_samples, ..., 1]

    @property
    def num_theta(self):
        return self.theta.size(dim=0)