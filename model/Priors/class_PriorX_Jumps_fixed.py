import torch
import torch.distributions as dist
from model.ParentClasses.class_ParentPriors import PriorX

class JumpPrior_fixed(PriorX):
    def __init__(self, prior_jumps_options) -> None:
        # inherit from nn.Module
        super().__init__(None)
        # Create the mask
        self.Neighbor_Mask = None # self.neighbor_mask(field)
        self.penalty = torch.tensor(prior_jumps_options["jumpPenalty"])
    
    def log_prob(self, x):
        jumps = self.eval_jumps(x)
        return dist.Normal(0, self.penalty.pow(-1)).log_prob(jumps).sum(dim=-1)
    
    def sample(self, x, num_samples):
        jumps = self.eval_jumps(x)
        return dist.Normal(jumps, self.penalty.pow(-1)).sample(num_samples)

    def eval_jumps(self, field):
        return torch.einsum('ij,...j->...i', self.Neighbor_Mask, field)

    def neighbor_mask(self, X):
        """
        Returns a mask to calculate the difference between each neighbor in X.

        Parameters
        ----------
        X : torch.Tensor
            Shape (n_row, n_col)
        
        """

        # How does my parameter vector relate to the field?
        
        n_row, n_col = X.shape
        num_neighbors_vertical = (n_row-1) * n_col 
        num_neighbors_horizontal = n_row * (n_col-1)
        num_neighbors = num_neighbors_horizontal + num_neighbors_vertical
        mask = torch.zeros((num_neighbors,n_row, n_col))
        # vertical neighbors
        for j in range(n_col):
            for i in range(n_row - 1):
                num = j * (n_row-1) + i
                mask[num, i, j] = 1
                mask[num, i + 1, j] = -1
        # horizontal neighbors
        for i in range(n_row):
            for j in range(n_col - 1):
                num = i * (n_col-1) + j + num_neighbors_vertical
                mask[num, i, j] = 1
                mask[num, i, j + 1] = -1
        
        # the parameters are flat, not a field
        return mask.flatten(start_dim=-2, end_dim=-1)

    def set_neighbour_mask(self, neighbor_mask):
        self.Neighbor_Mask = neighbor_mask


# X = torch.rand((2,2))
# mask = neighbor_mask(X)
# print(mask)
# X_samples = torch.rand((3,2,2))
# Distance = torch.einsum('ijk,...jk->...i', mask, X_samples)
# print(Distance)

# A = torch.rand((2,2))
# prior = JumpPrior(A, torch.tensor(1.), False)
# B = torch.rand((5,2,2))
# print(prior.log_likelihood(B))

