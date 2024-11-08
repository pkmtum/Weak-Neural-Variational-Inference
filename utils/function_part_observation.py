import numpy as np


class PartObservation:
    def __init__(self):
        # index is for dofs (there could be multiple dofs on the same node)
        self.index = []
        self.dim = None
        self.full_obs = False

    def filter(self, u):
        # Special case: If I want to observe the full field
        if self.full_obs:
            return u
        if self.dim == 2:
            return u[..., self.index[0], self.index[1]]
        else:
            return u[..., self.index[0], self.index[1], self.index[2]]

    # def get_rand_index(self, percent, len_y):
    #     len_index = int(np.ceil(len_y * percent))
    #     self.index = random.sample(range(len_y), len_index)
    #     self.index.sort()
    #     print(f"Out of {len_y} data points, we observe {len_index}, i.e. ~ {percent*100} %.")

    def get_regular_index(self, tensor, n_1, n_2, obs_on_boundary=False, full_obs=False):
        # Special case: If I want to observe the full field
        if full_obs:
            self.full_obs = True
            return

        if tensor.dim() == 2:
            n_y, n_x = tensor.shape  # e.g. 50 x 50
        elif tensor.dim() == 3:
            n_dim, n_y, n_x = tensor.shape  # e.g. 2 x 50 x 50
        if obs_on_boundary:
            step_y = n_y // (n_2 - 1)
            step_x = n_x // (n_1 - 1)
            lin_x = np.linspace(0, n_x-1, n_1).astype(int)
            lin_y = np.linspace(0, n_y-1, n_2).astype(int)
        else:
            step_y = n_y // n_2
            step_x = n_x // n_1
            lin_x = np.linspace(step_x/2, n_x- step_x/2, n_1).astype(int)
            lin_y = np.linspace(step_y/2, n_y- step_y/2, n_2).astype(int)
        if tensor.dim() == 2:
            self.index = np.ix_(lin_y, lin_x)
            self.dim = 2
        elif tensor.dim() == 3:
            lin_dim = np.arange(0, n_dim, 1).astype(int)
            self.index = np.ix_(lin_dim, lin_y, lin_x)
            self.dim = 3

    # def is_index(self, index):
    #     index_bool = False
    #     if index in self.index:
    #         index_bool = True
    #     return index_bool
