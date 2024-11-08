import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt
from utils.torch_funcs.function_index_to_2D_index import function_index_to_2D_index
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu

class ChebyshevPolynomials(ParameterToField):
    def __init__(self, options):
        # center for RBF
        self.bf_grid = options["bf_grid"]
        self.dim_bf_x, self.dim_bf_y = self.bf_grid.size(dim=1), self.bf_grid.size(dim=2)  # i guess

        self.max_x = options["s_grid"][1, :, :].max()
        self.min_x = options["s_grid"][1, :, :].min()
        self.max_y = options["s_grid"][0, :, :].max()
        self.min_y = options["s_grid"][0, :, :].min()

        # this is the entire length of the grid
        self.len_x = (self.max_x - self.min_x)
        self.len_y = (self.max_y - self.min_y)
        
        # bounds for the normed grid
        a = torch.tensor(-1 + 1e-10)
        b = torch.tensor(1 - 1e-10)

        # rescale between -1 and 1 for Chebyshev polynomials
        self.normed_x = (b-a) * (options["s_grid"][1, :, :] - self.min_x) / self.len_x + a
        self.normed_y = (b-a) * (options["s_grid"][0, :, :] - self.min_y) / self.len_y + a
        
        # arccos of normed_x and normed_y
        self.acos_normed_x = torch.acos(self.normed_x)
        self.acos_normed_y = torch.acos(self.normed_y)

        super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])

    def _eval_bf(self, i_bf):

        # get the indices of the basis function
        indices = function_index_to_2D_index(i_bf, self.dim_bf_x)
        # calculate Chebyshev polynomials
        Chebyshev_x = torch.cos(indices[0] * self.acos_normed_x)
        Chebyshev_y = torch.cos(indices[1] * self.acos_normed_y)
        return torch.mul(Chebyshev_x, Chebyshev_y)

    # DERIVATIVES ARE INDEPENDENT :)
    def _eval_d_bf__ds_x(self, i_bf):
        # get the indices of the basis function
        indices = function_index_to_2D_index(i_bf, self.dim_bf_x)
        # d1 = d T_n(x) * T_m(y) (* c) / dx = T_m(y) * d T_n(x) / dx (* c)
        d1 = torch.cos(indices[0] * self.acos_normed_x)
        # d2 = d cos ( n * arccos(x) ) / dx = n * sin ( n * arccos(x) ) / sqrt(1 - x^2)
        d2 = indices[1] * torch.sin(indices[1] * self.acos_normed_y) / torch.sqrt(1 - self.normed_y ** 2)
        # d3 = d (2 * (x - min_x) / len_x - 1) / dx = 2 / len_x
        d3 = 2 / self.len_y
        return torch.mul(torch.mul(d1, d2), d3)

    def _eval_d_bf__ds_y(self, i_bf):
        # get the indices of the basis function
        indices = function_index_to_2D_index(i_bf, self.dim_bf_x)
        # d1 = d T_n(x) * T_m(y) (* c) / dx = T_m(y) * d T_n(x) / dx (* c)
        d1 = torch.cos(indices[1] * self.acos_normed_y)
        # d2 = d cos ( n * arccos(x) ) / dx = n * sin ( n * arccos(x) ) / sqrt(1 - x^2)
        d2 = indices[0] * torch.sin(indices[0] * self.acos_normed_x) / torch.sqrt(1 - self.normed_x ** 2)
        # d3 = d (2 * (x - min_x) / len_x - 1) / dx = 2 / len_x
        d3 = 2 / self.len_x
        return torch.mul(torch.mul(d1, d2), d3)

    def _plot_func(self, fields, title=None):
        numOfShFuncs_0 = self.bf_grid.size(dim=1)
        numOfShFuncs_1 = self.bf_grid.size(dim=2)
        gridSize_0 = self.s_grid.size(dim=1)
        gridSize_1 = self.s_grid.size(dim=2)
        fig, axs = plt.subplots(numOfShFuncs_0, numOfShFuncs_1, figsize=(12, 12))
        if title is not None:
            fig.suptitle(title, fontsize=16)
        temp = fields.view(numOfShFuncs_0, numOfShFuncs_1, gridSize_0, gridSize_1)
        vmax = torch.max(temp)
        vmin = torch.min(temp)
        s_grid, temp = cuda_to_cpu(self.s_grid, temp)
        for i in range(numOfShFuncs_0):
            for j in range(numOfShFuncs_1):
                axs[i, j].pcolormesh(s_grid[0, :, :], s_grid[1, :, :], temp[i, j, :, :],
                                     cmap='coolwarm', shading='auto')#, vmin=vmin, vmax=vmax)
                axs[i, j].set_title("Shape Function: " + str(i * numOfShFuncs_1 + j))
                axs[i, j].set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def plot_bfs(self):
        self._plot_func(self.bfs, 'Chebychev Basis Functions')

    def plot_bfs_with_BC(self):
        self._plot_func(self.bfs_with_BC, 'Chebychev Basis Functions with BC mask')

    def plot_d_bf_with_BC__ds_x(self):
        self._plot_func(self.d_bf_with_BC__d_s_x, 'd_bf_with_BC__d_s_x')

    def plot_d_bf_with_BC__ds_y(self):
        self._plot_func(self.d_bf_with_BC__d_s_y, 'd_bf_with_BC__d_s_y')