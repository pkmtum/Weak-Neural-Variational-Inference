import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu
import matplotlib.pyplot as plt


class BumpBasisFunction(ParameterToField):
    def __init__(self, options):
        # center for RBF
        self.bf_grid = options["bf_grid"]

        self.max_x = options["s_grid"][1, :, :].max()
        self.min_x = options["s_grid"][1, :, :].min()
        self.max_y = options["s_grid"][0, :, :].max()
        self.min_y = options["s_grid"][0, :, :].min()

        self.len_x = (self.max_x - self.min_x) / (self.bf_grid.size(dim=1) - 1)
        self.len_y = (self.max_y - self.min_y) / (self.bf_grid.size(dim=2) - 1)

        super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])

    def _eval_bf(self, i_bf):
        center_x, center_y = self._center(i_bf)
        radius_trafo = self._radius_trafo(center_x, center_y)
        field = torch.exp(-1 / (1 - radius_trafo ** 2))
        field[radius_trafo > 1] = 0
        return field

    def _eval_d_bf__ds_x(self, i_bf):
        center_x, center_y = self._center(i_bf)
        radius_trafo = self._radius_trafo(center_x, center_y)
        field = ((-2) *  torch.exp(1/(radius_trafo ** 2 - 1)) * (self.s_grid[0, :, :] - center_x)) / (self.len_x ** 2 * (1 - radius_trafo ** 2) ** 2)
        field[radius_trafo > 1] = 0
        return field
        
    def _eval_d_bf__ds_y(self, i_bf):
        center_x, center_y = self._center(i_bf)
        radius_trafo = self._radius_trafo(center_x, center_y)
        field = ((-2) *  torch.exp(1/(radius_trafo ** 2 - 1)) * (self.s_grid[1, :, :] - center_y)) / (self.len_y ** 2 * (1 - radius_trafo ** 2) ** 2)
        field[radius_trafo > 1] = 0
        return field

    def _center(self, i_bf):
        return torch.reshape(self.bf_grid[0, :, :], [-1])[i_bf], \
               torch.reshape(self.bf_grid[1, :, :], [-1])[i_bf]

    def _radius_trafo(self, center_x, center_y):
        return torch.sqrt((self.s_grid[0, :, :] - center_x) ** 2 / self.len_x ** 2
                          + (self.s_grid[1, :, :] - center_y) ** 2 / self.len_y ** 2)

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
        s_grid, temp = cuda_to_cpu(self.s_grid.clone(), temp)
        for i in range(numOfShFuncs_0):
            for j in range(numOfShFuncs_1):
                axs[i, j].pcolormesh(s_grid[0, :, :], s_grid[1, :, :], temp[i, j, :, :],
                                     cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
                axs[i, j].set_title("Shape Function: " + str(i * numOfShFuncs_1 + j))
                axs[i, j].set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def plot_bfs(self):
        self._plot_func(self.bfs, 'Radian Basis Functions')

    def plot_bfs_with_BC(self):
        self._plot_func(self.bfs_with_BC, 'Radian Basis Functions with BC mask')

    def plot_d_bf_with_BC__ds_x(self):
        self._plot_func(self.d_bf_with_BC__d_s_x, 'd_bf_with_BC__d_s_x')

    def plot_d_bf_with_BC__ds_y(self):
        self._plot_func(self.d_bf_with_BC__d_s_y, 'd_bf_with_BC__d_s_y')

    def plot_d_bf__ds_y(self):
        self._plot_func(self.d_bf_with_BC__d_s_y, 'd_bf_with_BC__d_s_y')