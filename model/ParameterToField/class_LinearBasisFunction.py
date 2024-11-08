import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu

class LinearBasisFunction(ParameterToField):
    def __init__(self, options):
        # center for RBF
        self.bf_grid = options["bf_grid"]

        # grid bounds
        self.max_x = options["s_grid"][-2, :, :].max()
        self.min_x = options["s_grid"][-2, :, :].min()
        self.max_y = options["s_grid"][-1, :, :].max()
        self.min_y = options["s_grid"][-1, :, :].min()

        # distance between grid points
        self.len_x = (self.max_x - self.min_x) / (self.bf_grid.size(dim=-2) - 1)
        self.len_y = (self.max_y - self.min_y) / (self.bf_grid.size(dim=-1) - 1)

        super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])


    def _eval_bf(self, i_bf):
        field = self.val(i_bf, "val")
        return field

    def _eval_d_bf__ds_x(self, i_bf):
        # This is not implemented for piecewise constant basis functions
        return self.val(i_bf, "d_x")

    def _eval_d_bf__ds_y(self, i_bf):
        # This is not implemented for piecewise constant basis functions
        return self.val(i_bf, "d_y")
    
    def val(self, i_bf, tag):
        # numerical stability
        eps = 1e-5
        # center of the basis function is given by the grid
        center_x, center_y = torch.reshape(self.bf_grid[-2, :, :], [-1])[i_bf], torch.reshape(self.bf_grid[-1, :, :], [-1])[i_bf]
        # boundaries of the basis function
        left_x, right_x = center_x - self.len_x, center_x + self.len_x
        lower_y, upper_y = center_y - self.len_y, center_y + self.len_y

        # get the elements of the grid that are in the basis function
        field = torch.zeros_like(self.s_grid[0, :, :])
        mask_1D = torch.zeros_like(self.s_grid[0, :, :], dtype=torch.bool)
        mask_x = (self.s_grid[-2,:,:] >= left_x-eps) & (self.s_grid[-2,:,:] <= right_x+eps)
        mask_y = (self.s_grid[-1,:,:] >= lower_y-eps) & (self.s_grid[-1,:,:] <= upper_y+eps)
        mask_1D[mask_x & mask_y] = True
        mask_2D = torch.stack((mask_1D, mask_1D))
        grid_points = self.s_grid[mask_2D]

        # turn the 1D tensor grid points back into a [2 x dim_x x dim_y] tensor
        len_mask_x = torch.max(torch.sum(mask_x, dim=-2))
        len_mask_y = torch.max(torch.sum(mask_y, dim=-1))
        grid_points = grid_points.reshape(2, len_mask_x, len_mask_y)

        # deal with numerical errors
        trafo_x = grid_points[-2,:,:] - center_x
        trafo_y = grid_points[-1,:,:] - center_y
        trafo_x[torch.isclose(trafo_x, torch.tensor([0.0]), atol=eps)] = 0.0
        trafo_y[torch.isclose(trafo_y, torch.tensor([0.0]), atol=eps)] = 0.0

        # calculate either the value or the derivative
        if tag == "val":
            grid_point_val = (1 - torch.abs(trafo_x)/self.len_x) * (1 - torch.abs(trafo_y)/self.len_y)
        elif tag == "d_x":
            # the sign function is sensible due to the jump in the derivatives
            sign = - torch.sign(trafo_x)
            if torch.allclose(trafo_x[0,:], torch.tensor(0.0), atol=eps):
                sign[0,:] = -1.0
            if torch.allclose(trafo_x[-1,:], torch.tensor(0.0), atol=eps):
                sign[-1,:] = 1.0
            # when rescaling the trafo_y, you can easily get out of the domain and get uggly artifacts
            factor = (1 - torch.abs(trafo_y)/self.len_y)
            factor[torch.isclose(factor, torch.tensor([0.0]), atol=eps)] = 0.0
            # the actual derivative
            grid_point_val = sign * factor
        elif tag == "d_y":
            # same as above
            sign = -torch.sign(trafo_y)
            if torch.allclose(trafo_y[:,0], torch.tensor(0.0), atol=eps):
                sign[:,0] = -1.0
            if torch.allclose(trafo_y[:,-1], torch.tensor(0.0), atol=eps):
                sign[:,-1] = 1.0
            factor = (1 - torch.abs(trafo_x)/self.len_x)
            factor[torch.isclose(factor, torch.tensor([0.0]), atol=eps)] = 0.0
            grid_point_val = sign * factor
        field[mask_1D] = grid_point_val.flatten()
        return field

    def _plot_func(self, fields, title=None):
        numOfShFuncs_0 = self.bf_grid.size(dim=-2)
        numOfShFuncs_1 = self.bf_grid.size(dim=-1)
        gridSize_0 = self.s_grid.size(dim=-2)
        gridSize_1 = self.s_grid.size(dim=-1)
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
                                     cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
                axs[i, j].set_title("Shape Function: " + str(i * numOfShFuncs_1 + j))
                axs[i, j].set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def plot_bfs(self):
        self._plot_func(self.bfs, 'Linear Basis Functions')

    def plot_bfs_with_BC(self):
        self._plot_func(self.bfs_with_BC, 'Linear Basis Functions with BC mask')

    def plot_d_bf_with_BC__ds_x(self):
        self._plot_func(self.d_bf_with_BC__d_s_x, 'd_bf_with_BC__d_s_x')

    def plot_d_bf_with_BC__ds_y(self):
        self._plot_func(self.d_bf_with_BC__d_s_y, 'd_bf_with_BC__d_s_y')