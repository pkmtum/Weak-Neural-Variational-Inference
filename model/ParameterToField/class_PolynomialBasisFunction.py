import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu

class PolynomialBasisFunction(ParameterToField):
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
        trafo_x = trafo_x / self.len_x
        trafo_y = trafo_y / self.len_y

        # initialize
        val_x = torch.zeros_like(trafo_x)
        val_y = torch.zeros_like(trafo_y)
        
        # calculate either the value or the derivative
        if tag == "val":
            # calculate the value (function is piecewise defined for positive and negative values)
            val_x[trafo_x>0] = self._my_polynomial_pos(trafo_x[trafo_x>0])
            val_x[trafo_x<=0] = self._my_polynomial_neg(trafo_x[trafo_x<=0])
            val_y[trafo_y>0] = self._my_polynomial_pos(trafo_y[trafo_y>0])
            val_y[trafo_y<=0] = self._my_polynomial_neg(trafo_y[trafo_y<=0])
            # grid_point_val = (1 - torch.abs(trafo_x)/self.len_x) * (1 - torch.abs(trafo_y)/self.len_y)
            grid_point_val = val_x * val_y

        elif tag == "d_x":
            # function in y remains:
            val_y[trafo_y>0] = self._my_polynomial_pos(trafo_y[trafo_y>0])
            val_y[trafo_y<=0] = self._my_polynomial_neg(trafo_y[trafo_y<=0])
            # derivative of function in x
            val_x[trafo_x>0] = self._my_polynomial_pos_dx(trafo_x[trafo_x>0])
            val_x[trafo_x<=0] = self._my_polynomial_neg_dx(trafo_x[trafo_x<=0])
            grid_point_val = val_x * val_y
        elif tag == "d_y":
            # function in x remains:
            val_x[trafo_x>0] = self._my_polynomial_pos(trafo_x[trafo_x>0])
            val_x[trafo_x<=0] = self._my_polynomial_neg(trafo_x[trafo_x<=0])
            # derivative of function in y
            val_y[trafo_y>0] = self._my_polynomial_pos_dx(trafo_y[trafo_y>0])
            val_y[trafo_y<=0] = self._my_polynomial_neg_dx(trafo_y[trafo_y<=0])
            grid_point_val = val_x * val_y
        field[mask_1D] = grid_point_val.flatten()
        return field

    def _my_polynomial_pos(self, x):
        return 4 * x**5 - 10 * x**4 + 10 * x**3 - 5 * x**2 + 1
    
    def _my_polynomial_neg(self, x):
        return - 4 * x**5 - 10 * x**4 - 10 * x**3 - 5 * x**2 + 1
    
    def _my_polynomial_pos_dx(self, x):
        return 20 * x**4 - 40 * x**3 + 30 * x**2 - 10 * x

    def _my_polynomial_neg_dx(self, x):
        return - 20 * x**4 - 40 * x**3 - 30 * x**2 - 10 * x

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
                axs[i, j].pcolormesh(temp[i, j, :, :],
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