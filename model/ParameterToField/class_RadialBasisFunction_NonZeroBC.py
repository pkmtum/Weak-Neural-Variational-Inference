import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt


class RadialBasisFunction_NonZeroBC(ParameterToField):
    def __init__(self, s_grid, bf_grid, bf_args, BC_Mask_0, BC_Mask_val):
        # center for RBF
        self.bf_grid = bf_grid

        # shape parameter for RBF
        self.epsilon = bf_args["epsilon"]

        # mask with dirichlet values
        self.BC_Mask_val = BC_Mask_val.get_value()

        # dirichlet boundary condition mask (value) derivative wrt to spacial coordinates
        self.BC_Mask_val_grad_s_x = BC_Mask_val.get_grad_s_x()
        self.BC_Mask_val_grad_s_y = BC_Mask_val.get_grad_s_y()

        super().__init__(s_grid, len(self.bf_grid[0, :, :].flatten()), bf_args, BC_Mask_0)

    #%% Helper functions
    def _center(self, i_bf):
        return torch.reshape(self.bf_grid[0, :, :], [-1])[i_bf], \
               torch.reshape(self.bf_grid[1, :, :], [-1])[i_bf]

    # %% Key definitions
    def _eval_bf(self, i_bf):
        center_x, center_y = self._center(i_bf)
        return torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                          + (center_y - self.s_grid[1, :, :]) ** 2))

    def _eval_d_bf__ds_x(self, i_bf):
        center_x, center_y = self._center(i_bf)
        return -2 * self.epsilon * (self.s_grid[0, :, :] - center_x) * \
               torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                          + (center_y - self.s_grid[1, :, :]) ** 2))

    def _eval_d_bf__ds_y(self, i_bf):
        center_x, center_y = self._center(i_bf)
        return -2 * self.epsilon * (self.s_grid[1, :, :] - center_y) * \
               torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                          + (center_y - self.s_grid[1, :, :]) ** 2))

    #%% Overwritten functions to cover second mask
    def eval(self, parameter):
        return self.BC_Mask_val + torch.einsum('...i,ijk->...jk', parameter, self.bfs_with_BC)

    def eval_grad_s(self, parameter):
        dx = self.BC_Mask_val_grad_s_x + torch.einsum('...i,ijk->...jk', parameter, self.d_bf_with_BC__d_s_x)
        dy = self.BC_Mask_val_grad_s_y + torch.einsum('...i,ijk->...jk', parameter, self.d_bf_with_BC__d_s_y)
        return torch.stack((dx, dy), dim=-3)

    #%% Plots
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
        for i in range(numOfShFuncs_0):
            for j in range(numOfShFuncs_1):
                axs[i, j].pcolormesh(self.s_grid[0, :, :], self.s_grid[1, :, :], temp[i, j, :, :],
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
