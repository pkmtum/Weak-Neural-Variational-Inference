import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt


class RadialBasisFunction(ParameterToField):
    def __init__(self, options):
        # center for RBF
        self.bf_grid = options["bf_grid"]

        # shape parameter for RBF
        self.epsilon = options["bf_args"]["epsilon"] # (options["bf_args"]["epsilon"] - 1) **2

        super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])

        # self.BC_Mask_grad_s_x2 = options["BC_mask"].get_grad_s_x2()
        # self.BC_Mask_grad_s_y2 = options["BC_mask"].get_grad_s_y2()
        # self.BC_Mask_grad_s_xds_y = options["BC_mask"].get_grad_s_xds_y()

        # self.d_bf_with_BC__d_s_x_d_s_y = self._create_set_of_d2_bf_with_BC__d_s_xds_y()
        # self.d_bf_with_BC__d_s_x2 = self._create_set_of_d2_bf_with_BC__d_s_x2()
        # self.d_bf_with_BC__d_s_y2 = self._create_set_of_d2_bf_with_BC__d_s_y2()

    #%% Overloaded functions so I can VMAP
    # 0. order
    def _create_set_of_bf(self):
        return torch.vmap(self._eval_bf)(torch.reshape(self.bf_grid[0, :, :], [-1]), torch.reshape(self.bf_grid[1, :, :], [-1]))

    def _create_set_of_bf_with_BC(self):
        return torch.einsum("nij, ij -> nij",self.bfs, self.BC_Mask)

    # 1. order
    def _create_set_of_d_bf_with_BC__d_s_x(self):
        all__d_bf__ds_x = torch.vmap(self._eval_d_bf__ds_x)(torch.reshape(self.bf_grid[0, :, :], [-1]), torch.reshape(self.bf_grid[1, :, :], [-1]))
        return torch.einsum("nij, ij -> nij", all__d_bf__ds_x, self.BC_Mask) + torch.einsum("nij, ij -> nij", self.bfs, self.BC_Mask_grad_s_x)

    def _create_set_of_d_bf_with_BC__d_s_y(self):
        all__d_bf__ds_y = torch.vmap(self._eval_d_bf__ds_y)(torch.reshape(self.bf_grid[0, :, :], [-1]), torch.reshape(self.bf_grid[1, :, :], [-1]))
        return torch.einsum("nij, ij -> nij", all__d_bf__ds_y, self.BC_Mask) + torch.einsum("nij, ij -> nij", self.bfs, self.BC_Mask_grad_s_y)

    # 2. order
    def _create_set_of_d2_bf_with_BC__d_s_x2(self):
        all__d2_bf__ds_x2 = torch.vmap(self._eval_d2_bf__ds_x2)(torch.reshape(self.bf_grid[0, :, :], [-1]), torch.reshape(self.bf_grid[1, :, :], [-1]))
        return torch.einsum("nij, ij -> nij", all__d2_bf__ds_x2, self.BC_Mask) + torch.einsum("nij, ij -> nij", self.bfs, self.BC_Mask_grad_s_x2) \
                + torch.einsum("nij, ij -> nij", self.d_bf_with_BC__d_s_x, self.BC_Mask_grad_s_x)
    
    def _create_set_of_d2_bf_with_BC__d_s_y2(self):
        all__d2_bf__ds_y2 = torch.vmap(self._eval_d2_bf__ds_y2)(torch.reshape(self.bf_grid[0, :, :], [-1]), torch.reshape(self.bf_grid[1, :, :], [-1]))
        return torch.einsum("nij, ij -> nij", all__d2_bf__ds_y2, self.BC_Mask) + torch.einsum("nij, ij -> nij", self.bfs, self.BC_Mask_grad_s_y2) \
                + torch.einsum("nij, ij -> nij", self.d_bf_with_BC__d_s_y, self.BC_Mask_grad_s_y)
    
    def _create_set_of_d2_bf_with_BC__d_s_xds_y(self):
        all__d2_bf__ds_xds_y = torch.vmap(self._eval_d2_bf__ds_xds_y)(torch.reshape(self.bf_grid[0, :, :], [-1]), torch.reshape(self.bf_grid[1, :, :], [-1]))
        return torch.einsum("nij, ij -> nij", all__d2_bf__ds_xds_y, self.BC_Mask) + torch.einsum("nij, ij -> nij", self.bfs, self.BC_Mask_grad_s_xds_y) \
                + torch.einsum("nij, ij -> nij", self.d_bf_with_BC__d_s_x, self.BC_Mask_grad_s_y) \
                + torch.einsum("nij, ij -> nij", self.d_bf_with_BC__d_s_y, self.BC_Mask_grad_s_x)

    #%% Funcs
    # 0. order
    def _eval_bf(self, center_x, center_y):
        return torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                          + (center_y - self.s_grid[1, :, :]) ** 2))

    # 1. order
    def _eval_d_bf__ds_x(self, center_x, center_y):
        return -2 * self.epsilon * (self.s_grid[0, :, :] - center_x) * \
               torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                          + (center_y - self.s_grid[1, :, :]) ** 2))

    def _eval_d_bf__ds_y(self, center_x, center_y):
        return -2 * self.epsilon * (self.s_grid[1, :, :] - center_y) * \
               torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                          + (center_y - self.s_grid[1, :, :]) ** 2))
    
    # 2. order
    def _eval_d2_bf__ds_x2(self, center_x, center_y):
        return 2 * self.epsilon**2 * (2 * center_x ** 2 - 4 * center_x * self.s_grid[0, :, :] + 2 * self.s_grid[0, :, :] ** 2 - 1/self.epsilon) * \
                torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                    + (center_y - self.s_grid[1, :, :]) ** 2))

    def _eval_d2_bf__ds_y2(self, center_x, center_y):
        return 2 * self.epsilon**2 * (2 * center_y ** 2 - 4 * center_y * self.s_grid[1, :, :] + 2 * self.s_grid[1, :, :] ** 2 - 1/self.epsilon) * \
                torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                    + (center_y - self.s_grid[1, :, :]) ** 2))
    
    def _eval_d2_bf__ds_xds_y(self, center_x, center_y):
        return 4 * self.epsilon**2 * (center_x - self.s_grid[0, :, :]) * (center_y - self.s_grid[1, :, :]) * \
                torch.exp(-self.epsilon * ((center_x - self.s_grid[0, :, :]) ** 2
                                    + (center_y - self.s_grid[1, :, :]) ** 2))

    def _center(self, i_bf):
        return torch.reshape(self.bf_grid[0, :, :], [-1])[i_bf], \
               torch.reshape(self.bf_grid[1, :, :], [-1])[i_bf]

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
