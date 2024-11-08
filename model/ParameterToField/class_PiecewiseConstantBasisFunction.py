import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu

class PiecewiseConstantBasisFunction(ParameterToField):
    def __init__(self, options):
        # center for RBF
        self.bf_grid = options["bf_grid"]

        self.max_x = options["s_grid"][1, :, :].max()
        self.min_x = options["s_grid"][1, :, :].min()
        self.max_y = options["s_grid"][0, :, :].max()
        self.min_y = options["s_grid"][0, :, :].min()

        self.len_x = (self.max_x - self.min_x) / self.bf_grid.size(dim=1)
        self.len_y = (self.max_y - self.min_y) / self.bf_grid.size(dim=2)

        self.x_grid = options["s_grid"][0, :, :]
        self.y_grid = options["s_grid"][1, :, :]

        super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])

        # shape parameter (length of the square side)


    # def _eval_bf(self, i_bf):
    #     blank_bf = torch.zeros_like(self.s_grid[0, :, :])
    #     element_index = self.find_element_index(i_bf)
    #     blank_bf[*element_index.tolist()] = 1
    #     return blank_bf

    def _eval_d_bf__ds_x(self, i_bf):
        # This is not implemented for piecewise constant basis functions
        return torch.zeros_like(self.s_grid[0, :, :])

    def _eval_d_bf__ds_y(self, i_bf):
        # This is not implemented for piecewise constant basis functions
        return torch.zeros_like(self.s_grid[1, :, :])

    def _eval_bf(self, i_bf):
        # numerical stability
        eps = 1e-6
        # center of the basis function is given by the grid
        center_x, center_y = torch.reshape(self.bf_grid[0, :, :], [-1])[i_bf], torch.reshape(self.bf_grid[1, :, :], [-1])[i_bf]
        # boundaries of the basis function
        left_x, right_x = center_x - self.len_x/2, center_x + self.len_x/2
        lower_y, upper_y = center_y - self.len_y/2, center_y + self.len_y/2

        # dealing w\ numerical error
        if torch.isclose(left_x, self.min_x):
            left_x = self.min_x
        if torch.isclose(right_x, self.max_x):
            right_x = self.max_x + 2 * eps # this counteracts the -eps in the if statement (so we catch the right edge)
        if torch.isclose(lower_y, self.min_y):
            lower_y = self.min_y
        if torch.isclose(upper_y, self.max_y):
            upper_y = self.max_y + 2 * eps # same as above
        
        tensor = torch.zeros_like(self.s_grid[0, :, :])
        tensor[(self.x_grid >= left_x) & (self.x_grid <= right_x) & (self.y_grid >= lower_y) & (self.y_grid <= upper_y)] = 1
        # # find the element index
        # index = torch.tensor([])
        # for i in range(self.s_grid.size(dim=1)):
        #     for j in range(self.s_grid.size(dim=2)):
        #         if self.s_grid[0, i, j] >= left_x-eps and self.s_grid[0, i, j] <= right_x-eps and self.s_grid[1, i, j] >= lower_y-eps and self.s_grid[1, i, j] <= upper_y-eps:
        #             if index.size() == torch.Size([0]):
        #                 index = torch.tensor([i, j]).unsqueeze(1)
        #             else:
        #                 index = torch.hstack((index, torch.tensor([i, j]).unsqueeze(1)))
        return tensor
    
    def _create_set_of_bf_with_BC(self):
        return torch.einsum("nij, ij -> nij",self.bfs, self.BC_Mask)

    def _create_set_of_d_bf_with_BC__d_s_x(self):
        d_bf__ds_x = torch.zeros_like(self.s_grid[0, :, :])
        all__d_bf__ds_x = d_bf__ds_x.repeat(self.bf_grid.size(dim=1) * self.bf_grid.size(dim=2), 1, 1)
        return torch.einsum("nij, ij -> nij", all__d_bf__ds_x, self.BC_Mask) + torch.einsum("nij, ij -> nij", self.bfs, self.BC_Mask_grad_s_x)

    def _create_set_of_d_bf_with_BC__d_s_y(self):
        d_bf__ds_y = torch.zeros_like(self.s_grid[0, :, :])
        all__d_bf__ds_y = d_bf__ds_y.repeat(self.bf_grid.size(dim=1) * self.bf_grid.size(dim=2), 1, 1)
        return torch.einsum("nij, ij -> nij", all__d_bf__ds_y, self.BC_Mask) + torch.einsum("nij, ij -> nij", self.bfs, self.BC_Mask_grad_s_y)
    
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
                                     cmap='coolwarm', shading='auto', vmin=vmin, vmax=vmax)
                axs[i, j].set_title("Shape Function: " + str(i * numOfShFuncs_1 + j))
                axs[i, j].set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def plot_bfs(self):
        self._plot_func(self.bfs, 'Piecewise Constant Basis Functions')

    def plot_bfs_with_BC(self):
        self._plot_func(self.bfs_with_BC, 'Piecewise Constant Basis Functions with BC mask')

    def plot_d_bf_with_BC__ds_x(self):
        self._plot_func(self.d_bf_with_BC__d_s_x, 'd_bf_with_BC__d_s_x')

    def plot_d_bf_with_BC__ds_y(self):
        self._plot_func(self.d_bf_with_BC__d_s_y, 'd_bf_with_BC__d_s_y')