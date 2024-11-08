import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu

class FullField(ParameterToField):
    def __init__(self, options):
        # Note that bf_grid is fake in the scenario
        self.bf_grid = options["bf_grid"]

        self.max_x = options["s_grid"][1, :, :].max()
        self.min_x = options["s_grid"][1, :, :].min()
        self.max_y = options["s_grid"][0, :, :].max()
        self.min_y = options["s_grid"][0, :, :].min()

        self.x_grid = options["s_grid"][0, :, :]
        self.y_grid = options["s_grid"][1, :, :]

        super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])

        # shape parameter (length of the square side)

    def eval(self, parameter):
        # since I have the parameter at each point, I just need to respape into the s_grid shape
        if parameter.dim() == 1:
            return parameter.reshape(self.s_grid.size(dim=1), self.s_grid.size(dim=2)) * self.BC_Mask
        else:
            return parameter.reshape(parameter.size(dim=0), self.s_grid.size(dim=1), self.s_grid.size(dim=2)) * self.BC_Mask

    def eval_grad_s(self, parameter):
        raise NotImplementedError("I should never need this.")

    def eval_WF(self, parameter):
        raise NotImplementedError("I should never need this.")

    def eval_grad_s_WF(self, parameter):
        raise NotImplementedError("I should never need this.")

    # def _eval_bf(self, i_bf):
    #     blank_bf = torch.zeros_like(self.s_grid[0, :, :])
    #     element_index = self.find_element_index(i_bf)
    #     blank_bf[*element_index.tolist()] = 1
    #     return blank_bf

    def _eval_d_bf__ds_x(self, i_bf):
        # This is not implemented for piecewise constant basis functions
        return None

    def _eval_d_bf__ds_y(self, i_bf):
        # This is not implemented for piecewise constant basis functions
        return None

    def _eval_bf(self, i_bf):
        return None
    
    def _create_set_of_bf(self):
        return None
    
    def _create_set_of_bf_with_BC(self):
        return None

    def _create_set_of_d_bf_with_BC__d_s_x(self):
        return None
    
    def _create_set_of_d_bf_with_BC__d_s_y(self):
        return None
    
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