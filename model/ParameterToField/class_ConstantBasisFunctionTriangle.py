import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu
from utils.torch_funcs.function_LinearTriangle import get_triangular_nodes
from utils.torch_funcs.function_check_inside_triangle import point_in_triangle
from utils.torch_funcs.function_regular_mesh import regular_2D_mesh

class ConstantBasisFunctionTriangle(ParameterToField):
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

        # Intermediate store for the shape functions
        self.triangShapeFuncsNodes = None
        self.triangShapeFuncsNodes_dx = None
        self.triangShapeFuncsNodes_dy = None

        self.center_points = None

        super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])
    
    # evaluate the derivative of the basis functions with finite differences
    def eval_grad_s(self, parameter):
        field = self.eval(parameter) 
        if field.dim() == 2: # add channel dimension and batch dimension
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.dim() == 3: # a batch of fields -> add channel dimension
            field = field.unsqueeze(1)
        else:
            raise ValueError("Input tensor has wrong shape.")
        
        N = 1000
    #     # unsqueeze to add channel dimension
    #     # field_x = torch.nn.functional.interpolate(field, size=(field.size(-2)+1, field.size(-1)), mode='bilinear', align_corners=True) # .view(-1, t1.size(-2)+1, t1.size(-1)) # upsampling
    #     # dx = torch.diff(field_x * field.size(-2), dim=-2)
    #     # field_y = torch.nn.functional.interpolate(field, size=(field.size(-2), field.size(-1)+1), mode='bilinear', align_corners=True) # .view(-1, t1.size(-2), N+1)
    #     # dy = torch.diff(field_y * field.size(-1), dim=-1)
        field_x = torch.nn.functional.interpolate(field, size=(N+1, field.size(-1)), mode='bilinear', align_corners=True) # .view(-1, t1.size(-2)+1, t1.size(-1)) # upsampling
        dx = torch.diff(field_x * N, dim=-2)
        dx = torch.nn.functional.interpolate(dx, size=(field.size(-2), field.size(-1)), mode='bilinear', align_corners=True)
        field_y = torch.nn.functional.interpolate(field, size=(field.size(-2), N+1), mode='bilinear', align_corners=True) # .view(-1, t1.size(-2), N+1)
        dy = torch.diff(field_y * N, dim=-1)
        dy = torch.nn.functional.interpolate(dy, size=(field.size(-2), field.size(-1)), mode='bilinear', align_corners=True)
        
        return torch.cat((dx, dy), dim=-3).squeeze() # remove batch dimension if necessary


    # create ALL basis functions sets (BF + dx + dy)
    def _create_set_of_bf(self):
        # create the coordinates of the triangles 
        nodes, nodesMap = get_triangular_nodes(self.bf_grid)
        self.center_points = torch.mean(torch.stack(nodes)[:,:2,:], dim=2).T

        # initilize emptry shape function tensor
        triangle_shape_function = torch.zeros(len(nodes), self.s_grid.size(-2), self.s_grid.size(-1))

        # create all the individual shape functions for EACH triangle (3 per triangle)
        has_value = torch.ones(self.s_grid.size(-2), self.s_grid.size(-1), dtype=torch.bool)
        for i in range(len(nodes)): 
            triangle_shape_function[i] = point_in_triangle(self.s_grid.view(2, -1).t(), *nodes[i][:2,:].t()).reshape(self.s_grid.size(-2), self.s_grid.size(-1)).float()
            triangle_shape_function[i] = triangle_shape_function[i] * has_value
            has_value[triangle_shape_function[i] != 0] = False

        # I have overlaps, so I make multiple triangles contribute to the same point by adding them up 
        num_triangles_at_grid = triangle_shape_function.sum(0)
        triangle_shape_function = triangle_shape_function / num_triangles_at_grid

        # my bf_grid is not on the nodes now, but in the middle of the triangles
        bf_per_column = self.bf_grid.size(-2) - 1
        bf_per_row = self.bf_grid.size(-1) - 1
        bf_grid_0 = regular_2D_mesh(bf_per_column, bf_per_row, on_boundary=False)
        bf_grid_1 = bf_grid_0 + torch.einsum('i,ijk->ijk', torch.mul(torch.tensor([-0.25, 0.25]), torch.tensor([self.len_x, self.len_y])), torch.ones_like(bf_grid_0))
        bf_grid_2 = bf_grid_0 + torch.einsum('i,ijk->ijk', torch.mul(torch.tensor([0.25, -0.25]), torch.tensor([self.len_x, self.len_y])), torch.ones_like(bf_grid_0))
        bf_grid = torch.concat((bf_grid_1, bf_grid_2), dim=-2)
        # bf_grid = bf_grid + torch.tensor([0.5, 0.5]) * torch.tensor([self.len_x, self.len_y])
        self.bf_grid = bf_grid
        self.n_bfs = len(nodes)
        if self.flag_given_values == True:
            raise NotImplementedError("Given values are not implemented for triangle basis functions.")
        else:
            self.num_unknowns = self.n_bfs

        # all should sum to 1
        assert torch.allclose(triangle_shape_function.sum(0), torch.tensor(1.0))
        # assert torch.allclose(triangShapeFuncsNodes_dx.sum(0), torch.tensor(0.0))
        # assert torch.allclose(triangShapeFuncsNodes_dy.sum(0), torch.tensor(0.0))

        # store intermediate results
        self.triangShapeFuncsNodes_dx = torch.zeros_like(triangle_shape_function)
        self.triangShapeFuncsNodes_dy = torch.zeros_like(triangle_shape_function)

        return triangle_shape_function
    
    # overwrite the functions to create the set in a parallelized way
    def _create_set_of_bf_with_BC(self):
        return self.bfs * self.BC_Mask
    
    def _create_set_of_d_bf_with_BC__d_s_x(self):
        return self.triangShapeFuncsNodes_dx * self.BC_Mask + self.bfs * self.BC_Mask_grad_s_x

    def _create_set_of_d_bf_with_BC__d_s_y(self):
        return self.triangShapeFuncsNodes_dy * self.BC_Mask + self.bfs * self.BC_Mask_grad_s_y

    # plots
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

"""
from utils.torch_funcs.function_regular_mesh import regular_2D_mesh
from model.BCMasks.class_BCMask_None import BCMask_None

s_grid = regular_2D_mesh(128, 128, on_boundary=True)
bf_grid = regular_2D_mesh(3, 3, on_boundary=True)
BCMask = BCMask_None({"s_grid": s_grid})
options = {"bf_grid": bf_grid, "s_grid": s_grid, "bf_args": {}, "BC_mask": BCMask}
my_BF = ConstantBasisFunction(options)
my_BF.plot_bfs()
"""
