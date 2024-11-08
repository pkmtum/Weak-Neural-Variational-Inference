import torch
# from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt
from utils.torch_funcs.function_difference_matrix import create_difference_matrix
from utils.torch_funcs.function_regular_mesh import regular_2D_mesh
from utils.plots.plot_multiple_fields import plot_multiple_fields

# class QuadraticBasisFunction(ParameterToField):
#     def __init__(self, options):
#         # center for RBF
#         self.bf_grid = options["bf_grid"]

#         # grid bounds
#         self.max_x = options["s_grid"][-2, :, :].max()
#         self.min_x = options["s_grid"][-2, :, :].min()
#         self.max_y = options["s_grid"][-1, :, :].max()
#         self.min_y = options["s_grid"][-1, :, :].min()

#         # distance between grid points
#         self.len_x = (self.max_x - self.min_x) / (self.bf_grid.size(dim=-2) - 1)
#         self.len_y = (self.max_y - self.min_y) / (self.bf_grid.size(dim=-1) - 1)

#         # Intermediate store for the shape functions
#         self.triangShapeFuncsNodes = None
#         self.triangShapeFuncsNodes_dx = None
#         self.triangShapeFuncsNodes_dy = None

#         super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])
    
#     def N1(self, x_1, x_2, x_3, x_4, x_5, x_6, y_1, y_2, y_3, y_4, y_5, y_6):
#         pass


    

# Example usage
x = torch.tensor([0, 1, 0, 0.5, 0.5, 0])
y = torch.tensor([0, 0, 1, 0, 0.5, 0.5])
dx = create_difference_matrix(x, x)
dy = create_difference_matrix(y, y)

n = 20
s_grid = regular_2D_mesh(20, 20, on_boundary=True)
tril_indices = torch.tril_indices(row=n, col=n)

# def N1(s_grid, x_i, y_i ,x_ij, y_ij):
#     denominator1 = (x_ij[1,2] * y_ij[0,2] - x_ij[0,2] * y_ij[1,2])
#     denominator2 = (x_ij[3,5] * y_ij[0,5] - x_ij[0,5] * y_ij[3,5])
#     factor1 = x_ij[1,2] * (s_grid[1] - y_i[2]) - y_ij[1,2] * (s_grid[0] - x_i[2])
#     factor2 = x_ij[3,5] * (s_grid[1] - y_i[5]) - y_ij[3,5] * (s_grid[0] - x_i[5])
#     return (factor1 * factor2) / (denominator1 * denominator2)

def N_general(s_grid, coefs, x, y, dx, dy):
    factor1 = dx[coefs["a"][0]-1, coefs["a"][1]-1] * (s_grid[1] - y[coefs["b"]-1]) - dy[coefs["a"][0]-1, coefs["a"][1]-1] * (s_grid[0] - x[coefs["b"]-1])
    factor2 = dx[coefs["c"][0]-1, coefs["c"][1]-1] * (s_grid[1] - y[coefs["d"]-1]) - dy[coefs["c"][0]-1, coefs["c"][1]-1] * (s_grid[0] - x[coefs["d"]-1])
    denomintor1 = dx[coefs["e"][0]-1, coefs["e"][1]-1] * dy[coefs["f"][0]-1, coefs["f"][1]-1] - dx[coefs["f"][0]-1, coefs["f"][1]-1] * dy[coefs["e"][0]-1, coefs["e"][1]-1]
    denomintor2 = dx[coefs["g"][0]-1, coefs["g"][1]-1] * dy[coefs["h"][0]-1, coefs["h"][1]-1] - dx[coefs["h"][0]-1, coefs["h"][1]-1] * dy[coefs["g"][0]-1, coefs["g"][1]-1]
    return (factor1 * factor2) / (denomintor1 * denomintor2)

coefs1 = {"a": [2,3],
         "b": 3,
         "c":[4,6],
         "d": 6,
         "e": [2,3],
         "f": [1,3],
         "g": [4,6],
         "h": [1,6]      
         }

coefs2= {"a": [3,1],
         "b": 1,
         "c":[5,4],
         "d": 4,
         "e": [3,1],
         "f": [2,1],
         "g": [5,4],
         "h": [2,4]      
         }

coefs3 = {"a": [2,1],
         "b": 1,
         "c":[5,6],
         "d": 6,
         "e": [2,1],
         "f": [3,1],
         "g": [5,6],
         "h": [3,6]      
         }

coefs4 = {"a": [3,1],
         "b": 1,
         "c":[2,3],
         "d": 3,
         "e": [3,1],
         "f": [4,1],
         "g": [2,3],
         "h": [4,3]      
         }

coefs5 = {"a": [3,1],
         "b": 1,
         "c":[2,1],
         "d": 1,
         "e": [3,1],
         "f": [5,1],
         "g": [2,1],
         "h": [5,1]      
         }

coefs6 = {"a": [2,1],
         "b": 1,
         "c":[2,3],
         "d": 3,
         "e": [2,1],
         "f": [6,1],
         "g": [2,3],
         "h": [6,3]      
         }

mask = torch.zeros_like(s_grid[0])
mask[tril_indices[0], tril_indices[1]] = 1
mask = torch.flip(mask.T, dims=[1])

N1 = N_general(s_grid, coefs1, x, y, dx, dy) * mask
N2 = N_general(s_grid, coefs2, x, y, dx, dy) * mask
N3 = N_general(s_grid, coefs3, x, y, dx, dy) * mask
N4 = N_general(s_grid, coefs4, x, y, dx, dy) * mask
N5 = N_general(s_grid, coefs5, x, y, dx, dy) * mask
N6 = N_general(s_grid, coefs6, x, y, dx, dy) * mask

Ni = torch.stack([N1, N2, N3, N4, N5, N6], dim=0)

plot_multiple_fields(Ni, s_grid, titles="N")