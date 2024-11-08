import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
import matplotlib.pyplot as plt
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu
from utils.torch_funcs.function_LinearTriangle import generate_triangle_field, get_triangular_nodes
from utils.torch_funcs.function_roll_columns import roll_columns
from utils.torch_funcs.function_check_inside_triangle import point_in_triangle

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

        # Intermediate store for the shape functions
        self.triangShapeFuncsNodes = None
        self.triangShapeFuncsNodes_dx = None
        self.triangShapeFuncsNodes_dy = None

        self.map_dof_to_triangle_dx = None
        self.map_dof_to_triangle_dy = None

        super().__init__(options["s_grid"], len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])
    
    # # evaluate the derivative of the basis functions with finite differences
    # def eval_grad_s(self, parameter):
    #     field = self.eval(parameter) 
    #     if field.dim() == 2: # add channel dimension and batch dimension
    #         field = field.unsqueeze(0).unsqueeze(0)
    #     elif field.dim() == 3: # a batch of fields -> add channel dimension
    #         field = field.unsqueeze(1)
    #     else:
    #         raise ValueError("Input tensor has wrong shape.")
        
    #     N = 1000
    # #     # unsqueeze to add channel dimension
    # #     # field_x = torch.nn.functional.interpolate(field, size=(field.size(-2)+1, field.size(-1)), mode='bilinear', align_corners=True) # .view(-1, t1.size(-2)+1, t1.size(-1)) # upsampling
    # #     # dx = torch.diff(field_x * field.size(-2), dim=-2)
    # #     # field_y = torch.nn.functional.interpolate(field, size=(field.size(-2), field.size(-1)+1), mode='bilinear', align_corners=True) # .view(-1, t1.size(-2), N+1)
    # #     # dy = torch.diff(field_y * field.size(-1), dim=-1)
    #     field_x = torch.nn.functional.interpolate(field, size=(N+1, field.size(-1)), mode='bilinear', align_corners=True) # .view(-1, t1.size(-2)+1, t1.size(-1)) # upsampling
    #     dx = torch.diff(field_x * N, dim=-2)
    #     dx = torch.nn.functional.interpolate(dx, size=(field.size(-2), field.size(-1)), mode='bilinear', align_corners=True)
    #     field_y = torch.nn.functional.interpolate(field, size=(field.size(-2), N+1), mode='bilinear', align_corners=True) # .view(-1, t1.size(-2), N+1)
    #     dy = torch.diff(field_y * N, dim=-1)
    #     dy = torch.nn.functional.interpolate(dy, size=(field.size(-2), field.size(-1)), mode='bilinear', align_corners=True)
        
    #     return torch.cat((dx, dy), dim=-3).squeeze() # remove batch dimension if necessary

    # create ALL basis functions sets (BF + dx + dy)
    def _create_set_of_bf(self):
        # create the coordinates of the triangles 
        nodes, nodesMap = get_triangular_nodes(self.bf_grid)

        # initilize emptry shape function tensor
        triangle_shape_function = torch.zeros(len(nodes), 3, self.s_grid.size(-2), self.s_grid.size(-1))
        triangle_shape_function_dx = torch.zeros(len(nodes), 3, self.s_grid.size(-2), self.s_grid.size(-1))
        triangle_shape_function_dy = torch.zeros(len(nodes), 3, self.s_grid.size(-2), self.s_grid.size(-1))

        # create all the individual shape functions for EACH triangle (3 per triangle)
        for i in range(len(nodes)):
            rolled_coordinates = roll_columns(nodes[i][:2,:])
            for j in range(len(rolled_coordinates)):
                triangle_shape_function[i, j], triangle_shape_function_dx[i,j], triangle_shape_function_dy[i,j] = generate_triangle_field(rolled_coordinates[j].t(), self.s_grid)

        # initialize empty tensor for the shape functions of the nodes
        triangShapeFuncsNodes = torch.zeros(torch.max(nodesMap)+1, self.s_grid.size(-2), self.s_grid.size(-1))
        # triangShapeFuncsNodes_dx = torch.zeros(torch.max(nodesMap)+1, self.s_grid.size(-2), self.s_grid.size(-1))
        # triangShapeFuncsNodes_dy = torch.zeros(torch.max(nodesMap)+1, self.s_grid.size(-2), self.s_grid.size(-1))

        # add the shape functions of the triangles to the nodes
        for i in range(0, nodesMap.size(0)):
            for j in range(0, 3):
                # add shape functions (to nodes) so they dont overlap
                # check where there are already values
                mask = triangShapeFuncsNodes[nodesMap[i, j]].le(0.0)
                # mask_dx = triangShapeFuncsNodes_dx[nodesMap[i, j]].le(0.0)
                # mask_dy = triangShapeFuncsNodes_dy[nodesMap[i, j]].le(0.0)
                # only add values where there are no values yet
                A = triangle_shape_function[i, j].masked_fill_(~mask, 0.0)
                # A_dx = triangle_shape_function_dx[i, j].masked_fill_(~mask_dx, 0.0)
                # A_dy = triangle_shape_function_dy[i, j].masked_fill_(~mask_dy, 0.0)
                # add values
                triangShapeFuncsNodes[nodesMap[i, j]] += A

        #         # WELLCOME TO THE DANGER ZONE!
        #         # *This is not so dangerous, because I decided to calculate the derivatives with finite differences*
        #         # When an integration point is *exactly* on a edge, the derivative is not well defined.
        #         # This happens always on the diagonal of the grid, because of the symmetry.
        #         # So we want to avoid adding the derivative in this case, but instead take the average of the two adjacent triangles.
        #         triangShapeFuncsNodes_dx[nodesMap[i, j]] += A_dx
        #         triangShapeFuncsNodes_dy[nodesMap[i, j]] += A_dy

        # # all should sum to 1
        # # assert torch.allclose(triangShapeFuncsNodes.sum(0), torch.tensor(1.0))
        # # assert torch.allclose(triangShapeFuncsNodes_dx.sum(0), torch.tensor(0.0))
        # # assert torch.allclose(triangShapeFuncsNodes_dy.sum(0), torch.tensor(0.0))

        # # store intermediate results
        # self.triangShapeFuncsNodes_dx = triangShapeFuncsNodes_dx
        # self.triangShapeFuncsNodes_dy = triangShapeFuncsNodes_dy

        # initilize emptry shape function tensor
        triangle_shape_function_d = torch.zeros(len(nodes), self.s_grid.size(-2), self.s_grid.size(-1))

        # create all the individual shape functions for EACH triangle (3 per triangle)
        has_value = torch.ones(self.s_grid.size(-2), self.s_grid.size(-1), dtype=torch.bool)
        for i in range(len(nodes)): 
            triangle_shape_function_d[i] = point_in_triangle(self.s_grid.view(2, -1).t(), *nodes[i][:2,:].t()).reshape(self.s_grid.size(-2), self.s_grid.size(-1)).float()
            triangle_shape_function_d[i] = triangle_shape_function_d[i] * has_value
            has_value[triangle_shape_function_d[i] != 0] = False

        # I should not have overlaps. Check.
        assert torch.allclose(triangle_shape_function_d.sum(0), torch.tensor(1.0))

        # Now I need a matrix that maps the values of the nodes to the values of the triangles
        # Here I am using the fact that our mesh is regular and the triangles are regular
        map_dof_to_triangle_dx = torch.zeros(self.n_bfs, len(nodes))
        map_dof_to_triangle_dy = torch.zeros(self.n_bfs, len(nodes))
        for i, node in enumerate(nodes):
            if node[1,0] == node[1,1]: # right angle at 1
                map_dof_to_triangle_dx[int(node[2,1]),i] = 1.0 / self.len_x
                map_dof_to_triangle_dx[int(node[2,0]),i] = -1.0 / self.len_x
                map_dof_to_triangle_dy[int(node[2,2]),i] = 1.0 / self.len_y
                map_dof_to_triangle_dy[int(node[2,1]),i] = -1.0 / self.len_y
            else: # right angle at 2
                map_dof_to_triangle_dx[int(node[2,1]),i] = 1.0 / self.len_x
                map_dof_to_triangle_dx[int(node[2,2]),i] = -1.0 / self.len_x
                map_dof_to_triangle_dy[int(node[2,2]),i] = 1.0 / self.len_y
                map_dof_to_triangle_dy[int(node[2,0]),i] = -1.0 / self.len_y


            # # for dx
            # if node[0,0] > node[0,1]:
            #     map_dof_to_triangle_dx[int(node[2,0]),i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[int(node[2,1]),i] = -1.0 / self.len_x
            # elif node[0,0] < node[0,1]:
            #     map_dof_to_triangle_dx[int(node[2,1]),i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[int(node[2,0]),i] = -1.0 / self.len_x
            # elif node[0,0] > node[0,2]:
            #     map_dof_to_triangle_dx[int(node[2,0]),i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[int(node[2,2]),i] = -1.0 / self.len_x
            # elif node[0,0] < node[0,2]:
            #     map_dof_to_triangle_dx[int(node[2,2]),i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[int(node[2,0]),i] = -1.0 / self.len_x
            # elif node[0,1] > node[0,2]:
            #     map_dof_to_triangle_dx[int(node[2,1]),i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[int(node[2,2]),i] = -1.0 / self.len_x
            # elif node[0,1] < node[0,2]:
            #     map_dof_to_triangle_dx[int(node[2,2]),i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[int(node[2,1]),i] = -1.0 / self.len_x
            
            # # for dy
            # if node[1,0] > node[1,1]:
            #     map_dof_to_triangle_dy[int(node[2,0]),i] = 1.0 / self.len_y
            #     map_dof_to_triangle_dy[int(node[2,1]),i] = -1.0 / self.len_y
            # elif node[1,0] < node[1,1]:
            #     map_dof_to_triangle_dy[int(node[2,1]),i] = 1.0 / self.len_y
            #     map_dof_to_triangle_dy[int(node[2,0]),i] = -1.0 / self.len_y
            # elif node[1,0] > node[1,2]:
            #     map_dof_to_triangle_dy[int(node[2,0]),i] = 1.0 / self.len_y
            #     map_dof_to_triangle_dy[int(node[2,2]),i] = -1.0 / self.len_y
            # elif node[1,0] < node[1,2]:
            #     map_dof_to_triangle_dy[int(node[2,2]),i] = 1.0 / self.len_y
            #     map_dof_to_triangle_dy[int(node[2,0]),i] = -1.0 / self.len_y
            # elif node[1,1] > node[1,2]:
            #     map_dof_to_triangle_dy[int(node[2,1]),i] = 1.0 / self.len_y
            #     map_dof_to_triangle_dy[int(node[2,2]),i] = -1.0 / self.len_y
            # elif node[1,1] < node[1,2]:
            #     map_dof_to_triangle_dy[int(node[2,2]),i] = 1.0 / self.len_y
            #     map_dof_to_triangle_dy[int(node[2,1]),i] = -1.0 / self.len_y


            # if node[2] == (node[1]+1): # right angle at position 1 (?)
            #     map_dof_to_triangle_dx[node[2],i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[node[1],i] = -1.0 / self.len_x
            #     map_dof_to_triangle_dy[node[0],i] = -1.0 / self.len_y
            #     map_dof_to_triangle_dy[node[2],i] = 1.0 / self.len_y
            # elif node[0] == (node[1]-1): # right angle at position 0
            #     map_dof_to_triangle_dx[node[1],i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[node[0],i] = -1.0 / self.len_x
            #     map_dof_to_triangle_dy[node[0],i] = -1.0 / self.len_y
            #     map_dof_to_triangle_dy[node[2],i] = 1.0 / self.len_y
            # elif node[1] == (node[2]+1): # right angle at position 2 (?)
            #     map_dof_to_triangle_dx[node[2],i] = 1.0 / self.len_x
            #     map_dof_to_triangle_dx[node[1],i] = -1.0 / self.len_x
            #     map_dof_to_triangle_dy[node[0],i] = -1.0 / self.len_y
            #     map_dof_to_triangle_dy[node[2],i] = 1.0 / self.len_y
            # else:
            #     raise ValueError("This is not a regular mesh?")
        
        self.map_dof_to_triangle_dx = map_dof_to_triangle_dx
        self.map_dof_to_triangle_dy = map_dof_to_triangle_dy

        # derivatives are:
        # p_i * m_ij * f_jkl, where 
        # p_i is the parameter vector,
        # m_ij is the mapping matrix from the nodes to the triangles
        # f_jkl is the shape function of the field.
        # index: i for nodes, j for triangles, k and l for the s_grid
        # So I can precalculate m_ij * f_jkl = f_ikl, which is then my new shape function
        self.triangShapeFuncsNodes_dx = torch.einsum('ij,jkl->ikl', map_dof_to_triangle_dx, triangle_shape_function_d)
        self.triangShapeFuncsNodes_dy = torch.einsum('ij,jkl->ikl', map_dof_to_triangle_dy, triangle_shape_function_d)

        return triangShapeFuncsNodes
    
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