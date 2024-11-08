import torch
from model.ParentClasses.class_ParameterToField import ParameterToField
from utils.torch_funcs.function_LinearTriangle import get_triangular_nodes
from utils.torch_funcs.function_check_inside_triangle import point_in_triangle

class AnalyticalLinearBasisFunction(ParameterToField):
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
        self.triangShapeFuncsNodes_deriv = None

        self.nodes = None
        self.nodesMap = None

        self.Two_A = None

        self.flag_obs_nodes = False
        self.obs_element = None
        self.obs_xi = None

        super().__init__(options["s_grid"], 2*len(self.bf_grid[0, :, :].flatten()), options["bf_args"], options["BC_mask"])

        self.top_node_list = None
        self.bottom_node_list = None
        self.left_node_list = None
        self.right_node_list = None
        self._get_boundary_elements()

        self.elements_per_node_list = self._get_elements_per_node_list()

        self.neighbours_mask = None
        self._get_neighbours_mask()

    def eval_grad_s(self, parameter):
        # if I have given values, I have to place them in the parameter vector
        parameter = self.apply_know_values(parameter)

        # from 1D [vx1, vx2, ..., vy1, vy2, ...] to 2D: N x [[vx1, vy1], [vx2, vy2], ...], where N is the number of elements
        parameter_local = self._rewrite_into_local(parameter)
        deriv = torch.einsum('...ijk,kli->...ijl', parameter_local, self.triangShapeFuncsNodes_deriv) # [..., Nele, 2, 3] x [3, 2, Nele] -> [..., Nele, 2, 2]
        return deriv # element-wise

    def eval(self, x):
        raise NotImplementedError("You should not need this if you are bypassing the Neumann term.")

    def eval_at_locations(self, x):
        x_global = self.apply_know_values(x) # [..., N_bfs]
        if self.flag_obs_nodes:
            val_obs = x_global.view(-1, 2, self.bf_grid.size(dim=-2), self.bf_grid.size(dim=-1)).permute(0,1,3,2).flatten(start_dim=-2, end_dim=-1) # [..., 2, N_bfs]
        else:
            # please run _location_precompute before this
            x_local = self._rewrite_into_local(x_global)
            x_ele = x_local[...,self.obs_element, :, :]
            val_obs = torch.einsum('...ijk, ki-> ...ij', x_ele, self.obs_xi) # [..., Nobs, 2, 3] x [3, Nobs] -> [..., Nobs, 2]
            val_obs = val_obs.permute(0, 2, 1) # [..., 2, Nobs]
        return val_obs

    def _location_precompute(self, locations):
        # FIXME: This has some problem when point is in multiple elements (e.g. on edges or nodes)
        # init solution (-1 is for the case that the location is not in any triangle)
        result = torch.ones(locations.size(0), 3) * (-1)
        # find out in which triangle the location is
        for i, node in enumerate(self.nodes):
            mask = point_in_triangle(locations, node[:2, 0], node[:2, 1], node[:2, 2])
            if mask.sum() == 0:
                continue
            locs = locations[mask, :]
            node_num = i * torch.ones(locs.size(0))
            new_row = torch.concat([node_num.unsqueeze(1), locs], dim = 1)
            result[mask] = new_row
        # check if all locations are inside a triangle
        if (result == -1).sum() > 0:
            raise ValueError("Some locations are not inside any triangle.")
        
        # get the element and the local coordinates
        xy1 = torch.concat((result[:, 1:], torch.ones(result.size(0), 1)), dim=1)
        element = result[:, 0].to(torch.int64)
        xi = torch.einsum('ijk,kj -> ik', self.triangShapeFuncsNodes[:,:,element], xy1) # [3 x Nobs] # 1/2A is already in shape functions

        self.obs_element = element
        self.obs_xi = xi

    def _create_set_of_bf(self):
        # create the coordinates of the triangles 
        nodes, nodesMap = get_triangular_nodes(self.bf_grid)
        nodes = torch.stack(nodes, dim=0)

        # order them counter-clockwise (or make sure that they are)
        # if det([[x1, x2, x3], [y1, y2, y3], [1, 1, 1]])
        # < 0 then clockwise (reorder)
        # > 0 then counter-clockwise (okay)
        # = 0 then colinear (error)
        # (https://math.stackexchange.com/questions/1324179/how-to-tell-if-3-connected-points-are-connected-clockwise-or-counter-clockwise)
        coords = nodes[:, :2, :]
        ones = torch.ones(coords.size(0), 1, coords.size(2))
        coords = torch.cat((coords, ones), dim=1)
        det = torch.det(coords)
        for i in range(det.size(0)):
            if det[i] < 0:
                nodes[i] = nodes[i, :, [0, 2, 1]]
                nodesMap[i] = nodesMap[i, [0, 2, 1]]
        
        # save for later use
        self.nodes = nodes
        self.nodesMap = nodesMap

        # short definitions
        x1 = nodes[:, 0, 0]
        x2 = nodes[:, 0, 1]
        x3 = nodes[:, 0, 2]
        y1 = nodes[:, 1, 0]
        y2 = nodes[:, 1, 1]
        y3 = nodes[:, 1, 2]

        # calculate triangle area(s) 
        # 2A = (x_32 y_12 - x_12 y_32)
        # (Belytschko, Appendix 3, p.761)
        self.Two_A = (x3 - x2) * (y1 - y2) - (x1 - x2) * (y3 - y2)

        # tensor of the shape functions
        s1 = torch.stack([y2 - y3, x3 - x2, x2 * y3 - x3 * y2], dim=0)
        s2 = torch.stack([y3 - y1, x1 - x3, x3 * y1 - x1 * y3], dim=0)
        s3 = torch.stack([y1 - y2, x2 - x1, x1 * y2 - x2 * y1], dim=0)
        self.triangShapeFuncsNodes = 1/self.Two_A * torch.stack([s1, s2, s3], dim=0) # Nele x 3 x 3
        # self.triangShapeFuncs = 1/Two_A * torch.tensor([[y2 - y3, x3 - x2, x2 * y3 - x3 * y2], [y3 - y1, x1 - x3, x3 * y1 - x1 * y3], [y1 - y2, x2 - x1, x1 * y2 - x2 * y1]])

        # tensor of the shape functions derivatives
        a1 = torch.stack([y2 - y3, x3 - x2], dim=0)
        a2 = torch.stack([y3 - y1, x1 - x3], dim=0)
        a3 = torch.stack([y1 - y2, x2 - x1], dim=0)
        self.triangShapeFuncsNodes_deriv = 1/self.Two_A * torch.stack([a1, a2, a3], dim=0)
        # self.triangShapeFuncsNodes_deriv = 1/Two_A * torch.tensor([, [y3 - y1, x1 - x3], [y1 - y2, x2 - x1]])

        return None

    def _rewrite_into_local(self, x):
        # x is a 1D tensor
        x2 = x.view(-1 ,2, int(self.n_bfs/2)) # [v1x, v2x, ..., v1y, v2y, ...] -> [[v1x, v1y], [v2x, v2y], ...] -> # [..., 2, Nbfs]

        # get the local triangle coordinates
        x3 = x2[..., :, self.nodesMap] # [..., Nele, 2, 3]
        x_new = x3.permute(0, 2, 1, 3) # [..., Nele, 3, 2]
        return x_new

    def _get_boundary_elements(self):
        nodes = self.nodes[:, :2, :]
        # get the boundary elements
        # top 
        top_mask_nodes = (nodes[:, 1, :] - self.max_y) > -1e-6
        self.top_node_list = self._get_BC_node_list(top_mask_nodes)

        # bottom
        bottom_mask = (nodes[:, 1, :] - self.min_y) < 1e-6
        self.bottom_node_list = self._get_BC_node_list(bottom_mask)

        # left
        left_mask = (nodes[:, 0, :] - self.min_x) < 1e-6
        self.left_node_list = self._get_BC_node_list(left_mask)

        # right
        right_mask = (nodes[:, 0, :] - self.max_x) > -1e-6
        self.right_node_list = self._get_BC_node_list(right_mask)

    def _get_BC_node_list(self, mask_nodes):
        top_mask_elements = mask_nodes.sum(dim=1) == 2
        top_mask_nodes_of_elements = mask_nodes * top_mask_elements.repeat(3,1).T
        top_element_nodes = top_mask_nodes_of_elements * self.nodesMap
        top_element_nodes[~top_mask_nodes_of_elements] = -1 # this is so I dont count zeros later
        top_sum_nodes = torch.zeros(self.n_bfs // 2) # I have to divide by 2 because I have x and y, but only one set of nodes
        for i in range(len(top_sum_nodes)):
            top_sum_nodes[i] = torch.sum(top_element_nodes == i)
        return top_sum_nodes

    def _get_elements_per_node_list(self):
        elements_per_node_list = torch.zeros(self.n_bfs // 2)
        for i in range(self.n_bfs // 2):
            elements_per_node_list[i] = torch.sum(self.nodesMap == i)
        return elements_per_node_list

    def _get_neighbours_mask(self):
        # - creates a matrix that is of dimension N_ele x N_Neighbours
        # where always one of the elements is 1 and the other -1
        # - you can thus calculate the differences between neighbouring elements
        # - neighbourhood is defined by common edges

        NM = self.nodesMap
        nele = len(NM)
        solution = None
        for i in range(nele):
            # create a tensor with nodes of [a all nodes > idx_a]
            # of dim [nele - idx_a - 1, 6]
            a = NM[i].repeat(nele-(i+1), 1)
            b = NM[(i+1):]
            c = torch.cat([a, b], dim=1)
            # bread and butter: sort and then see how many times the elements change
            # identical to finding number of unique elements per row
            # scales O(n)
            e = torch.sort(c, axis=1)
            e = e[0]
            num_unique = 6 - ((e[:,1:] != e[:,:-1]).sum(axis=1)+1)
            # get index of neighbours and number of neighbours
            idx_nbr = 1 + i  + torch.where(num_unique.gt(1))[0]
            num_nbr = len(idx_nbr)
            # create intermediate solution, where +1 at i and -1 at idx_nbr
            helper = torch.zeros(nele, num_nbr)
            helper[i,:] = 1
            helper[idx_nbr] = torch.eye(num_nbr) * (-1)
            if solution is None:
                solution = helper
            else:
                solution = torch.cat((solution, helper), dim=1)
        self.neighbours_mask = solution.permute(1, 0)

    def _create_set_of_bf_with_BC(self):
        return None
    
    def _create_set_of_d_bf_with_BC__d_s_x(self):
        return None
    
    def _create_set_of_d_bf_with_BC__d_s_y(self):
        return None



if __name__ == "__main__":
    from utils.torch_funcs.function_regular_mesh import regular_2D_mesh
    from model.BCMasks.class_BCMask_None import BCMask_None

    s_grid = regular_2D_mesh(128, 128, on_boundary=True)
    bf_grid = regular_2D_mesh(3, 3, on_boundary=True)
    parameter = bf_grid.clone()
    parameter[0] = parameter.clone()[0].T 
    parameter[1] = parameter.clone()[1].T
    parameter = parameter.flatten()
    parameter += torch.randn_like(parameter) * 0.1
    parameter = parameter.repeat(2, 1)
    BCMask = BCMask_None({"s_grid": s_grid})
    options = {"bf_grid": bf_grid, "s_grid": s_grid, "bf_args": {}, "BC_mask": BCMask}
    my_BF = AnalyticalLinearBasisFunction(options)
    a = my_BF.eval_grad_s(parameter)
    print(my_BF.elements_per_node_list)
    theta = torch.ones(9)
    bf1 = torch.sum(my_BF.elements_per_node_list * theta * 0 * my_BF.Two_A[0]/6, dim=-1)
    bf2 = torch.sum(my_BF.elements_per_node_list * theta * -1 * my_BF.Two_A[0]/6, dim=-1)
    print(bf1)
    print(bf2)
    # torch.einsum('ij,...j->...i', my_BF.neighbours_mask, torch.rand(10, len(my_BF.nodesMap)))
    # print(a)
    # locations = torch.tensor([[0.5, 0.5], [0.75, 0.45], [0.0, 0.0], [0.9, 0.9]])
    # my_BF._location_precompute(locations)
    # b = my_BF.eval_at_locations(parameter)
    # print(b)
    # my_BF._get_boundary_elements()


