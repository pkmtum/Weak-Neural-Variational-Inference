import torch


def mesh_vector(n: int, on_boundary: bool):
    """
    creates a torch_funcs vector with coordinates of the mesh.
    if on_boundary:
    *----*----*----*
    else:
    --*---*---*---*--
    :param n: number of nodes in mesh in one direction
    :param on_boundary: are there nodes on the boundary?
    :return: torch_funcs vector of dim(n)
    """
    if n == 1:
        return torch.tensor(0.5)

    # make empty tensors for coordinates of grid
    result = torch.empty(n)

    # get grid spacing
    if on_boundary:
        delta = 1 / (n - 1)
    else:
        delta = 1 / n

    # calculate coordinates of grid
    for i in range(n):
        if i == 0:
            if on_boundary:
                result[i] = 0
            else:
                result[i] = delta / 2
        else:
            result[i] = result[i - 1] + delta

    return result


def regular_2D_mesh(n_x, n_y, on_boundary=False, scale_x=1, scale_y=1):
    """
    Creates a stacked torch_funcs meshgrid
    :param n_x: Number of mesh points in x
    :param n_y: Number of mesh points in y
    :param on_boundary: if node is on boundaries
    :return: torch_funcs tensor of size (2, n_x, n_y)
    """
    # create mesh vectors
    x = mesh_vector(n_x, on_boundary) * scale_x
    y = mesh_vector(n_y, on_boundary) * scale_y

    # create grids
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    # stacks the grids
    return torch.stack((grid_x, grid_y), dim=0)
