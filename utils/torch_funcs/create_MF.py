import torch


def make_inclusion_MF_torch(s_grid, val_0, val_inclusion, middle_point, radius):
    """
    Creates a constant material field with a circular inclusion
    :param s_grid: spacial grid [2, dim_s1, dim_s2]
    :param val_0: value of MF
    :param val_inclusion: value of inclusion
    :param middle_point: of inclusion [s1, s2]
    :param radius: of inclusion
    :return: torch_funcs tensor MF; dim: [dim_s1, dim_s2]
    """
    # empty material field tensor
    result = torch.zeros_like(s_grid[0, :, :])

    # size of grid
    dim_s1 = result.size(0)
    dim_s2 = result.size(1)

    # circular inclusion
    def circle_inclusion(x):
        return torch.sqrt(pow((x[0] - middle_point[0]), 2) + pow((x[1] - middle_point[1]), 2)) <= radius

    # loop trough grid
    for i in range(dim_s1):
        for j in range(dim_s2):

            # get coordinates
            s = s_grid[:, i, j]

            # check if we are in the circular inclusion
            if circle_inclusion(s):
                result[i, j] = val_inclusion
            else:
                result[i, j] = val_0

    return result


def make_constant_MF_torch(s_grid, val):
    """
    Creates constant material field like s_grid
    :param s_grid: spacial grid [2, dim_s1, dim_s2]
    :param val: value of MF
    :return: torch_funcs tensor MF; dim: [dim_s1, dim_s2]
    """

    # empty material field tensor
    return torch.ones_like(s_grid[0, :, :]) * val

