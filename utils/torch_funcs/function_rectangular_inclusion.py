import torch

def function_rectangular_inclusion(s_grid, options):
    """Function to create a circular inclusion in the integration grid
    !!! CAREFUL: THIS IS NOT APPLIED TO THE BASIS FUNCTION GRID; ONLY TO THE INTEGRATION GRID !!!
    Input:
    dict{Inclusion_1: dict{center_x: float, x1: float, y1: float, x2: float, y2: float, value: float},
        Inclusion_2: dict{center_x: float, x1: float, y1: float, x2: float, y2: float, value: float},
        ...
        Inclusion_n: dict{center_x: float, x1: float, y1: float, x2: float, y2: float, value: float}}
        
    """
    field = torch.zeros_like(s_grid[0, :, :])
    for key, val in options.items():
        x1_mask = torch.where(s_grid[0,:,:] > options[key]["x1"], True, False)
        x2_mask = torch.where(s_grid[0,:,:] < options[key]["x2"], True, False)
        y1_mask = torch.where(s_grid[1,:,:] > options[key]["y1"], True, False)
        y2_mask = torch.where(s_grid[1,:,:] < options[key]["y2"], True, False)
        mask = x1_mask * x2_mask * y1_mask * y2_mask
        field += torch.zeros_like(s_grid[0, :, :]).masked_fill_(mask, options[key]["value"])
    return field

