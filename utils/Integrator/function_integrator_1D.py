import torch


def trapzInt1D(f_x):
    return torch.trapezoid(f_x, torch.linspace(0, 1, f_x.size(dim=-1)), dim=-1)
    
