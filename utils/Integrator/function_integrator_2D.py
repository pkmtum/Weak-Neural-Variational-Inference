import torch


def trapzInt2D(f_xy):
    out = torch.trapezoid(f_xy, torch.linspace(0, 1, f_xy.size(dim=-2)), dim=-2)
    return torch.trapezoid(out, torch.linspace(0, 1, f_xy.size(dim=-1)), dim=-1)
