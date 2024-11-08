import torch

from model.ParentClasses.class_BCMask import BCMask


class BCMask_None(BCMask):
    """
    Mask is 1 at all nodes in the mesh.
    """
    def __init__(self, options):
        # inherit
        super().__init__(options)

    def _eval(self):
        return torch.ones_like(self.s_grid[0, :, :])

    def _eval_grad_s_x(self):
        return torch.zeros_like(self.s_grid[0, :, :])

    def _eval_grad_s_y(self):
        return torch.zeros_like(self.s_grid[0, :, :])
