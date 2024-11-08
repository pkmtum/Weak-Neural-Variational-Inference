import torch
from model.ParentClasses.class_BCMask import BCMask


class BCMask_top_zero(BCMask):
    def __init__(self, s_grid):
        # inherit
        super().__init__(s_grid)

    def _eval(self):
        """
        Field: f = 1 - s2
        00000000
        --------
        --------
        --------
        """
        return 1 - self.s_grid[1, :, :]

    def _eval_grad_s_x(self):
        # independent of x
        return torch.zeros_like(self.s_grid[0, :, :])

    def _eval_grad_s_y(self):
        # df/ds2 = val
        return - torch.ones_like(self.s_grid[0, :, :])
