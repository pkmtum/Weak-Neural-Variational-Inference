import torch
from model.ParentClasses.class_BCMask import BCMask


class BCMask_left_zero(BCMask):
    def __init__(self, options):
        # inherit
        super().__init__(options)

    def _eval(self):
        """
        Field: f = s1
        0-------
        0-------
        0-------
        0-------
        """
        return self.s_grid[0, :, :]

    def _eval_grad_s_x(self):
        # df/ds1 = val
        return torch.ones_like(self.s_grid[-1, :, :])

    def _eval_grad_s_y(self):
        # independent of y
        return torch.zeros_like(self.s_grid[-2, :, :]) 