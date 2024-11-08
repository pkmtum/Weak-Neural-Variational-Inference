import torch
from model.ParentClasses.class_BCMask import BCMask


class BCMask_bottom_zero(BCMask):
    def __init__(self, options):
        # inherit
        super().__init__(options)

    def _eval(self):
        """
        Field: f = s2
        --------
        --------
        --------
        00000000
        """
        return self.s_grid[-1, :, :]

    def _eval_grad_s_x(self):
        # independent of x
        return torch.zeros_like(self.s_grid[-2, :, :])

    def _eval_grad_s_y(self):
        # df/ds2 = val
        return torch.ones_like(self.s_grid[-1, :, :])
