import torch
from model.ParentClasses.class_BCMask import BCMask


class BCMask_top_bottom_zero(BCMask):
    def __init__(self, s_grid, c=1):
        # this is some constant for the mask. Don't know about this
        self.c = c

        # inherit
        super().__init__(s_grid)

    def _eval(self):
        """
        Field: f = s2 * s_1
        0-------
        0-------
        0-------
        00000000
        """
        return self.s_grid[0, :, :] * self.s_grid[1, :, :] / self.c

    def _eval_grad_s_x(self):
        # independent of x
        return self.s_grid[1, :, :] / self.c

    def _eval_grad_s_y(self):
        return self.s_grid[0, :, :] / self.c

    # TODO: define second order derivatives