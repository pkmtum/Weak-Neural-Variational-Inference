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
        Field: f = s2 * (1-s2) / C
        00000000
        --------
        --------
        00000000
        """
        return (1 - self.s_grid[1, :, :]) * self.s_grid[1, :, :] / self.c

    def _eval_grad_s_x(self):
        # independent of x
        return torch.zeros_like(self.s_grid[0, :, :])

    def _eval_grad_s_y(self):
        return (1 - 2 * self.s_grid[1, :, :]) / self.c

    # TODO: define second order derivatives