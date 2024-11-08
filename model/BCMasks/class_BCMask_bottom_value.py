import torch
from model.ParentClasses.class_BCMask import BCMask


class BCMask_bottom_value(BCMask):
    def __init__(self, s_grid, val_bottom):
        # value at the upper bound
        self.val = val_bottom

        # inherit
        super().__init__(s_grid)

    def _eval(self):
        """
        Field: f = val * (1 - s2)
        -------- (this is acctually 0 at the top, but no one cares)
        --------
        --------
        valvalval
        """
        return self.val * (1 - self.s_grid[1, :, :])

    def _eval_grad_s_x(self):
        # independent of x
        return torch.zeros_like(self.s_grid[0, :, :])

    def _eval_grad_s_y(self):
        # df/ds2 = val
        return - self.val * torch.ones_like(self.s_grid[0, :, :])
