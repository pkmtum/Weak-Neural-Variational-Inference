from model.ParentClasses.class_BCMask import BCMask


class BCMask_allZero(BCMask):
    def __init__(self, s_grid, c=1):
        # this is some constant for the mask. Don't know about this
        self.c = c

        # inherit
        super().__init__(s_grid)

    def _eval(self):
        return (1 - self.s_grid[0, :, :]) * self.s_grid[0, :, :] * (1 - self.s_grid[1, :, :]) * \
               self.s_grid[1, :, :] / self.c

    def _eval_grad_s_x(self):
        return -self.s_grid[0, :, :] * (1 - self.s_grid[1, :, :]) * self.s_grid[1, :, :] / self.c + \
               (1 - self.s_grid[0, :, :]) * (1 - self.s_grid[1, :, :]) * self.s_grid[1, :, :] / self.c

    def _eval_grad_s_y(self):
        return -(1 - self.s_grid[0, :, :]) * self.s_grid[0, :, :] * self.s_grid[1, :, :] / self.c + \
               (1 - self.s_grid[0, :, :]) * self.s_grid[0, :, :] * (1 - self.s_grid[1, :, :]) / self.c
