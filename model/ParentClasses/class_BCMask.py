import torch

class BCMask:
    def __init__(self, options):
        # spacial grid
        self.s_grid = options["s_grid"]

        """
        Idea: I can precalculate the fields, because for the derivatives wrt to s_x and s_y I can use the derivative
        summation rule. And for the derivative wrt to the parameters x, y and theta, the BCMask is just a constant.
        """
        # precalculate value matrix
        value = self._eval()
        self.value = value

        # precalculate values of derivatives
        grad_x = self._eval_grad_s_x()
        grad_y = self._eval_grad_s_y()
        self.grad_x = grad_x
        self.grad_y = grad_y

        # usually they will be zero, except for special cases, where we define this extra :)
        if "_eval_grad_s_x2" in dir(self):
            self.grad_x2 = self._eval_grad_s_x2()
        else:
            self.grad_x2 = torch.zeros_like(self.s_grid[0, :, :])
        if "_eval_grad_s_y2" in dir(self):
            self.grad_y2 = self._eval_grad_s_y2()
        else:
            self.grad_y2 = torch.zeros_like(self.s_grid[0, :, :])
        if "_eval_grad_s_xs_y" in dir(self):
            self.grad_xy = self._eval_grad_s_xs_y()
        else:
            self.grad_xy = torch.zeros_like(self.s_grid[0, :, :])


    #%% Stuff you have to define in a child class
    def _eval(self):
        """
        This should evaluate the function on the grid
        :return: torch_funcs.tensor of dim(s_x, s_y)
        """
        pass

    def _eval_grad_s_x(self):
        """
        This should evaluate the derivative wrt to s_x of the function on the grid
        :return: torch_funcs.tensor of dim(s_x, s_y)
        """
        pass

    def _eval_grad_s_y(self):
        """
        This should evaluate the derivative wrt to s_y of the function on the grid
        :return: torch_funcs.tensor of dim(s_x, s_y)
        """
        pass

    #%% public functions
    def get_value(self):
        return self.value

    def get_grad_s_x(self):
        return self.grad_x

    def get_grad_s_y(self):
        return self.grad_y
    
    def get_grad_s_x2(self):
        return self.grad_x2
    
    def get_grad_s_y2(self):
        return self.grad_y2

    def get_grad_s_xds_y(self):
        return self.grad_xy