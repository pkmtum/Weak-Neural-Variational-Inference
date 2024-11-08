import torch


class ConstantField:
    def __init__(self, options):
        # spacial grid ([dim, nele_x, nele_y])
        self.s_grid = options["s_grid"]

        # number of basis functions (needed because this will be asked for at some other point in the code)
        self.n_bfs = 0
        self.num_unknowns = self.n_bfs

        # optional arguments for bfs (containing e.g. center of BFs and shape parameters)
        self.bf_args = options["bf_args"]

        # Constant
        self.c = self.bf_args["value"]

        # Constant field
        self.c_field = torch.ones_like(self.s_grid[0]) * self.c

        # And their derivatives (dc / ds = 0)
        self.d_c_field__d_s_x = torch.zeros_like(self.c_field)
        self.d_c_field__d_s_y = torch.zeros_like(self.c_field)

    # %% public functions to call
    def eval(self, *args):
        """
        :param args: Just a dummy so I don't get an error if a parameter is passed
        :return: spacial grid with a constant at each point 
        """
        if args[0] == None:
            return self.c_field
        else:
            return torch.ones(args[0].shape[:-1] + self.c_field.shape) * self.c_field

    def eval_grad_s(self, *args):
        """
        :param args: Just a dummy so I don't get an error if a parameter is passed
        :return: spacial grid with the derivative of the constant field (i.e. 0) at each point
        """
        return torch.stack((self.d_c_field__d_s_x, self.d_c_field__d_s_y), dim=0)
