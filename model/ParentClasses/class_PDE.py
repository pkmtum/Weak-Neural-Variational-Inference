from abc import ABC

import torch

# My imports
# from utils.torch_funcs.function_check_if_scalar import check_if_scalar
from utils.Integrator.function_integrator_2D import trapzInt2D


class PDE(ABC, torch.nn.Module):

    # This is a parent class that needs to be overwritten by the user defining
    # a) lhs + args_lhs
    # b) rhs + args_rhs

    def __init__(self, args_rhs, args_lhs):
        # Call super constructor
        super().__init__()

        self.args_lhs = args_lhs
        self.args_rhs = args_rhs

        # # Dolfin functions for MaterialField, solution, and weighting function
        # self.MFs = MFs  # this is a list of MFs

    def forward(self, x, y, theta):
        """
        This function takes care of calculating ONE residual. This is all done with a 2D trapez integrator in torch_funcs,
        thus, I get the derivatives for free with autograd.
        :param x: input parameters
        :param y: solution field parameters
        :param theta: weighting function parameters
        :return: ONE residual
        """

        # See the autograd section for explanation of what happens here.
        
        return trapzInt2D(self._lhs(x, y, theta, self.args_lhs) - self._rhs(theta, self.args_rhs)) - self._neumann(theta, self.args_rhs)

    # reference solve with the torch solver
    def torch_reference_solve(self):
        pass
