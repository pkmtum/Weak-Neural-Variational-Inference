import ufl
import torch

# My imports
from model.ParentClasses.class_PDE import PDE
from utils.fenics_funcs.standard_fenics_setup import fenics_setup


class Poisson(PDE):
    def __init__(self, C_eigenValues, C_eigenVectors):
        self.C_eigenValues = C_eigenValues
        self.C_eigenVectors = C_eigenVectors

        # Standard fenics_funcs setup
        self.mesh, self.V, self.Vc, _, self.bc, self.constrained_dofs, self.unconstrained_dofs, \
        self.bc_values, self.f_load, _, _, _, _, self.y, self.w = fenics_setup()

        # Dicts for additional arguments on rhs and lhs
        args_lhs = dict()
        args_rhs = dict()

        # f_load is for my specific problem
        args_rhs["f_load"] = self.f_load
        super().__init__(args_lhs, args_rhs)

    def residual(self, MF, y, w, args_lhs, args_rhs):
        return - torch.einsum('ijk,ijk->jk',
                              torch.einsum('ijk,jk->ijk', self.shapeFunc.cdWeighFunc(w), MF),
                              self.shapeFunc.cdWeighFunc(y))  - self.rhs * self.shapeFunc.cWeighFunc(w))

    def x_to_MF(self, x):
        x = x.squeeze()
        help_1 = torch.multiply(x, self.C_eigenValues)
        z_l = torch.einsum('l,kl->k', help_1, self.C_eigenVectors)
        MF = torch.exp(z_l)
        return MF

    # This is a help method to set C_eigenValues and vector AFTER initialization
    def set_EVal_and_EVec(self, C_eigenValues, C_eigenVectors):
        self.C_eigenValues = C_eigenValues
        self.C_eigenVectors = C_eigenVectors
