import torch
from model.ParentClasses.class_displacementPDE import DisplacementPDE


class Hook(DisplacementPDE):
    def __init__(self, options):
        XToField = options["XToField"]
        YToField = options["YToField"]
        ThetaToField = options["ThetaToField"]
        if "XGTToField" in options:
            XGTToField = options["XGTToField"]
        else:
            XGTToField = XToField
        args_rhs = options["args_rhs"]
        super().__init__(XToField, YToField, ThetaToField, XGTToField, args_rhs)

    def _firstPK(self, y, x):
        """
        Formulas:
        sigma = 2 \mu \epsilon + \lambda I tr(\epsilon)
        Note that for small deformation, \sigma approx P holds, see
        (https://fenicsproject.org/pub/tutorial/html/._ftut1008.html)
        :return: Calculates stresses for the first Piola-Kirchhoff stress tensor P for Hook material
        """
        # calculate Material Fields given x --> log(E), \nu
        MFs = self.XToField.eval(x)

        # # create solution field u from y --> [2 x 2 x dim_s1 x dim_s2]
        # u = self.YToField.eval(y)

        # log(E), \nu --> E, \nu
        E = torch.exp(MFs[..., 0, :, :])
        nu = MFs[..., 1, :, :]

        # calculate Lamees constants out of those
        mu = E / (2*(1+nu))
        lmbda = E * nu / ((1+nu) * (1-2*nu))

        # release MFs from memory
        del MFs, E, nu

        # get F --> [2 x 2 x dim_s1 x dim_s2]
        # this also transformes y to u internally
        F = self._F(y)

        # eps = 0.5 ( F.T + F) - I --> [2 x 2 x dim_s1 x dim_s2]
        eps = 0.5 * (torch.einsum('...ijkl->...jikl', F) + F) - self.I_grid

        # release F from memory
        del F

        # # get F[i,j,k,l] --> F[k,l,i,j] to calc other stuff
        # F_help = torch.einsum('ijkl->klij', F)
        #
        # # calc J = det(F) --> [dim_s1 x dim_s2]
        # J = torch.linalg.det(F_help)
        #
        # # calc F^{-T} --> [2 x 2 x dim_s1 x dim_s2]
        # F_inv_T = torch.einsum('klij->jikl', torch.linalg.inv(F_help))

        # get tr(epsilon) [dim_s1 x dim_s2]
        tr_eps = torch.einsum('...iijk->...jk', eps)

        # summand 1: 2 \mu \epsilon --> [2 x 2 x dim_s1 x dim_s2]
        summand_1 = 2 * torch.einsum('...kl,...ijkl->...ijkl', mu, eps)

        # release eps and mu from memory
        del eps, mu

        # summand 2: factor 1:  \lambda tr(\epsilon) --> [dim_s1 x dim_s2]
        factor_1 = torch.einsum('...ij,...ij->...ij', lmbda, tr_eps)

        # release tr_eps and lmbda and from memory
        del tr_eps, lmbda

        # summand 2: ( \lambda tr(\epsilon) ) dyadic I --> [2 x 2 x dim_s1 x dim_s2]
        summand_2 = torch.einsum('...ij,...kl->...ijkl', self.I, factor_1)

        # release factor_1 from memory
        del factor_1

        # Cauchy stress --> [2 x 2 x dim_s1 x dim_s2]
        sigma = summand_1 + summand_2

        # # 1st P.-K. -->  [2 x 2 x dim_s1 x dim_s2]
        # # P = J * sigma * F^(-T);
        # P_help = torch.einsum('kl,ijkl->ijkl', J, sigma)
        # return torch.einsum('ijkl,ijkl->ijkl', P_help, F_inv_T)
        return sigma

    # def psi(self, MFs, F):
    #     # Right Cauchy-Green tensor
    #     C = ufl.variable(F.T * F)
    #
    #     # Invariants of deformation tensors
    #     I_1 = ufl.variable(ufl.tr(C))
    #     I_2 = ufl.variable((I_1**2 - ufl.tr(C*C)/2))
    #
    #     # formula for hyperelastic potential (psi)
    #     return 0.5 * MFs[0] * (I_1 - 3) + 0.5 * MFs[1] * (I_2 - 3)
