import torch
from model.ParentClasses.class_displacementPDE import DisplacementPDE


class NeoHook(DisplacementPDE):
    def __init__(self, options):
        XToField = options["XToField"]
        YToField = options["YToField"]
        ThetaToField = options["ThetaToField"]
        args_rhs = options["args_rhs"]
        super().__init__(XToField, YToField, ThetaToField, args_rhs)

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

        # (B-I)) = F * F.T - I --> [2 x 2 x dim_s1 x dim_s2]
        B_minus_I = torch.einsum('...aikl, ...bikl -> ...abkl',F, F) - torch.eye(2).unsqueeze(-1).unsqueeze(-1)
        # det(F)
        F_prep = F.swapaxes(-3, -1).swapaxes(-4, -2) # [..., d1, d2, dim_s1, dim_s2] -> [..., dim_s1, dim_s2, d1, d2]
        J = torch.linalg.det(F_prep) # [..., dim_s1, dim_s2]
        J_inv = 1 / J
        J_minus_1 = J - 1

        # summand 1: mu * J^{-1} * (B-I) --> [2 x 2 x dim_s1 x dim_s2]
        summand_1 = torch.einsum('...ij, ...ij, ...klij -> ...klij', mu, J_inv, B_minus_I)

        # summand 2: ( \lambda tr(\epsilon) ) dyadic I --> [2 x 2 x dim_s1 x dim_s2]
        summand_2 = torch.einsum('...ij, ...ij, kl -> ...klij', lmbda, J_minus_1, torch.eye(2))
        # Cauchy stress --> [2 x 2 x dim_s1 x dim_s2]
        sigma = summand_1 + summand_2

        # # 1st P.-K. -->  [2 x 2 x dim_s1 x dim_s2]
        # # P = J * sigma * F^(-T);
        # P_help = torch.einsum('kl,ijkl->ijkl', J, sigma)
        # return torch.einsum('ijkl,ijkl->ijkl', P_help, F_inv_T)
        return sigma

