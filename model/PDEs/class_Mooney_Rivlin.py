import torch
import ufl
from model.ParentClasses.class_displacementPDE import DisplacementPDE


class MooneyRivlin(DisplacementPDE):
    def __init__(self, XToField, YToField, ThetaToField, f_load):
        super().__init__(XToField, YToField, ThetaToField, f_load)

    def _firstPK(self, y, x):
        """
        Formulas:
        S = mu_1 I - mu_2 (I_1 I - C)
        :return: Calculates stresses for the first Piola-Kirchhoff stress tensor P for the incompressible
        Mooney-Rivlin material
        """
        # calculate Material Fields given x
        MFs = self.XToField.eval(x)

        # get C --> [2 x 2 x dim_s1 x dim_s2]
        F = self._F(y)
        C = self._C(F)

        # factor --> [2 x 2 x dim_s1 x dim_s2]
        # I == [2 x 2], C == [2 x 2 x dim_s1, dim_s2], I_1 = [dim_s1, dim_s2]
        factor = torch.einsum('ij,kl->ijkl', self.I, self._I_1(C)) - C

        # summand_1 --> [2 x 2 x dim_s1 x dim_s2]
        # I == [2 x 2], MFs[0] == [dim_s1 x dim_s2]
        summand_1 = torch.einsum('ij,kl->ijkl', self.I, MFs[0, :, :])

        # summand_2 --> [2 x 2 x dim_s1 x dim_s2]
        # MFs[0] == [dim_s1 x dim_s2], factor == [2 x 2 x dim_s1 x dim_s2]
        summand_2 = torch.einsum('kl,ijkl->ijkl', MFs[1, :, :], factor)

        # 2nd P.-K. --> [2 x 2 x dim_s1 x dim_s2]
        # S = 2 dPsi / dC
        S = summand_1 - summand_2

        # 1st P.-K. -->  [2 x 2 x dim_s1 x dim_s2]
        # P = F * S; where S = 2 dPsi / dC
        return torch.einsum('ijkl,jnkl->inkl', F, S)

    def psi(self, MFs, F):
        # Right Cauchy-Green tensor
        C = ufl.variable(F.T * F)

        # Invariants of deformation tensors
        I_1 = ufl.variable(ufl.tr(C))
        I_2 = ufl.variable((I_1**2 - ufl.tr(C*C)/2))

        # formula for hyperelastic potential (psi)
        return 0.5 * MFs[0] * (I_1 - 3) + 0.5 * MFs[1] * (I_2 - 3)
