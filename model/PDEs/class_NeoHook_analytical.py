import torch

class NeoHookAnalytical():
    def __init__(self, options):
        self.nu = options["nu"]
        self.YToField = options["YToField"]
        self.ThetaToField = options["ThetaToField"]
        self.args_rhs = options["args_rhs"]

    def _stress(self, x, y):
        """
        Formulas:
        sigma = 2 \mu \epsilon + \lambda I tr(\epsilon)
        Note that for small deformation, \sigma approx P holds, see
        (https://fenicsproject.org/pub/tutorial/html/._ftut1008.html)
        :return: Calculates stresses for the first Piola-Kirchhoff stress tensor P for Hook material
        """
        # calculate Material Fields given x --> log(E), \nu

        # log(E), \nu --> E, \nu
        E = torch.exp(x) # [... x Nele]
        nu = self.nu # [1]

        # calculate Lamees constants out of those
        mu = E / (2*(1+nu)) # [... x Nele]
        lmbda = E * nu / ((1+nu) * (1-2*nu)) # [... x Nele]

        # get the element wise derivatives of the displacmemnt field
        du = self.YToField.eval_grad_s(y) # [... x Nele x 2 x 2 ]

        # F = I + du
        F = torch.eye(2) + du # and prey to god that this is broadcasted correctly, thus [... x Nele x 2 x 2 ]

        # (B-I)) = F * F.T - I --> [... x Nele x 2 x 2 ]
        B_minus_I = torch.einsum('...kai, ...kbi -> ...kab',F, F) - torch.eye(2) # broadcastable (?)
        # det(F)
        J = torch.linalg.det(F) # [..., Nele]
        J_inv = 1 / J
        J_minus_1 = J - 1

        # summand 1: mu * J^{-1} * (B-I) : [... x Nele] * [... x Nele] * [... x Nele x 2 x 2 ] --> [... x Nele x 2 x 2]
        summand_1 = torch.einsum('...i, ...i, ...ikl -> ...ikl', mu, J_inv, B_minus_I)

        # summand 2: ( \lambda (J-1) ) dyadic I -->  [... x Nele] * [..., Nele] dyadic [2 x 2] --> [... x Nele x 2 x 2]
        summand_2 = torch.einsum('...i, ...i, kl -> ...ikl', lmbda, J_minus_1, torch.eye(2))
        # Cauchy stress -->  [... x Nele x 2 x 2] +  [... x Nele x 2 x 2] --> [... x Nele x 2 x 2]
        sigma = summand_1 + summand_2

        # # 1st P.-K. -->  [2 x 2 x dim_s1 x dim_s2]
        # # P = J * sigma * F^(-T);
        # P_help = torch.einsum('kl,ijkl->ijkl', J, sigma)
        # return torch.einsum('ijkl,ijkl->ijkl', P_help, F_inv_T)
        return sigma


    def _neumann(self, theta, args_rhs):
        # theta is values at nodes !
        # NOTE: This is only for my case of CONSTANT neumann bcs. in general, you have to actually redo this.

        # calculate neumann BCs
        # dim dw: N x 2 x dim_s1 or N x 2 x dim_s2
        # dim neumann_...: 2 
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        theta = self.ThetaToField.apply_know_values(theta)
        theta_middle = theta.size(dim=1) // 2
        theta_x = theta[:, :theta_middle]
        theta_y = theta[:, theta_middle:]
        neumann = torch.zeros(theta.size(dim=0))
        if "neumann_left" in args_rhs.keys():
            neumann += torch.sum(self.ThetaToField.left_node_list * theta_x * args_rhs["neumann_left"][0] * self.ThetaToField.len_x * 0.5, dim=-1)
            neumann += torch.sum(self.ThetaToField.left_node_list * theta_y * args_rhs["neumann_left"][1] * self.ThetaToField.len_x * 0.5, dim=-1)
        if "neumann_right" in args_rhs.keys():
            neumann += torch.sum(self.ThetaToField.right_node_list * theta_x * args_rhs["neumann_right"][0] * self.ThetaToField.len_x * 0.5, dim=-1)
            neumann += torch.sum(self.ThetaToField.right_node_list * theta_y * args_rhs["neumann_right"][1] * self.ThetaToField.len_x * 0.5, dim=-1)
        if "neumann_bottom" in args_rhs.keys():
            neumann += torch.sum(self.ThetaToField.bottom_node_list * theta_x * args_rhs["neumann_bottom"][0] * self.ThetaToField.len_y * 0.5, dim=-1)
            neumann += torch.sum(self.ThetaToField.bottom_node_list * theta_y * args_rhs["neumann_bottom"][1] * self.ThetaToField.len_y * 0.5, dim=-1)
        if "neumann_top" in args_rhs.keys():
            neumann += torch.sum(self.ThetaToField.top_node_list * theta_x * args_rhs["neumann_top"][0] * self.ThetaToField.len_y * 0.5, dim=-1)
            neumann += torch.sum(self.ThetaToField.top_node_list * theta_y * args_rhs["neumann_top"][1] * self.ThetaToField.len_y * 0.5, dim=-1)
        return neumann

    def _body_force(self, theta, args_rhs):
        # NOTE: I assume a constant body force! I further assume that all elements have the same area.
        # FIXME: This is not correct yet.
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        theta = self.ThetaToField.apply_know_values(theta)
        theta_middle = theta.size(dim=1) // 2
        theta_x = theta[:, :theta_middle]
        theta_y = theta[:, theta_middle:]
        body_force = torch.zeros(theta.size(dim=0))
        body_force += torch.sum(self.ThetaToField.elements_per_node_list * theta_x * args_rhs["f"][0] * self.ThetaToField.Two_A[0]/6, dim=-1)
        body_force += torch.sum(self.ThetaToField.elements_per_node_list * theta_y * args_rhs["f"][1] * self.ThetaToField.Two_A[0]/6, dim=-1)
        return body_force

    def _lhs(self, x, y, theta):
        sigma = self._stress(x, y)
        dw = self.ThetaToField.eval_grad_s(theta)
        if dw.dim() == 3:
            dw = dw.unsqueeze(0)
        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(0)
        integrant_i = torch.einsum('hikj,...ijk->...hi', sigma, dw) # [... x Nele x 2 x 2] x [... x Nele x 2 x 2] -> [... x Nele]
        integral_i = torch.einsum('i, ...i->...i', self.ThetaToField.Two_A * 0.5, integrant_i) # [... x Nele]
        integral = torch.sum(integral_i, dim=-1)
        return integral
    
    def _rhs(self, x, y, theta, args_rhs):
        neumann = self._neumann(theta, args_rhs)
        body_force = self._body_force(theta, args_rhs)
        return neumann + body_force

    def forward(self, x, y, theta):
        lhs = self._lhs(x, y, theta)
        rhs = self._rhs(x, y, theta, self.args_rhs)
        if lhs.dim() == 2:
            rhs = rhs.unsqueeze(1)
        return lhs - rhs


# if __name__ == "__main__":
#     from utils.torch_funcs.function_regular_mesh import regular_2D_mesh
#     from model.BCMasks.class_BCMask_None import BCMask_None
#     from model.ParameterToField.class_AnalyticalLinearBasisFunctionTriangle import AnalyticalLinearBasisFunction
#     s_grid = regular_2D_mesh(128, 128, on_boundary=True)
#     bf_grid = regular_2D_mesh(3, 3, on_boundary=True)
#     parameter = bf_grid.clone()
#     parameter[0] = 0.0
#     parameter[1] = parameter.clone()[1].T
#     parameter = parameter.flatten()
#     BCMask = BCMask_None({"s_grid": s_grid})
#     options = {"bf_grid": bf_grid, "s_grid": s_grid, "bf_args": {}, "BC_mask": BCMask}
#     my_BF = AnalyticalLinearBasisFunction(options)

#     theta = torch.eye(len(parameter))[:10]

#     x = torch.ones(5, len(my_BF.nodesMap))
#     y = parameter.repeat(5, 1)

#     options = {"nu": 0.45, "YToField": my_BF, "ThetaToField": my_BF, "args_rhs": {"neumann_top": [0.0, 1.], "neumann_right": [1., 0.0]}}
#     my_Hook = Hook(options)
#     a = my_Hook.forward(x, y, theta)
#     print(a)

