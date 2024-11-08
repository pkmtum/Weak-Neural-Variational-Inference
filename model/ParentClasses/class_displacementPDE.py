import torch

from model.ParentClasses.class_PDE import PDE
# from utils.fenics_funcs.standard_fenics_setup_displacements import fenics_setup
from utils.function_Newton_solver import Newton_solver, least_square_solver, gradient_descent_solver
from utils.Integrator.function_integrator_1D import trapzInt1D


class DisplacementPDE(PDE):
    def __init__(self, XToField, YToField, ThetaToField, XGTToField, args_rhs):

        # Constructors for parameters x, y, w --> Fields
        """
        Note: I assume here that all components of the fields are constructed in the same way, i.e.
        dim(x_1) == dim(x_2); dim(y_1) == dim(y_2), dim(w_1) == dim(w_2);
        which, in general, does not have to be the case.
        """
        self.XToField = XToField
        self.YToField = YToField
        self.ThetaToField = ThetaToField
        self.XGTToField = XGTToField
        self.XBufferToField = None

        # dimensions of the spacial grid (is same for all fields!)
        self.s_grid = XToField.s_grid
        self.dim_s1 = XToField.s_grid.size(1)
        self.dim_s2 = XToField.s_grid.size(2)

        # Identity matrix and Identity matrix on [2 x 2 x dim_s1 x dim_s2] grid
        self.I = torch.eye(2)
        self.I_grid = self.I.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.dim_s1, self.dim_s2)

        # This is my FEniCSX structure
        # self.mesh, self.V, self.Vc, x, self.bc, u_ref, u_obs, u_sol, a_x, self.u, self.v = fenics_setup()

        # This is just for the reference solve
        self.x_true = None
        self.x_other = None

        # Dicts for additional arguments on rhs and lhs
        args_lhs = dict()

        super().__init__(args_rhs, args_lhs)

    # %% Please define this in a child class
    def _firstPK(self, y, x):
        """
        Torch: Overwrite this using the hyperelastic material law you want.
        :param y: parameters for the y field
        :param x: parameters for the x field
        :return: 1st Piola-Kirchhoff stress tensor [2 x 2 x dim_s1 x dim_s2]
        """
        pass

    def psi(self, MFs, F):
        """
        Used solve the reference
        :param MFs: ufl material fields
        :param F: ufl deformation gradient
        :return: strain energy density function for FEniCS
        """
        pass

    # %% Torch functions for the SGA / SVI (These are the functions you do not need to touch)

    def _lhs(self, x, y, w, args_lhs):
        # calculate derivatives jacobian for the weighting function
        # d_w__d_s == [2 x 2 x dim_s1 x dim_s2]
        d_w__d_s = self.ThetaToField.eval_grad_s_WF(w)

        # weak form lhs -> P : grad(w) == [dim_s1 x dim_s2]
        # NEW: P : grad(w) == [num_w x dim_s1 x dim_s2]
        return torch.einsum('...ijkl,...hijkl->...hkl', self._firstPK(y, x), d_w__d_s)


    def _rhs(self, w, args_rhs):
        if "f_load" in args_rhs.keys():
            # calculate weighting functions w_s1 and w_s2 and stack
            # w == [2 x dim_s1 x dim_s2]
            w = self.ThetaToField.eval_WF(w)

            # weak form rhs -> f * w == [dim_s1 x dim_s2]
            # f == [2], w == [2 x dim_s1 x dim_s2]
            # NEW: f * w == [num_w x dim_s1 x dim_s2]
            # f == [2], w == [num_w x 2 x dim_s1 x dim_s2]
            return torch.einsum('i,...ijk->...jk', args_rhs["f_load"], w)
        else:
            return torch.tensor(0.0)

    def _neumann(self, w, args_rhs):
        # weight function dim: N x 2 x dim_s1 x dim_s2
        w = self.ThetaToField.eval_WF(w)

        # calculate neumann BCs
        # dim dw: N x 2 x dim_s1 or N x 2 x dim_s2
        # dim neumann_...: 2 
        neumann = torch.zeros(w.size(dim=0))
        if "neumann_left" in args_rhs.keys():
            dw = w[:, :, 0, :]
            integrant = torch.einsum('j,...jk->...k', args_rhs["neumann_left"], dw)
            neumann += trapzInt1D(integrant)
        if "neumann_right" in args_rhs.keys():
            dw = w[:, :, -1, :]
            integrant = torch.einsum('j,...jk->...k', args_rhs["neumann_right"], dw)
            neumann += trapzInt1D(integrant)
        if "neumann_top" in args_rhs.keys():
            dw = w[:, :, :, -1]
            integrant = torch.einsum('j,...jk->...k', args_rhs["neumann_top"], dw)
            neumann += trapzInt1D(integrant)
        if "neumann_bottom" in args_rhs.keys():
            dw = w[:, :, :, 0]
            integrant = torch.einsum('j,...jk->...k', args_rhs["neumann_bottom"], dw)
            neumann += trapzInt1D(integrant)
        return neumann

    @staticmethod
    def _I_1(C):
        """
        :param C: F.T F, dim = [2 x 2 x dim_s1 x dim_s2]
        :return: trace(C), dim = [dim_s1 x dim_s2]
        """
        return torch.einsum('iikl->kl', C)

    def _F(self, y):
        """
        :param y: parameters for the output (displacement) field
        :return: Deformation gradient F = I + du/dX
        """
        # Split y into the parameters for y_s1 and y_s2 and calc the derivatives
        # d y_x d s --> [2 y 2 x dim_s1 x dim_s2]
        d_u__d_s = self.YToField.eval_grad_s(y)

        # deformation gradient F --> [2 x 2 x dim_s1 x dim_s2]
        return self.I_grid + d_u__d_s

    @staticmethod
    def _C(F):
        """
        right Cauchy-Green tensor C
        :param F: Deformation gradient
        :return: F.T F
        """
        # C = F.T F --> [2 x 2 x dim_s1 x dim_s2]
        return torch.einsum('jikl,ijkl->ijkl', F, F)

    def _epsilon(self, y):
        """
        Small strain tensor \epsilon = 0.5 (u_{i,j} + u_{j,i})
        :param y: parameters of u
        :return: \epsilon
        """
        # Split y into the parameters for y_s1 and y_s2 and calc the derivatives
        # d y_x d s --> [2 x 2 x dim_s1 x dim_s2]
        d_u__d_s = self.YToField.eval_grad_s(y)

        # eps --> [2 x 2 x dim_s1 x dim_s2]
        return 0.5 * (d_u__d_s + d_u__d_s.T)

    # %% Solve by Newton solver in torch
    def eqn_system_for_Newton_solver(self, y):
        # x is also done with RBFs (so I don't get a discretization error by approximating MFs with the RBFs)
        x = self.x_true

        theta = torch.eye(self.ThetaToField.len_parameter_list)
        r = self.forward(x, y, theta)

        return r

    def eqn_system_for_Newton_solver_2(self, y):
        # x is also done with RBFs (so I don't get a discretization error by approximating MFs with the RBFs)
        x = self.x_other

        theta = torch.eye(self.ThetaToField.len_parameter_list)
        r = self.forward(x, y, theta)

        return r

    def torch_reference_solve(self, true_solve=True):
        tol = 1e-1

        # initilize zeros starting values
        y_start = torch.zeros(self.YToField.len_parameter_list, requires_grad=True)

        if true_solve:
            self.XBufferToField = self.XToField
            self.XToField = self.XGTToField
            SoE = self.eqn_system_for_Newton_solver
        else:
            SoE = self.eqn_system_for_Newton_solver_2
        
        # solve the system with newton solver
        if self.ThetaToField.len_parameter_list == self.YToField.len_parameter_list:
            y, n = Newton_solver(SoE, x=y_start, eps=tol, step_size=1)
        else:  
            y, n = least_square_solver(SoE, x=y_start, eps=tol, step_size=1)

        if true_solve:
            self.XToField = self.XBufferToField
            
        # y, n = gradient_descent_solver(SoE, x=y_start, eps=tol, step_size=1)
        return y

    # %% FEniCSX functions to solve the reference solution
    # def reference_solve(self, MFs_fenics):
    #     """
    #     only used to create reference solution.
    #     NOTE THAT THIS HAS NO IMPLEMENTATION TO WORK WITH PYRO!!!!
    #     """
    #
    #     # Define the problem
    #     y_sol = dfx.fem.Function(self.V)
    #     problem = dfx.fem.petsc.NonlinearProblem(
    #         self.dfx_residual(MFs_fenics, y_sol, self.v, self.args_lhs, self.args_rhs),
    #         y_sol,
    #         bcs=self.bc)
    #
    #     # solver settings
    #     dfx.log.set_log_level(dfx.log.LogLevel.INFO)
    #     solver = dfx.nls.petsc.NewtonSolver(self.mesh.comm, problem)
    #     solver.atol = 1e-6
    #     solver.rtol = 1e-6
    #     solver.convergence_criterion = "residual"
    #
    #     # solve the FE problem
    #     solver.solve(y_sol)
    #
    #     # solution to torch_funcs tensor
    #     y_sol_torch = probe_FEniCS_function(self.s_grid, y_sol)
    #
    #     # Return the solution
    #     return y_sol_torch
    #
    # def dfx_residual(self, MF, y, w, args_lhs, args_rhs):
    #     d = len(y)
    #
    #     # Identity tensor
    #     I = ufl.variable(ufl.Identity(d))
    #     # Deformation gradient
    #     F = ufl.variable(I + ufl.grad(y))
    #     psi = self.psi(MF, F)
    #     a = ufl.inner(ufl.grad(w), ufl.diff(psi, F)) * ufl.dx
    #     # f_load = dfx.fem.Constant(self.mesh, Vec(np.array(args_rhs["f_load"])))
    #     f_load = dfx.fem.Constant(self.mesh, ScalarType(np.array(args_rhs["f_load"])))
    #     L = ufl.dot(f_load, w) * ufl.dx
    #     return a - L
    #
    # @staticmethod
    # def epsilon(y):
    #     return ufl.sym(ufl.grad(y))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
