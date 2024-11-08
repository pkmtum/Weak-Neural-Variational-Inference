import torch
from model.ParentClasses.class_BCMask import BCMask
from utils.Integrator.function_integrator_2D import trapzInt2D


class ParameterToField:
    def __init__(self, s_grid, n_bfs, bf_args, BC_Mask: BCMask):
        # spacial grid ([dim, nele_x, nele_y])
        self.s_grid = s_grid

        # number of basis functions
        self.n_bfs = n_bfs

        # optional arguments for bfs (containing e.g. center of BFs and shape parameters)
        self.bf_args = bf_args

        # check if my parameters are in part given a-priori
        if "flag_given_values" in bf_args:
            self.flag_given_values = bf_args["flag_given_values"]
        else:
            self.flag_given_values = False
        if self.flag_given_values == True:
            self.given_value = bf_args["given_values"]
            self.mask_given_values = bf_args["mask_given_values"].flatten()
            self.num_given_values = torch.sum(self.mask_given_values)
            self.num_unknowns = self.n_bfs - self.num_given_values # number of unknowns
        else:
            self.num_unknowns = self.n_bfs  # number of unknowns

        # flag to norm (or don't) the field for weight function
        if "flag_norm" in bf_args:
            self.flag_norm = bf_args["flag_norm"]
        else:
            self.flag_norm = True

        # I can precalculate EVERYTHING!!1!111
        # Basis function
        self.bfs = self._create_set_of_bf()

        # dirichlet boundary condition mask value (field)
        self.BC_Mask = BC_Mask.get_value()

        # dirichlet boundary condition mask derivative wrt to spacial coordinates
        self.BC_Mask_grad_s_x = BC_Mask.get_grad_s_x()
        self.BC_Mask_grad_s_y = BC_Mask.get_grad_s_y()

        # Basis function with dirichlet boundary conditions
        self.bfs_with_BC = self._create_set_of_bf_with_BC()

        # spacial derivatives of basis function with dirichlet boundary conditions
        self.d_bf_with_BC__d_s_x = self._create_set_of_d_bf_with_BC__d_s_x()
        self.d_bf_with_BC__d_s_y = self._create_set_of_d_bf_with_BC__d_s_y()

    # %% Stuff you have to define in a child class
    def _eval_bf(self, i_bf):
        """
        This evaluates ONE BasisFunction on the whole domain (without BC mask).
        :param i: used to identify the correct parameters for the basis function from the options
        :return: 2D BF
        """
        pass

    def _eval_d_bf__ds_x(self, i_bf):
        """
        Derivative of ONE bf wrt to s_x.
        :return:
        """
        pass

    def _eval_d_bf__ds_y(self, i_bf):
        """
        Derivative of ONE bf wrt to s_y.
        :return:
        """
        pass

    # %% public functions to call
    def eval(self, parameter):
        """
        Evaluates ALL basis functions (i.e. generates a grid representation from the basis functions and a given
        parameter vector)
        :param parameter: 1D torch_funcs vector of length number of basis functions
        :return: spacial grid with evaluated field
        """
        # if I have given values, I have to place them in the parameter vector
        parameter = self.apply_know_values(parameter)

        # actual evaluation
        return torch.einsum('...i,ijk->...jk', parameter, self.bfs_with_BC)

    def eval_grad_s(self, parameter):
        """
        Evaluates the spacial derivatives of ALL basis functions (i.e. generates a grid representation from the
        basis functions and a given parameter vector)
        :param parameter: 1D torch_funcs vector of length number of basis functions
        :return: spacial grid with evaluated field with the derivatives (d_BF__d_x, d_BF__d_y)
        """
        # if I have given values, I have to place them in the parameter vector
        parameter = self.apply_know_values(parameter)
        # actual evaluation
        dx = torch.einsum('...i,ijk->...jk', parameter, self.d_bf_with_BC__d_s_x)
        dy = torch.einsum('...i,ijk->...jk', parameter, self.d_bf_with_BC__d_s_y)
        return torch.stack((dx, dy), dim=-3)
    
    def eval_WF(self, parameter):
        # if I have given values, I have to place them in the parameter vector
        # if self.flag_given_values == True:
        #     helper = torch.zeros_like(self.mask_given_values, dtype=parameter.dtype)
        #     helper[self.mask_given_values] = self.given_value
        #     if parameter.dim() == 2:
        #         helper2 = helper.repeat(parameter.shape[0], 1)
        #     else:
        #         helper2 = helper
        #     helper2[:, ~self.mask_given_values] = parameter
        #     parameter = helper2
        # actual evaluation
        WF = self.eval(parameter)  # [n x dim_s1 x dim_s2]
        if self.flag_norm == False:
            return WF
        norm_WF = trapzInt2D(torch.abs(WF)) # [n]
        one_by_norm_WF = torch.divide(1.0, norm_WF)
        one_by_norm_WF = one_by_norm_WF.nan_to_num_()
        return torch.einsum('i,ijk->ijk', one_by_norm_WF, WF)

    def eval_grad_s_WF(self, parameter):
        if self.flag_norm == False:
            return self.eval_grad_s(parameter)
        
        # norm factor
        WF = self.eval(parameter)  # [n x dim_s1 x dim_s2]
        norm_WF = trapzInt2D(torch.abs(WF)) # [n]
        one_by_norm_WF = torch.divide(1.0, norm_WF)
        one_by_norm_WF = one_by_norm_WF.nan_to_num_()
        # actual evaluation
        return torch.einsum('i,ijkl->ijkl', one_by_norm_WF, self.eval_grad_s(parameter))

    def apply_know_values(self, parameter):
        if self.flag_given_values == True:
            helper = torch.zeros_like(self.mask_given_values, dtype=parameter.dtype)
            helper[self.mask_given_values] = self.given_value
            if parameter.dim() == 2:
                helper2 = helper.repeat(parameter.shape[0], 1)
            else:
                helper2 = helper
            if helper2.dim() == 1:
                helper2 = helper2.unsqueeze(0)
            helper2[:, ~self.mask_given_values] = parameter
            parameter = helper2
        return parameter

    # %% private functions
    def _create_set_of_bf(self):
        """
        ONLY FOR PLOTTING REALLY
        evaluate ALL the BFs  (dim(s_x, s_y)) and stacks them into a torch_funcs.tensor (dim(n_bfs, s_x, s_y))
        :return: torch_funcs.tensor (dim(n_bfs, s_x, s_y)) with ALL precalculated BFs
        """
        list_bfs = []
        for i_bf in range(self.n_bfs):
            list_bfs.append(self._eval_bf(i_bf))
        return torch.stack(list_bfs)

    def _create_set_of_bf_with_BC(self):
        """
        evaluate ALL the BFs with BCs (dim(s_x, s_y)) and stacks them into a torch_funcs.tensor (dim(n_bfs, s_x, s_y))
        :return: torch_funcs.tensor (dim(n_bfs, s_x, s_y)) with ALL precalculated BFs
        """
        list_bfs = []
        for i_bf in range(self.n_bfs):
            list_bfs.append(self._eval_bf_with_BC(i_bf))
        return torch.stack(list_bfs)

    def _create_set_of_d_bf_with_BC__d_s_x(self):
        """
        evaluate the derivative of ALL the BFs with BCs (dim(s_x, s_y)) wrt to s_x and stacks them into a
        torch_funcs.tensor (dim(n_bfs, s_x, s_y))
        :return: torch_funcs.tensor (dim(n_bfs, s_x, s_y)) with ALL precalculated derivatives of BFs wrt s_x
        """
        list_grad_bfs = []
        for i_bf in range(self.n_bfs):
            list_grad_bfs.append(self._eval_d_bf_with_BC__d_s_x(i_bf))
        return torch.stack(list_grad_bfs)

    def _create_set_of_d_bf_with_BC__d_s_y(self):
        """
        evaluate the derivative of ALL the BFs with BCs (dim(s_x, s_y)) wrt to s_y and stacks them into a
        torch_funcs.tensor (dim(n_bfs, s_x, s_y))
        :return: torch_funcs.tensor (dim(n_bfs, s_x, s_y)) with ALL precalculated derivatives of BFs wrt s_y
        """
        list_grad_bfs = []
        for i_bf in range(self.n_bfs):
            list_grad_bfs.append(self._eval_d_bf_with_BC__d_s_y(i_bf))
        return torch.stack(list_grad_bfs)

    def _eval_bf_with_BC(self, i_bf):
        """
        :param i: used to identify the correct parameters for the basis function
        :return: Values of Basisfunction on the whole s_grid with applied BCs
        """
        return self._eval_bf(i_bf) * self.BC_Mask

    def _eval_d_bf_with_BC__d_s_x(self, i_bf):
        """
        Uses the derivative addition rule to split the BC_Mask in a different class
        :return: Derivative of a bf with applied BCs wrt to s_x.
        """
        return self._eval_d_bf__ds_x(i_bf) * self.BC_Mask + self.bfs[i_bf] * self.BC_Mask_grad_s_x

    def _eval_d_bf_with_BC__d_s_y(self, i_bf):
        """
        Uses the derivative addition rule to split the BC_Mask in a different class
        :return: Derivative of a bf with applied BCs wrt to s_y.
        """
        return self._eval_d_bf__ds_y(i_bf) * self.BC_Mask + self.bfs[i_bf] * self.BC_Mask_grad_s_y
