import torch
from utils.function_slice_it import slice_it
from utils.Integrator.function_integrator_2D import trapzInt2D

class nParametersToFields:
    """
    Here you can stack as many fields as you want and evaluate them together.
    """

    def __init__(self, list_ParameterToField):
        # list with entries of class ParameterToField
        self.list = list_ParameterToField

        # length of required parameters list input
        self.list_n_bfs = [a.num_unknowns for a in self.list]
        self.len_parameter_list = sum(self.list_n_bfs)
        # list which tells you how to split the parameter vector
        self.indices = []
        for i in range(len(self.list_n_bfs)):
            if i < len(self.list_n_bfs) - 1:
                self.indices.append(sum(self.list_n_bfs[:i + 1]))

        # show me the s_grid
        self.s_grid = list_ParameterToField[0].s_grid

    #%% Public functions
    def eval(self, parameters):
        """
        call this to evaluate all basis functions the ParameterToField tensors from here
        :param parameters: torch_funcs.tensor vector with the parameters
        :return: [n x dim_s1 x dim_s2] torch_funcs tensor
        """
        # there is a special case, where we input the whole field and not the parameters
        if parameters.size(-1) == self.len_parameter_list:
            return self._eval_list(parameters, 'eval')
        # This is the uggliest BYPASS ever. I am sorry for this.
        # NOTE: This only works for a single sample. If you want to evaluate multiple samples, you have to do it yourself.
        else:
            # check that given fields + fields that dont need parameters == len(self.list)
            if parameters.dim() == 2: # it should be 3
                parameters = parameters.unsqueeze(0)
            num_fields = parameters.size(0)
            num_fields_no_parameters = len([a for a in self.list_n_bfs if a == 0])
            assert num_fields + num_fields_no_parameters == len(self.list), "The number of fields (+ fields that dont need parameters) given is not equal to the number of fields required."
            counter = 0
            for i, ParameterToField in enumerate(self.list):
                if self.list_n_bfs[i] == 0:
                    field = ParameterToField.eval(None)
                else:
                    field = parameters[counter, :, :]
                    counter += 1
                field = field.unsqueeze(-3)
                if i == 0:
                    # save it in the first iteration or...
                    result = field
                else:
                    # ... stack them in a tensor
                    result = torch.concat((result, field), -3)
            return result

    def eval_grad_s(self, parameters):
        """
        call this to evaluate all spacial derivatives of the basis functions the ParameterToField tensors from here
        :param parameters: torch_funcs.tensor vector with the parameters
        :return: [n x ... x dim_s1 x dim_s2] torch_funcs tensor
        """
        return self._eval_list(parameters, 'eval_grad_s')
    
    def eval_WF(self, parameters):
        """
        this is a special function to evaluate the weight function.
        It is different, since the abs of the weight function is normalized to 1.
        """
        return self._eval_list(parameters, 'eval_WF')
    
    def eval_grad_s_WF(self, parameters):
        return self._eval_list(parameters, 'eval_grad_s_WF')

    #%% Private functions
    def _eval_list(self, parameters, name_function_from_ParameterToField):
        """

        :param parameters: torch_funcs.tensor vector with the parameters
        :param name_function_from_ParameterToField: string that gives the name of the function we want to call
        :return: [n x ... x dim_s1 x dim_s2] torch_funcs tensor
        """
        
        # split list along the indices
        parameter_lists = slice_it(parameters, self.indices)

        # initialize result
        result = None

        # loop thought all fields
        for i, ParameterToField in enumerate(self.list):
            # find the function you want to evaluate from the given string
            fun = getattr(ParameterToField, name_function_from_ParameterToField)
            # evaluate the functions
            field = fun(parameter_lists[i])


            if name_function_from_ParameterToField == "eval" or name_function_from_ParameterToField == "eval_WF":
                # make its dimensions useful
                field = field.unsqueeze(-3)
                if i == 0:
                    # save it in the first iteration or...
                    result = field
                else:
                    # ... stack them in a tensor
                    result = torch.concat((result, field), -3)
            elif name_function_from_ParameterToField == "eval_grad_s" or name_function_from_ParameterToField == "eval_grad_s_WF":
                # make its dimensions useful
                field = field.unsqueeze(-4)
                if i == 0:
                    # save it in the first iteration or...
                    result = field
                else:
                    # ... stack them in a tensor
                    result = torch.concat((result, field), -4)


        return result
