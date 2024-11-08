import torch
from utils.torch_funcs.function_check_if_scalar import check_if_scalar


def make_torch_tensor(T, shape):
    if check_if_scalar(T):
        # if your shape is a square matrix: return eye with T on main diagonal
        if len(shape) > 1:
            if shape[0] == shape[1]:
                return torch.eye(shape[0]) * T
            else:
                raise RuntimeError(f"This function does not support this shape. Only shapes supported are vectors " + \
                                   "and diagonal matrices. You shape input was {shape}.")
        # if shape is vector: return ones with value T
        else:
            return torch.ones(shape) * T

    # if you don't have a scalar: just return T (after checking the dimensions)
    else:
        assert T.size() == shape, f"Your input tensor has shape {T.size()}, but you requested shape {shape}. " + \
                                  "Please check your input."
        return T
