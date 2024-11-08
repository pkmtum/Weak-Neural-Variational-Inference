import torch

def function_index_to_2D_index(i, counter_inner_max):
    """
    This function converts a function index to a 2D index.
    :param i: function index
    :param c_1_max: maximum number of functions in the first dimension
    :param c_2_max: maximum number of functions in the second dimension
    :return: 2D index
    """
    return torch.tensor([i // counter_inner_max, i % counter_inner_max])