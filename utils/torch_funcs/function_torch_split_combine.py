import torch

torch.set_default_dtype(torch.float64)


def torch_split(tensor, indices):
    """
    Splits a torch_funcs tensor into t1 = t[not(index)] and t2 = t[index].
    :param tensor: torch_funcs tensor
    :param indices: list with indices
    :return: t1, t2
    """

    # make a mask that shows where t1 is
    mask = torch.ones(tensor.numel(), dtype=torch.bool)
    mask[indices] = False
    # mask that shows where t2 is
    mask_inv = torch.logical_not(mask)
    # split the tensor.
    return tensor[mask], tensor[mask_inv]


class torch_combine:
    def __init__(self, dim_y, indices_of_t2):
        # make a mask that shows where t1 is going
        mask = torch.ones(torch.Size([dim_y]), dtype=torch.bool)
        mask[indices_of_t2] = False
        self.mask = mask
        # mask that shows where t2 is going
        self.mask_inv = torch.logical_not(mask)

        # new tensor with t1 and t2
        self.t = torch.zeros_like(mask, dtype=torch.float64)

    def apply(self, tensor_1, tensor_2):
        """
        Recombines t1 and t2 which were split by torch_split
        :param tensor_1: torch_funcs tensor t[not(index)]
        :param tensor_2: torch_funcs tensor t[index]
        :return: t
        """

        t = torch.clone(self.t)
        t[self.mask] = tensor_1
        t[self.mask_inv] = tensor_2

        return t
