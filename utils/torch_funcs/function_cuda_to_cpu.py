import torch


def cuda_to_cpu(*args):
    res = list()
    for arg in args:
        if torch.is_tensor(arg):
            if arg.is_cuda:
                arg = torch.Tensor.cpu(arg)
        res.append(arg)
    if len(res) == 1:
        return arg
    else:
        return *res,
