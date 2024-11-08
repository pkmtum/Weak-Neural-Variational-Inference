import torch


def check_if_scalar(x):
    # check if we have a torch_funcs object on our hands
    if torch.is_tensor(x):
        # now it is either scalar or not
        if x.size() == torch.Size([]):
            return True
        else:
            return False
    # else we assume it is float or list
    # check if it has a length (== no scalar)
    # Question: What about e.g. np.array([1])? Isn't this "scalar" like torch_funcs.tensor(1)?
    # Answer: Yeah, but I write this to help me work with torch_funcs, so I don't want np.arrays anyway because
    # you cant just calculate together torch_funcs and np objects.
    elif hasattr(x, "__len__"):
        return False
    else:
        return True
