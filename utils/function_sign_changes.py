import torch


def count_sign_changes(tensor):
    """
    Counts the sign changes in a torch tensor.
    """
    # get the sign of the tensor
    sign = torch.sign(tensor)
    # get the sign of the next element
    sign_next = torch.roll(sign, -1, -1)
    # get the sign change
    sign_change = sign != sign_next
    # count the sign changes
    sign_change_count = torch.sum(sign_change[..., :-1], dim=-1)
    return sign_change_count


# X = torch.tensor([[-1, 1, -1, 1, 1], [1, 1, 1, 1, 1]])
# print(count_sign_changes(X))
