import torch

def roll_columns(T):
    rolled_tensors = []
    n, m = T.size()
    for i in range(m):
        rolled_tensor = torch.roll(T, shifts=-i, dims=1)
        rolled_tensors.append(rolled_tensor)
    return rolled_tensors
