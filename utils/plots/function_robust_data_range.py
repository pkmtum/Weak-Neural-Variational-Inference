import torch
import numpy as np


def compute_robust_data_range(data, percentile=95):
    data = data.flatten()
    low = np.percentile(data, 100 - percentile)
    high = np.percentile(data, percentile)
    return low, high


# data = torch.rand((5, 5)) * 5
# data = torch.exp(data)
# print(data.flatten())
# low, high = compute_robust_data_range(data)
# print(low)
# print(high)