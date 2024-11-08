import torch

def create_difference_matrix(x, y):
    # Reshape tensors for broadcasting
    x = x.view(-1, 1)  # Reshape to (n, 1)
    y = y.view(1, -1)  # Reshape to (1, m)
    
    # Calculate difference matrix
    difference_matrix = x - y
    
    return difference_matrix

