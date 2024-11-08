# Import the necessary modules
import torch
import torch.nn as nn

def build_ffn(input_dim, output_dim, activation_func_name, hidden_layers):
    activation_func = getattr(nn, activation_func_name)()
    layers = []
    prev_dim = input_dim

    for num_neurons in hidden_layers:
        layers.append(nn.Linear(prev_dim, num_neurons))
        layers.append(activation_func)
        prev_dim = num_neurons

    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


# # Define the input and output dimensions
# input_dim = 10
# output_dim = 5

# # Define the activation function name and hidden layers
# activation_func_name = "ReLU"
# hidden_layers = [20, 15, 10]

# # Build the feedforward neural network
# ffn = build_ffn(input_dim, output_dim, activation_func_name, hidden_layers)

# # Print the network architecture
# print(ffn)

