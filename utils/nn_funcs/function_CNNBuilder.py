# Import the necessary modules
import torch
import torch.nn as nn

def build_cnn(input_feature_dim, activation_func_name, feature_layers, average_pooling_layers=True):
    activation_func = getattr(nn, activation_func_name)() # get the activation function
    layers = [] # initialize the layers list
    # make sure that average_pooling_layers is a list
    if average_pooling_layers is bool:
        average_pooling_layers = [average_pooling_layers] * len(feature_layers)
    prev_dim = input_feature_dim

    for i, current_dim in enumerate(feature_layers):
        layers.append(nn.Conv2d(prev_dim, current_dim, kernel_size=3, stride=1, padding=1)) # changes the number of channels
        layers.append(activation_func)
        layers.append(nn.BatchNorm2d(current_dim))
        if average_pooling_layers[i]:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2)) # halfes the size of the image in each dimension
        prev_dim = current_dim

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

