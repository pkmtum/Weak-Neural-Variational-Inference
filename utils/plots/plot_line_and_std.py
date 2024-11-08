import numpy as np
import os
import matplotlib.pyplot as plt
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu


def plot_line_with_std(approx_mean, std, true_value=None, obs_points=None, results_path=None, title=None):
    """
    (Author: ChatGPT, Vincent Scholz)
    Plot a line with its given standard deviation as a shaded area.

    Parameters:
        :param approx_mean: (array-like): Array of values representing the approximate mean line.
        :param std: (array-like): Array of values representing the standard deviation for each point of the line.
        :param true_value: (array-like, optional): Array of values representing the true value line (default: None).
        :param title: optional, string: name of parameter
        :param results_path: optional, string: absolute path to saving location

    Returns:
        None

    """
    approx_mean, std, true_value, obs_points = cuda_to_cpu(approx_mean, std, true_value, obs_points)
    if true_value is not None:
        true_value = true_value.numpy()
    if obs_points is not None:
        obs_points = obs_points.numpy()

    approx_mean, std = approx_mean.numpy(), std.numpy()
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Define the x-axis values
    x = np.arange(len(approx_mean))
    
    # Plot the approximation mean line
    ax.plot(x, approx_mean, color='blue', label='Approx. Mean')

    # Calculate the upper and lower bounds of the shaded region
    upper_bound = approx_mean + 2 * std
    lower_bound = approx_mean - 2 * std

    # Fill the area between the upper and lower bounds
    ax.fill_between(x, lower_bound, upper_bound, color='blue', alpha=0.3, label=r'$\pm 2 \sigma$')

    # Plot the true value line if provided
    if true_value is not None:
        ax.plot(x, true_value, color='red', label='True Value')
    
    if obs_points is not None:
        delta = len(approx_mean) / (len(obs_points) - 1)
        x_obs = []
        for i in range(len(obs_points)):
            x_obs.append(int(i*delta))
        ax.scatter(x_obs, obs_points, color='black', marker='X', label='Observed Points')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title is None:
        ax.set_title('Parameters with Standard Deviation')
    else:
        ax.set_title(title + ' with Standard Deviation')

    # Add a legend
    ax.legend()

    # save plot
    if results_path is not None:
        plt.savefig(os.path.join(results_path, title + "_w_std" + ".png"), dpi=200)

    # Display the plot
    plt.show()
    plt.close()
