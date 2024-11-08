import matplotlib.pyplot as plt
import numpy as np
import os
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu


def plot_parameter_evolution_individual_plots(true_values, history_values, title=None, results_path=None):
    """
    Plots true values along with their histories in a square formation of subplots.

    Args:
        true_values (numpy.ndarray): 1D array of true values (length N).
        history_values (numpy.ndarray): 2D array of history values (N x T).

    Returns:
        None
    """

    true_values, history_values = cuda_to_cpu(true_values, history_values)

    N, T = history_values.shape

    # Calculate the number of rows and columns for the square grid
    rows = int(np.sqrt(N))
    cols = int(np.ceil(N / rows))

    # Create subplots for each true value in a square grid
    fig, axs = plt.subplots(rows, cols, figsize=(8 * cols, 6 * rows))

    # Flatten the axs array if it's a 2D grid
    if rows > 1 and cols > 1:
        axs = axs.ravel()

    for i in range(N):
        # Plot the history values
        axs[i].plot(range(T), history_values[i], label='History', marker='o')

        # Plot the true value as a dashed horizontal line
        axs[i].axhline(y=true_values[i], color='r',
                       linestyle='--', label='True Value')

        # Set plot labels and title
        # axs[i].set_xlabel('Time Step (T)')
        # axs[i].set_ylabel('Value')
        axs[i].set_title(f'{title} {i + 1}')

        # Add legend
        # axs[i].legend()

    # Remove any empty subplots
    for i in range(N, len(axs)):
        fig.delaxes(axs[i])

    # Adjust subplot spacing
    plt.tight_layout()

    # Save the plot
    if results_path:
        plt.savefig(os.path.join(
            results_path, 'each_{}_with_history.png'.format(title)))

    # Show the plot
    plt.show()


# # Example usage:
# true_values = np.array([10, 20, 30, 40])
# history_values = np.array([[9, 15, 25, 35], [18, 22, 28, 38], [
#                           28, 33, 40, 45], [38, 42, 47, 55]])
# plot_parameter_evolution_individual_plots(true_values, history_values, title='test')
