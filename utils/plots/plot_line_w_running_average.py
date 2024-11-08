import numpy as np
import matplotlib.pyplot as plt
import os


def plot_line_w_runnning_avg(vectors,
                             parameter_names=None,
                             running_avg_range=None,
                             results_path=None):
    """
        Plot one or multiple numpy vectors as line plots in one figure.
        Includes an additional line for each vector showing the running average.

        Parameters:
            vectors (list or np.ndarray): The input vector(s) to plot.
            parameter_names (list, optional): Names of the parameters for the legend. Default is None.
            running_avg_range (int, optional): Number of previous data points to consider for calculating the running average.
                                               Default is None, which calculates the running average for all previous data points.

        Returns:
            None
        """

    # Convert single vector to a list
    if not isinstance(vectors, list):
        vectors = [vectors]

    # Determine the number of vectors
    num_vectors = len(vectors)

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Set y-axis as logarithmic
    ax.set_yscale("log")

    # Set a colormap for the running averages
    cmap = plt.cm.get_cmap("tab10")

    # Plot each vector and its running average
    for i, vector in enumerate(vectors):
        # Plot the vector
        val_label = parameter_names[i] if parameter_names else f"Vector {i + 1}"
        ax.plot(vector, label=val_label)

        # Calculate the running average
        if running_avg_range is None:
            running_avg = np.cumsum(vector) / np.arange(1, len(vector) + 1)
        else:
            running_avg = np.convolve(vector, np.ones(running_avg_range) / running_avg_range, mode='same')

        # Plot the running average with a more prominent color
        avg_label = f"Moving Average ({parameter_names[i]})" if parameter_names else f"Moving Average (Vector {i + 1})"
        ax.plot(range(len(running_avg)), running_avg, linestyle="--", color=cmap(i), label=avg_label)

    # Turn on the grid
    ax.grid(True)

    # Add legend
    ax.legend()

    # Set labels
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")

    # save plot
    if results_path is not None:
        plt.savefig(os.path.join(results_path, val_label + "_w_running_aver" + ".png"), dpi=200)

    # Show the plot
    plt.show()
