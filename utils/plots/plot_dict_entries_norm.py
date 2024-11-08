import matplotlib.pyplot as plt
import numpy as np
import os


def plot_dict_entries_norm(dictionary,
                           iterations,
                           running_avg_window=None,
                           title='Exponential of Running Average of Log Norm',
                           results_path=None):
    """
    Plots the exponential of the running average of the logarithm of numpy arrays in a dictionary as a function of iterations in a single plot.

    Args:
        dictionary (dict): A dictionary where each entry contains multiple numpy arrays.
        iterations (array-like): An array or list of iteration numbers corresponding to the numpy arrays.
        running_avg_window (int, optional): The window size for calculating the running average.
            If None (default), the running average is calculated over all previous values.

    Returns:
        None

    """

    fig, ax = plt.subplots(figsize=(10, 6))

    for key, values in dictionary.items():
        norms = [np.linalg.norm(arr) for arr in values]
        log_norms = np.log(norms)

        if running_avg_window is not None:
            running_avg_log = np.convolve(log_norms, np.ones(running_avg_window) / running_avg_window, mode='valid')
            running_avg = np.exp(running_avg_log)
            iterations_running_avg = iterations[running_avg_window - 1:]
        else:
            running_avg_log = np.cumsum(log_norms) / np.arange(1, len(log_norms) + 1)
            running_avg = np.exp(running_avg_log)
            iterations_running_avg = iterations[:len(running_avg)]

        ax.plot(iterations_running_avg, running_avg, label=key)

        # Plot original data with transparency
        ax.plot(iterations[:len(norms)], norms, color=ax.lines[-1].get_color(), alpha=0.2)

    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Iterations')
    ax.set_ylabel(title)
    ax.set_title('Norm Value')
    ax.legend()

    plt.tight_layout()

    # save plot
    if results_path is not None:
        plt.savefig(os.path.join(results_path, title + "_w_running_aver.png"), dpi=200)

    plt.show()
