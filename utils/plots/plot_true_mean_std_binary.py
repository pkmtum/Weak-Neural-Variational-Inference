import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu
from utils.plots.function_robust_data_range import compute_robust_data_range


def plot_true_mean_std_binary(true_field, posterior_mean_field, std_field, mesh, suptitle=None, results_path=None):
    """
    (Author: ChatGPT, Vincent Scholz)
    Create subplots of fields and a binary plot.

    Parameters:
        true_field (np.ndarray): The true field to plot.
        posterior_mean_field (np.ndarray): The posterior mean field to plot.
        std_field (np.ndarray): The standard deviation field to plot.
        mesh (tuple): Tuple of mesh arrays (X, Y) representing the grid.

    Returns:
        None

    Command:
        Please write a Python function to create 4 subplots in a 2 by 2 grid fashion. In each of the subplots, you
        shall plot a field using pcolormesh with the same given numpy two-dimensional regular grid mesh. The first
        subplot shows the "true" field, the second subplot the "posterior mean" field and the third subplot the
        "Standard deviation" field. The fourth subplot shall be a plot, which is 0 where the "posterior mean" field
        plus or minus the "Standard deviation" field does not enclose the "true" field, and 1 where it does enclose it.

    Example:
        import numpy as np

        # Generate sample fields and mesh grid
        X, Y = np.meshgrid(np.arange(10), np.arange(10))
        true_field = np.random.rand(10, 10)
        posterior_mean_field = np.random.rand(10, 10)
        std_field = np.random.rand(10, 10)

        # Plot the fields
        plot_fields(true_field, posterior_mean_field, std_field, (X, Y))

    """
    true_field, posterior_mean_field, std_field, mesh = cuda_to_cpu(true_field, posterior_mean_field, std_field, mesh)
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Set subplot titles
    titles = ['True Field', 'Posterior Mean', 'Standard Deviation', 'Envelops truth by 2 sigma']
    
    # same data range for true and approx fields
    true_field_min = torch.min(true_field)
    true_field_max = torch.max(true_field)
    posterior_mean_field_min = torch.min(posterior_mean_field)
    posterior_mean_field_max = torch.max(posterior_mean_field)
    # only different range when approx field is outside true field range
    if true_field_min > posterior_mean_field_min or true_field_max < posterior_mean_field_max:
        all_data_points = torch.concatenate((true_field.flatten(), posterior_mean_field.flatten()))
        low_value, high_value = compute_robust_data_range(all_data_points)
        if low_value > true_field_min:
            low_value = true_field_min
        if high_value < true_field_max:
            high_value = true_field_max
        norm = colors.PowerNorm(gamma=1, vmin=low_value, vmax=high_value)
    else:
        norm = colors.PowerNorm(gamma=1, vmin=true_field_min, vmax=true_field_max)

    # Plot the fields and binary plot
    for i, field, ax, title in zip(range(4),
                                   [true_field, posterior_mean_field, std_field, (posterior_mean_field, std_field)],
                                   axs.flatten(), titles):
        if i == 3:  # Plot the binary enclosure
            posterior_plus_std = posterior_mean_field + 2 * std_field
            posterior_minus_std = posterior_mean_field - 2 * std_field
            enclosure = np.where((posterior_plus_std >= true_field) & (posterior_minus_std <= true_field), 1, 0)
            try:
                plot = ax.pcolormesh(mesh[0].numpy(), mesh[1].numpy(), enclosure, cmap='binary')
            except:
                plot = ax.pcolormesh(enclosure, cmap='binary')
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plot.set_clim(0, 1)  # Set the colorbar limits
            fig.colorbar(plot, ax=ax, ticks=[0, 1])
        elif i == 0 or i == 1:  # Plot the fields
            try:
                plot = ax.pcolormesh(mesh[0].numpy(), mesh[1].numpy(), field, norm=norm)
            except:
                plot = ax.pcolormesh(field, norm=norm)
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            fig.colorbar(plot, ax=ax)
        else:  
            # Plot the fields
            try:
                plot = ax.pcolormesh(mesh[0].numpy(), mesh[1].numpy(), field)
            except:
                plot = ax.pcolormesh(field)
            ax.set_title(title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            fig.colorbar(plot, ax=ax)

    # main title
    plt.suptitle(suptitle)

    # Adjust spacing between subplots
    plt.tight_layout()

    if results_path is not None:
        plt.savefig(os.path.join(results_path, suptitle + ".png"))

    # Display the plot
    plt.show()
