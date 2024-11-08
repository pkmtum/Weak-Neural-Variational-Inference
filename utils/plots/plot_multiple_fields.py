import numpy as np
import matplotlib.pyplot as plt
import os
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu


def plot_multiple_fields(fields, mesh, titles=None, results_path=None):
    """
    (Author: ChatGPT, Vincent Scholz)
    Create subplots of multiple fields.

    Parameters:
        fields (torch.tensor): The input array containing multiple fields to plot.
        mesh (tuple): Tuple of mesh arrays (X, Y) representing the grid.
        titles (str or list, optional): Titles for each subplot or a single title to be applied to all subplots.
                                        If not provided, the subplot numbers will be used as titles.
                                        Default is None.
        results_path (str, optional): Absolute path to results folder

    Returns:
        None

    Commands:
        Please write a Python function to create multiple subplots. In each of the subplots, you shall plot a field
        using pcolormesh with the same given numpy two-dimensional regular grid mesh. The number of subplots shall be
        determined by the first dimension of the given input array. Given this, each subplot shall contain one field
        determined by the value of the other dimensions of the input array.
        Also, incorporate a list of titles you can specify for each subplot or just give a single title which then
        applies to all subplots with the according subplot number.

    Example:
        import numpy as np

        # Generate sample fields and mesh grid
        X, Y = np.meshgrid(np.arange(10), np.arange(10))
        fields = np.random.rand(5, 10, 10)  # 5 fields with shape (10, 10)
        # Plot the multiple fields
        titles = ['Field A', 'Field B', 'Field C', 'Field D', 'Field E']
        plot_multiple_fields(fields, (X, Y), titles=titles)

    """
    fields, mesh = cuda_to_cpu(fields, mesh)

    # Determine the number of subplots based on the first dimension of the input array
    if fields.ndim == 2:
        fields = fields.unsqueeze(0)
    if fields.ndim > 3:
        fields = fields.squeeze()

    num_subplots = fields.shape[0]

    # Calculate the number of rows and columns for the subplot grid
    num_rows = int(np.sqrt(num_subplots))
    num_cols = int(np.ceil(num_subplots / num_rows))

    # Create a figure with subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*5))

    # Flatten the subplot axes array if it's not already flattened
    if num_subplots > 1:
        axs = axs.flatten()

    # Plot the fields in each subplot
    for i in range(num_subplots):
        field = fields[i]
        ax = axs[i] if num_subplots > 1 else axs
        try:
            # plot = ax.pcolormesh(*mesh, field)
            plot = ax.pcolormesh(mesh[0].numpy(), mesh[1].numpy(), field.numpy())
        except:
            plot = ax.pcolormesh(field)

        # Set subplot title
        if titles is None:
            title = f'Field {i + 1}'
        elif isinstance(titles, str):
            title = titles + f' {i + 1}'
        elif isinstance(titles, list) and len(titles) == num_subplots:
            title = titles[i]
        else:
            raise ValueError("Invalid titles format. Please provide a single title or a list of titles matching the "
                             "number of subplots.")

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal', adjustable='box')
        fig.colorbar(plot, ax=ax)

    # Adjust spacing between subplots
    plt.tight_layout()

    if results_path is not None:
        plt.savefig(os.path.join(results_path, title + ".png"))

    # Display the plot
    plt.show()
    plt.close()
