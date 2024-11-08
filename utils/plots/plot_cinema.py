import matplotlib.pyplot as plt
import numpy as np
import os
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu



# def plot_cinema_mpl(u_true, foo, s_grid, title, results_path, *args):
#     fig = plt.figure(figsize=(12, 6))
#     gs = fig.add_gridspec(3, 4, width_ratios=[4, 1, 1, 1])

#     # Create the right-hand side subplot for the original field 'u'
#     ax_main = fig.add_subplot(gs[:, 0])

#     # Generate the mesh
#     s_grid = cuda_to_cpu(s_grid)
#     X = np.asarray(s_grid[0, :, :])
#     Y = np.asarray(s_grid[1, :, :])

#     # Plot the original field 'u' on the right-hand side
#     u_true_plot = cuda_to_cpu(u_true)
#     cax_main = ax_main.pcolormesh(X, Y, u_true_plot, shading='auto', cmap='viridis')
#     fig.colorbar(cax_main, ax=ax_main)

#     # Generate variations of 'u' and plot them on the subplots
#     for i in range(9):
#         row = i // 3
#         col = i % 3
#         ax = fig.add_subplot(gs[row, col + 1])
#         variation = np.asarray(cuda_to_cpu(foo(u_true, *args)))
#         cax = ax.pcolormesh(X, Y, variation, shading='auto', cmap='viridis')
#         # fig.colorbar(cax, ax=ax)

#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(os.path.join(results_path, title + ".png"), dpi=200)
#     plt.show()


def plot_cinema_mpl(u_true, foo, s_grid, title, results_path, *args):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 4, width_ratios=[4, 1, 1, 1])

    # Create the right-hand side subplot for the original field 'u'
    ax_main = fig.add_subplot(gs[:, 0])

    # Generate the mesh
    s_grid = cuda_to_cpu(s_grid)
    X = np.asarray(s_grid[0, :, :])
    Y = np.asarray(s_grid[1, :, :])

    # Plot the original field 'u' on the right-hand side
    u_true_plot = cuda_to_cpu(u_true)
    cax_main = ax_main.pcolormesh(X, Y, u_true_plot, shading='auto', cmap='viridis')
    # colorbar_main = fig.colorbar(cax_main, ax=ax_main)

    # Generate variations of 'u' and plot them on the subplots
    cax_min, cax_max = np.max(np.asarray(u_true_plot)), np.min(np.asarray(u_true_plot))  # Initialize color bar limits

    for i in range(9):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col + 1])
        variation = np.asarray(cuda_to_cpu(foo(u_true, *args)))
        cax = ax.pcolormesh(X, Y, variation, shading='auto', cmap='viridis')
        cax_min = min(cax_min, np.min(variation))
        cax_max = max(cax_max, np.max(variation))

    # Set the same color bar range for all plots
    for i in range(9):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col + 1])
        cax = ax.pcolormesh(X, Y, variation, shading='auto', cmap='viridis', vmin=cax_min, vmax=cax_max)
        # colorbar = fig.colorbar(cax, ax=ax)
    
    cax_main = ax_main.pcolormesh(X, Y, u_true_plot, shading='auto', cmap='viridis', vmin=cax_min, vmax=cax_max)
    colorbar_main = fig.colorbar(cax_main, ax=ax_main)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, title + ".png"), dpi=200)
    plt.show()


# def plot_cinema_FEniCSX(u_true, foo, fenics_fun, title, results_path, *args):
#     """
#     :param u_true: Ground truth the noise is applied to
#     :param foo: Function that applies the noise. First argument has to be the ground truth field.
#     :param args: other parameters for function foo. Mind the order of the arguments!
#     :return: 3 x 3 plot for ground truth + noise + plot of ground truth
#     """
#     colorbar_range = [min(u_true), max(u_true)]
#
#     # set up plotter
#     plotter = pv.Plotter(shape=(3, 4), off_screen=True)
#     plotter.add_text(title, font_size=14)
#
#     # plot true field
#     plotter.subplot(1, 0)
#     # prepare plot.
#     u_grid = dfx_to_pv(fenics_fun, "value", fun_values=np.asarray(u_true))
#     # plot pyvista
#     plotter.add_mesh(u_grid, show_edges=True, clim=colorbar_range)
#     plotter.add_text("True field", font_size=14)
#     plotter.view_xy()
#
#     for i in range(3):
#         for j in range(3):
#             plotter.subplot(i, j + 1)
#             # apply noise
#             u_obs = foo(u_true, *args)
#             # create fenics_funcs function to plot
#             u_grid = dfx_to_pv(fenics_fun, "value", fun_values=np.asarray(u_obs))
#             # plot
#             plotter.add_mesh(u_grid, show_edges=True, clim=colorbar_range, show_scalar_bar=False)
#             plotter.view_xy()
#
#     plotter.screenshot(os.path.join(results_path, title + ".png"))
