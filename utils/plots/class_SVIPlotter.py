import matplotlib.pyplot as plt
import numpy as np
import torch
# import dolfinx as dfx
import os
# import pyvista as pv

# from utils.plots.function_dfx_to_pv import dfx_to_pv
from utils.function_integer_factors import integer_factors
# from utils.fenics_funcs.function_dfx_dim import dfx_dim

# pv.global_theme.transparent_background = True
# pv.global_theme.font.color = 'black'

def max_value(inputlist):
    return max([i for lis in inputlist for i in lis])


def min_value(inputlist):
    return min([i for lis in inputlist for i in lis])

class SVIPlotter:
    def __init__(self, log, results_path):
        self.log = log
        self.results_path = results_path

    def plot_ELBOs(self, tag_list):
        # check automatic size and num of subplots
        i_max = len(tag_list)
        n_columns, n_rows = integer_factors(int(i_max))
        # n_columns = int(np.sqrt(i_max))
        # n_rows = int(np.ceil(i_max / n_columns))

        plt.figure(figsize=(n_columns * 3.5, n_rows * 3))

        # plot the data over iterations
        iterations = self.log.iteration

        # loop over all tags given for plot
        for i, tag in enumerate(tag_list):
            plt.subplot(n_rows, n_columns, int(i + 1))
            data = np.asarray(getattr(self.log, tag))

            # checks if we have an array inside the dict -> only plot first element
            # if scalar -> plot normally
            if hasattr(data[0], "__len__"):
                data_for_plot = np.zeros((data.shape[0],))
                tag = tag + " (1. element)"
                for j in range(data.shape[0]):
                    data_for_plot[j] = data[j].flat[0]
            else:
                data_for_plot = data

            # if these tags appear, use log scale
            log_tags = ["elbo", "stepsize"]
            log_1 = any(x in tag for x in log_tags)
            # if data spans multiple scales, use log scale
            log_2 = (abs(max(data_for_plot)) / abs(min(data_for_plot))) > 1e2
            log_scale = log_1 or log_2

            # change tag and scale to log
            if log_scale:
                data_for_plot = np.absolute(data_for_plot)
                tag = "|" + tag + "|"
                plt.yscale('log')

            # plt.xscale('log')
            plt.plot(iterations, data_for_plot)
            plt.gca().set_title(tag)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "grad_ELBOs.png"))
        plt.show()

    def plot_ELBO_parts(self):
        # check automatic size and num of subplots
        i_max = len(self.log.elbo_parts)
        n_columns, n_rows = integer_factors(int(i_max))
        # n_columns = int(np.sqrt(i_max))
        # n_rows = int(np.ceil(i_max / n_columns))

        plt.figure(figsize=(n_columns * 3.5, n_rows * 3))

        # plot the data over iterations
        iterations = self.log.iteration

        # loop over all tags given for plot
        for i, tag in enumerate(self.log.elbo_parts):
            plt.subplot(n_rows, n_columns, int(i + 1))
            data = np.asarray(self.log.elbo_parts[tag])

            # checks if we have an array inside the dict -> only plot first element
            # if scalar -> plot normally
            if hasattr(data[0], "__len__"):
                data_for_plot = np.zeros((data.shape[0],))
                tag = tag + " (1. element)"
                for j in range(data.shape[0]):
                    data_for_plot[j] = data[j].flat[0]
            else:
                data_for_plot = data

            # if data spans multiple scales, use log scale
            abs_data = abs(data_for_plot)
            log_scale = (max(abs_data) / (min(abs_data)+1e-8)) > 1e2
            # # in this case, only if all values are positive (so you see the negative values)
            # log_3 = all(j >= 0 for j in data_for_plot)
            # log_scale = log_2 and log_3

            # change tag and scale to log
            if log_scale:
                plt.yscale('symlog')
            tag = "elbo_" + tag

            # plt.xscale('log')
            plt.plot(iterations, data_for_plot)
            plt.gca().set_title(tag)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "ELBO_parts.png"))
        plt.show()

    # def print_results(self, true_MF):
    #     m_SVI = np.array(self.log["m"][-1])
    #     m_true = true_MF.xi_l
    #     weights = true_MF.C_eigenValues[:len(m_true)] / sum(true_MF.C_eigenValues[:len(m_true)]) * 100
    #     variance = np.diag(np.array(self.log["S"][-1]))
    #     results = pd.DataFrame(np.array([m_SVI.T, m_true.T, variance, weights]).T,
    #                            columns=["True Data", "SVI mean", "SVI std (main diag)", "EV weight [%]"])
    #     print(results)

    # def plot_convergence_mean(self, xi_true, string_means):
    #     m_SVI = np.asarray(torch_funcs.stack(self.log.posterior_parameters[string_means]))
    #     iteration = self.log.iteration
    #     xi_true = np.asarray(xi_true).squeeze()
    #     # check automatic size and num of subplots
    #     i_max = min(16, len(m_SVI[0, :]))  # only plot maximum 16 pannels
    #     n_columns, n_rows = integer_factors(int(i_max))
    #     # n_columns = int(np.sqrt(i_max))
    #     # n_rows = int(np.ceil(i_max / n_columns))
    #
    #     plt.figure(figsize=(n_columns * 4, n_rows * 3))
    #
    #     for count in range(i_max):
    #         try:
    #             plt.subplot(n_rows, n_columns, int(count + 1))
    #
    #             plt.plot(iteration, m_SVI[:, count])
    #             plt.title(label="Var {}".format(count + 1))
    #             plt.axhline(y=xi_true[count], color='r', linestyle='-')
    #         except:
    #             pass
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.results_path, "means.png"))
    #     plt.show()

    # def plot_S(self):
    #     S_SVI = np.asarray(self.log["S"])
    #     iteration = self.log["iteration"]
    #     dim = len(S_SVI[0])
    #     plt.figure(figsize=(10, 6), dpi=300)
    #     for count in range(dim):
    #         plt.plot(iteration, S_SVI[:, count, count], label='S {0},{0}'.format(int(count + 1)))
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    # def plot_theta(self, V):
    #
    #
    #     my_theta = self.log.theta[-1]
    #     u_grid = dfx_to_pv(dfx.fem.Function(V), "theta", my_theta)
    #
    #     u_plotter = pv.Plotter(off_screen=True)
    #     u_plotter.add_mesh(u_grid, show_edges=True)
    #     u_plotter.add_text("Weighting function w", font_size=24, position='upper_edge')
    #     u_plotter.view_xy()
    #     u_plotter.screenshot(os.path.join(self.results_path, "theta.png"))
    #

        # plt.figure(figsize=(6, 6), dpi=300)
        # plt.title("Weighting function w")
        # plt.colorbar(df.plot(fenics_fun))
        # plt.savefig(os.path.join(self.results_path, "theta.png"))
        # plt.show()
