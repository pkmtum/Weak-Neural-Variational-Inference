import torch
import numpy as np
import warnings

# plots
from utils.plots.class_SVIPlotter import SVIPlotter
from utils.plots.plot_cinema import plot_cinema_mpl
from utils.plots.plot_parameter_evolution import plot_parameter_evolution
from utils.plots.plot_multiple_fields import plot_multiple_fields
from utils.plots.plot_line_w_running_average import plot_line_w_runnning_avg
from utils.plots.plot_true_mean_std_binary import plot_true_mean_std_binary
from utils.plots.plot_line_and_std import plot_line_with_std
from utils.plots.plot_dict_entries_norm import plot_dict_entries_norm
from utils.plots.plot_abs_error_std import plot_abs_error_std
from utils.plots.plot_parameter_evolution_individual_plots import plot_parameter_evolution_individual_plots

# torch
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu

# others
from utils.function_FEniCS_reference_solve import solve_hook_pde_fenics

def postprocessing(my_log,
                   my_PDE,
                   posterior,
                   XToField,
                   YToField,
                   ThetaToField,
                   s_grid,
                   MF_true,
                   u_true,
                   u_hat_full,
                   sigma_u_hat,
                   y_true,
                   observer,
                   result_path):

    plotter = SVIPlotter(my_log,
                         result_path)
    # plotter.plot_convergence_mean(x_true,
    #                               "x_mean")
    plotter.plot_ELBOs(["grad_elbo",
                        "grad_elbo_theta",
                        "grad_elbo_phi",
                        "Lambda"])
    plotter.plot_ELBO_parts()
    try:
        plot_line_w_runnning_avg([my_log.elbo], ["ELBO"], running_avg_range=None, results_path=result_path)
    except:
        warnings.warn("Plotting: Crashed while plotting ELBO.")
    plot_dict_entries_norm(my_log.grad_elbo_phi_parts, my_log.iteration,
                           title="grad_elbo_phi", results_path=result_path)

    # plot the Evolutions of the mean y's and x's
    # plot the Evolutions of the mean y's and x's
    if "x_0" in my_log.posterior_parameters.keys():
        x_mean_evol = np.asarray(torch.stack(my_log.posterior_parameters["x_0"]).T)
    if "cov_x_parameters" in my_log.posterior_parameters.keys():
        # gives all of cov matrix (non-squared)
        x_cov_evol = np.asarray(torch.stack(
            my_log.posterior_parameters["cov_x_parameters"]).T)
        x_var_evol = x_cov_evol
    if "y_0" in my_log.posterior_parameters.keys():
        y_mean_evol = np.asarray(torch.stack(my_log.posterior_parameters["y_0"]).T)
    if "cov_y_parameters" in my_log.posterior_parameters.keys():
        # gives all of cov matrix (non-squared)
        y_cov_evol = np.asarray(torch.stack(
            my_log.posterior_parameters["cov_y_parameters"]).T)
        y_var_evol = y_cov_evol
    theta_evol = np.asarray(my_log.theta).T
    if "JumpPenalty" in my_log.posterior_parameters.keys():
        jumpPenalty_evol = np.asarray(torch.stack(my_log.posterior_parameters["JumpPenalty"]).T)
    try:
        plot_parameter_evolution(
            x_mean_evol, results_path=result_path, param_name="mean of x")
    except:
        warnings.warn("Plotting: Crashed while plotting evolution of mean of q(x).")
    # this plot can cause problems and kill the job
    try:
        plot_parameter_evolution(
            x_var_evol, results_path=result_path, param_name="Var[x]")
    except:
        warnings.warn("Plotting: Crashed while plotting evolution of variance of q(x).")
    try:
        plot_parameter_evolution(
            y_var_evol, results_path=result_path, param_name="Var[y]")
    except:
        warnings.warn("Plotting: Crashed while plotting evolution of variance of q(y).")
    try:
        plot_parameter_evolution(
            y_mean_evol, results_path=result_path, param_name="mean of y")
    except: 
        warnings.warn("Plotting: Crashed while plotting evolution of mean of q(y).")
    try:
        plot_parameter_evolution(
            theta_evol, results_path=result_path, param_name="theta")
    except:
        warnings.warn("Plotting: Crashed while plotting evolution of theta.")
    # plot_parameter_evolution(
    #     jumpPenalty_evol, results_path=result_path, param_name="JumpPenalty")
    # plot_parameter_evolution_individual_plots(
    #     y_true, y_mean_evol, title="y", results_path=result_path)
    # Grad Elbo and it's parts and their running averages
    plot_line_w_runnning_avg([my_log.grad_elbo_phi, my_log.grad_elbo_theta, my_log.grad_elbo],
                                [r'$|\nabla_\varphi \textrm{ELBO}|$',
                                r'$|\nabla_\theta \textrm{ELBO}|$',
                                r'$|\nabla \textrm{ELBO}|$'],
                                running_avg_range=None)

    # Get the mean field and standard deviation fields
    # extract mean fields directly
    # MF_mean = XToField.eval(posterior.Posteriors["x"].mean_x.clone().detach())
    # u_mean = YToField.eval(posterior.Posteriors["y"].mean_y.clone().detach())
    # u_part_mean = observer.filter(u_mean)
    # standard deviation
    # x_sample, y_sample = posterior.sample()
    # x_sample, y_sample = x_sample.clone().detach(), y_sample.clone().detach()
    # MF_samples, u_samples = XToField.eval(x_sample), YToField.eval(y_sample)
    # MF_samples, u_samples = MF_samples.unsqueeze(0), u_samples.unsqueeze(0)

    samples = posterior.sample(num_samples=1_000)
    if "x" in samples.keys():
        x_sample = samples["x"].clone().detach()
        x_mean = torch.mean(x_sample, dim=0)
        MF_samples= XToField.eval(x_sample)
        MF_mean = MF_samples.mean(dim=0)
        MF_std = torch.std(MF_samples, dim=0)
        # plot true value + posterior mean + posterior standard deviation + enclosure True/False
        plot_true_mean_std_binary(MF_true[0], MF_mean[0], MF_std[0], s_grid,
                                suptitle=r'Posterior - Material Field $log(E)$', results_path=result_path)
        # plot of diagonal true value + posterior mean + posterior standard deviation
        plot_line_with_std(torch.diagonal(MF_mean[0], 0),
                        torch.diagonal(MF_std[0], 0),
                        true_value=torch.diagonal(MF_true[0], 0),
                        title=r'Material Field $log(E)$',
                        results_path=result_path)
    else:
        warnings.warn("Plotting: No x in the samples")
    
    if "y" in samples.keys():
        y_sample = samples["y"].clone().detach()
        u_samples = YToField.eval(y_sample)
        u_mean = u_samples.mean(dim=0)
        u_part_mean = observer.filter(u_mean)

        u_std = torch.std(u_samples, dim=0)
        u_part_std = observer.filter(u_std)
        plot_true_mean_std_binary(u_true[0], u_mean[0], u_std[0], s_grid,
                            suptitle=r'Posterior - Displacements $u_1$', results_path=result_path)
        plot_true_mean_std_binary(u_true[1], u_mean[1], u_std[1], s_grid,
                                suptitle=r'Posterior - Displacements $u_2$', results_path=result_path)
        plot_abs_error_std(u_hat_full[0], u_mean[0], sigma_u_hat, s_grid,
                       suptitle=r'Absolute Error $u_1$', results_path=result_path)
        plot_abs_error_std(u_hat_full[1], u_mean[1], sigma_u_hat, s_grid,
                        suptitle=r'Absolute Error $u_2$', results_path=result_path)
        
        u_part_true = observer.filter(u_true)
        u_part_obs = observer.filter(u_hat_full)
        plot_abs_error_std(u_part_true[0], u_part_mean[0], sigma_u_hat, observer.filter(s_grid),
                                suptitle=r'Comparison $u_1$ at Obs', results_path=result_path)
        plot_abs_error_std(u_part_true[1], u_part_mean[1], sigma_u_hat, observer.filter(s_grid),
                                suptitle=r'Comparison $u_2$ at Obs', results_path=result_path)
        plot_abs_error_std(u_part_obs[0], u_part_mean[0], sigma_u_hat, observer.filter(s_grid),
                                suptitle=r'Observations comparison $u_1$', results_path=result_path)
        plot_abs_error_std(u_part_obs[1], u_part_mean[1], sigma_u_hat, observer.filter(s_grid),
                                suptitle=r'Observations comparison $u_2$', results_path=result_path)
        
        plot_line_with_std(torch.diagonal(u_mean[0], 0),
                    torch.diagonal(u_std[0], 0),
                    true_value=torch.diagonal(u_true[0], 0),
                    obs_points=torch.diagonal(u_part_obs[0], 0),
                    title=r'Solution field $u_1$',
                    results_path=result_path)
        
        try: 
            plot_line_with_std(u_mean[1,:,0],
                            u_std[1,:,0],
                            true_value=u_true[1,:,0],
                            obs_points=u_part_obs[1,:,0],
                            title=r'Solution field $u_2$ at $x_1=0$',
                            results_path=result_path)
            
            plot_line_with_std(u_mean[0,0,:],
                            u_std[0,0,:],
                            true_value=u_true[0,0,:],
                            obs_points=u_part_obs[0,0,:],
                            title=r'Solution field $u_2$ at $x_2=0$',
                            results_path=result_path)
        except:
            warnings.warn("Plotting: Was not able to BCs plot line with std.")
        
        try:
            du_samples = YToField.eval_grad_s(y_sample)
            du_mean = du_samples.mean(dim=0)
            du_mean = du_mean.flatten(start_dim=0, end_dim=1)
            plot_multiple_fields(du_mean, s_grid, results_path=result_path, titles=["dux_dx", "dux_dy", "duy_dx","duy_dy"])
        except:
            warnings.warn("Plotting: Was not able to plot du(x)/dx.")

        # you actually need both to plot the error
        if "x" in samples.keys():
            my_PDE.x_other = x_mean
            try:
                y_ref = my_PDE.torch_reference_solve(true_solve=False).detach()
                u_ref = cuda_to_cpu(YToField.eval(y_ref))

                plot_abs_error_std(u_ref[0], u_mean[0], torch.tensor(1), s_grid, suptitle=r'calculated from MF with solver vs. Inferred $u_1$', results_path=result_path, r_error_instead_std=True)
                plot_abs_error_std(u_ref[1], u_mean[1], torch.tensor(1), s_grid, suptitle=r'calculated from MF with solver vs. Inferred $u_2$', results_path=result_path, r_error_instead_std=True)
            except: 
                warnings.warn("Plotting: Was not able to calculate u(x_mean) via torch solver.")
    else:
        warnings.warn("Plotting: No 'y' in the samples for plotting.")

    # Sqres
    plot_line_w_runnning_avg([np.abs(my_log.Sqres)],
                             [r'E[$r^2$]'],
                             running_avg_range=None,
                             results_path=result_path)

    # MF_mean_fenics = cuda_to_cpu(torch.exp(MF_mean[0]))
    # u_mean_fenics = solve_hook_pde_fenics(MF_mean_fenics, cuda_to_cpu(XToField.list[1].c),  cuda_to_cpu(my_PDE.args_rhs["f_load"]), plotSol=True)
    # u_mean_fenics = u_mean_fenics.transpose(dim0=-2, dim1=-1)

    # plot theta
    try:
        plot_multiple_fields(ThetaToField.eval(torch.tensor(my_log.theta[-1])), s_grid, results_path=result_path, titles="theta")
    except:
        warnings.warn("Plotting: Was not able to plot theta.")
