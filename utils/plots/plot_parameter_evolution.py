import matplotlib.pyplot as plt
import os
import numpy as np


def plot_parameter_evolution(parameters, results_path=None, param_name="", plt_show=True):
    """
    (Author: ChatGPT, Vincent Scholz)
    Write a python function using matplotlib for a line plot of the evolution of a set of parameters.
    Each parameter should be depicted by one line with different color.
    All line plots shall be in one figure.
    The input parameter is two-dimensional numpy tensor, where each row corresponds to one parameter and each
    column represents one time step.
    :return:
    """

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Get the number of parameters and time steps
    num_parameters, num_time_steps = parameters.shape

        
    # Define a color map for the lines
    cmap = plt.get_cmap('tab10')

    # Plot each parameter as a separate line
    for i in range(num_parameters):
        # Get the parameter values for the current parameter
        parameter_values = parameters[i]

        # Generate a color for the line based on the parameter index
        color = cmap(i)

        # Plot the line for the current parameter
        if param_name != "":
            label = param_name + " " + str(i+1)
        else:
            label = f'Parameter {i + 1}'
        ax.plot(range(num_time_steps), parameter_values, color=color, label=label)

    # Set labels and title
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Parameter Value')
    ax.set_title('Evolution of Parameters ' + param_name)

    # Set x-axis to logarithmic scale
    # ax.set_xscale('log')

    # Add a legend
    if num_parameters <= 10:
        ax.legend(loc="upper right")

    # Display the plot
    if results_path is not None:
        plt.savefig(os.path.join(results_path, 'Evolution_of_Parameters_' + param_name + ".png"), dpi=200)
    plt.show()
    plt.close()
