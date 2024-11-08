import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import torch
import os


def create_animation(x_data, y_data, w_data, poisson, results_path):
    """

    :param x_data: phi at respective values
    :param w_data: line 1: dim[len(x_data), i] for my theta
    :param save_path:
    :return: nice animation
    """
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], lw=2, color='blue', label='Approximation')
    line2, = ax.plot([], [], lw=2, color='red', label='True Value')

    len_w = np.size(w_data, 0)
    dim_w = np.size(w_data, 1)

    nodes_num = np.linspace(1, dim_w, dim_w)

    w_true = []
    for i in range(len_w):
        r_vec = poisson.all_residuals(x_data[i], y_data[i], "constrained")
        r_vec = torch.abs(r_vec / torch.linalg.norm(r_vec))
        w_true.append(np.asarray(r_vec))

    ax.set_xlim(0, dim_w)
    ax.set_ylim(np.min(np.concatenate([w_data, w_true])), np.max(np.concatenate([w_data, w_true])))
    ax.legend()

    def animate(i):
        line1.set_data(nodes_num, np.abs(w_data[i]))
        line2.set_data(nodes_num, w_true[i])
        return line1, line2

    ani = animation.FuncAnimation(fig, animate, frames=len_w, interval=20, blit=True)
    ani.save(os.path.join(results_path, "theta.mp4"))

    plt.show()


# x_data = np.linspace(0, 10, 100)
# w_data1 = np.sin(x_data)
# w_data2 = np.cos(x_data)
# create_animation(x_data, w_data1, w_data2, "animation.mp4")
