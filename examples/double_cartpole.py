# coding: utf-8

from __future__ import print_function

import six
import torch
import warnings
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import pddp
import pddp.examples

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

torch.set_flush_denormal(True)

N = 50  # Horizon length.
DT = 0.05  # Time step (s).
PLOT = True  # Whether to plot or not.
RENDER = True  # Whether to render the environment or not.
ENCODING = pddp.StateEncoding.DEFAULT


def plot_loss(J_hist):
    if not PLOT:
        return

    plt.plot(J_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Total loss")
    plt.title("Loss path")

    plt.show()


def plot_path(Z, encoding=ENCODING, indices=None, std_scale=1.0, legend=True):
    if not PLOT:
        return

    mean_ = pddp.utils.encoding.decode_mean(Z, encoding)
    std_ = pddp.utils.encoding.decode_std(Z, encoding)

    labels = [
        "Position (m)",
        "Velocity (m/s)",
        "Link 1 orientation (rad)",
        "Link 1 angular velocity (rad/s)",
        "Link 2 orientation (rad)",
        "Link 2 angular velocity (rad/s)",
    ]

    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    if indices is None:
        indices = list(range(mean_.shape[-1]))

    t = torch.arange(Z.shape[0]).detach().numpy()
    for index in indices:
        mean = mean_[:, index].detach().numpy()
        std = std_[:, index].detach().numpy()

        plt.plot(t, mean, label=labels[index], color=colors[index])

        for i in range(1, 4):
            j = std_scale * i
            plt.gca().fill_between(
                t.flat, (mean - j * std).flat, (mean + j * std).flat,
                color=colors[index],
                alpha=1.0 / (i + 1))

    if legend:
        plt.legend(
            bbox_to_anchor=(0.0, 1.0, 1.0, 2.0),
            loc="upper left",
            ncol=3,
            mode="expand",
            borderaxespad=0.)

    plt.xlim(0, N)
    plt.axhline(0, linestyle="--", color="#333333", linewidth=0.25)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    J_hist = []

    if PLOT:
        plt.figure(figsize=(10, 9))
        plt.ion()
        plt.show()

    def on_trial(trial, X, U):
        plt.subplot(7, 1, 1)
        plt.cla()
        plt.title("Trial {}".format(trial + 1))
        plot_path(X, encoding=pddp.StateEncoding.IGNORE_UNCERTAINTY)

    def on_iteration(iteration, Z, U, J_opt, accepted, converged):
        J_hist.append(J_opt.detach().numpy())
        if iteration % 10 == 9 or iteration == 0:
            for i in range(model.state_size):
                plt.subplot(7, 1, i + 2)
                plt.cla()
                if i == 0:
                    plt.title("Iteration {}".format(iteration + 1))
                if i == model.state_size:
                    plt.xlabel("Time step")
                plot_path(Z, indices=[i], legend=False)

    cost = pddp.examples.double_cartpole.DoubleCartpoleCost()
    env = pddp.examples.double_cartpole.DoubleCartpoleEnv(dt=DT, render=RENDER)
    model_class = pddp.examples.double_cartpole.DoubleCartpoleDynamicsModel

    model = pddp.models.bnn.bnn_dynamics_model_factory(
        env.state_size,
        env.action_size,
        [200, 200],
        model_class.angular_indices,
        model_class.non_angular_indices,
    )(n_particles=100)

    U = torch.randn(N, model.action_size)
    controller = pddp.controllers.PDDPController(
        env,
        model,
        cost,
        model_opts={
            "use_predicted_std": False,
            "infer_noise_variables": True,
        },
        training_opts={
            "n_iter": 1000,
            "learning_rate": 1e-3,
        },
    )

    controller.train()
    Z, U, K = controller.fit(
        U,
        encoding=ENCODING,
        n_iterations=200,
        max_var=0.4,
        tol=0,
        on_iteration=on_iteration,
        on_trial=on_trial,
        max_J=0,
        max_trials=20,
        start_from_bestU=True)

    plt.figure()
    plot_loss(J_hist)

    if RENDER:
        # Wait for user interaction before trial.
        _ = six.moves.input("Press ENTER to run")

    Z_ = torch.empty_like(Z)
    Z_[0] = env.get_state().encode(ENCODING)
    for i, u in enumerate(U):
        dz = Z_[i] - Z[i]
        u = u + K[i].matmul(dz)
        env.apply(u)
        Z_[i + 1] = env.get_state().encode(ENCODING)

    # Wait for user interaction to close everything.
    _ = six.moves.input("Press ENTER to exit")

    env.close()
