# coding: utf-8

from __future__ import print_function

import six
import torch
import numpy as np
import matplotlib.pyplot as plt

import pddp
from pddp.examples import SampleProblems

from utils import plot_pause, rollout

torch.set_flush_denormal(True)

N = 25  # Horizon length.
DT = 0.1  # Time step (s).
PLOT = True  # Whether to plot or not.
RENDER = True  # Whether to render the environment or not.
ENCODING = pddp.StateEncoding.DEFAULT
PROBLEM = SampleProblems.CARTPOLE

UMAX = {
    SampleProblems.CARTPOLE: torch.tensor(10.0),
    SampleProblems.DOUBLE_CARTPOLE: torch.tensor(20.0),
    SampleProblems.PENDULUM: torch.tensor(2.5),
    SampleProblems.RENDEZVOUS: None,
}[PROBLEM]
UMIN = -UMAX if UMAX is not None else None

LABELS = {
    SampleProblems.CARTPOLE: [
        "Position (m)",
        "Velocity (m/s)",
        "Orientation (rad)",
        "Angular velocity (rad/s)",
    ],
    SampleProblems.DOUBLE_CARTPOLE: [
        "Position (m)",
        "Velocity (m/s)",
        "Link 1 orientation (rad)",
        "Link 1 angular velocity (rad/s)",
        "Link 2 orientation (rad)",
        "Link 2 angular velocity (rad/s)",
    ],
    SampleProblems.PENDULUM: [
        "Orientation (rad)",
        "Angular velocity (rad/s)",
    ],
    SampleProblems.RENDEZVOUS: [
        "Vehicle 1 x position (m)",
        "Vehicle 1 y position (m)",
        "Vehicle 2 x position (m)",
        "Vehicle 2 y position (m)",
        "Vehicle 1 x velocity (m/s)",
        "Vehicle 1 y velocity (m/s)",
        "Vehicle 2 x velocity (m/s)",
        "Vehicle 2 y velocity (m/s)",
    ],
}[PROBLEM]


def plot_loss(J_hist):
    if not PLOT:
        return

    plt.plot(J_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Total loss")
    plt.title("Loss path")

    plt.show(block=False)


def plot_path(Z,
              encoding=ENCODING,
              indices=None,
              reality=None,
              std_scale=1.0,
              legend=True):
    if not PLOT:
        return

    mean_ = pddp.utils.encoding.decode_mean(Z, encoding)
    std_ = pddp.utils.encoding.decode_std(Z, encoding)

    if reality is not None:
        real_mean = pddp.utils.encoding.decode_mean(reality, encoding)

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

    if indices is None:
        indices = list(range(mean_.shape[-1]))

    t = torch.arange(Z.shape[0]).detach().numpy()
    for index in indices:
        mean = mean_[:, index].detach().numpy()
        std = std_[:, index].detach().numpy()

        if reality is not None:
            y = real_mean[:, index].detach().numpy()
            plt.plot(t, y, color=colors[index], linestyle="dashed")

        plt.plot(t, mean, label=LABELS[index], color=colors[index])

        for i in range(1, 4):
            j = std_scale * i
            plt.gca().fill_between(
                t.flat, (mean - j * std).flat, (mean + j * std).flat,
                color=colors[index],
                alpha=1.0 / (i + 1))

    if legend:
        plt.legend(
            bbox_to_anchor=(0.0, 1.0, 1.0, 0.7),
            loc="upper left",
            ncol=4,
            mode="expand",
            borderaxespad=0.)

    plt.xlim(0, N)
    plt.axhline(0, linestyle="--", color="#333333", linewidth=0.25)

    plt.tight_layout()
    plt.draw()
    plot_pause(0.001)


if __name__ == "__main__":
    J_hist = []

    if PLOT:
        plt.figure(figsize=(10, 9))
        plt.ion()
        plt.show(block=False)

    def on_trial(trial, X, U):
        plt.subplot(model.state_size + 1, 1, 1)
        plt.cla()
        plt.title("Trial {}".format(trial + 1))
        plot_path(X, encoding=pddp.StateEncoding.IGNORE_UNCERTAINTY)

    def on_iteration(iteration, state, Z, U, J_opt):
        J_hist.append(J_opt.detach().numpy())
        if iteration % 10 == 9 or iteration == 0:
            real_Z = rollout(real_model, Z[0], U, ENCODING)
            for i in range(model.state_size):
                plt.subplot(model.state_size + 1, 1, i + 2)
                plt.cla()
                if i == 0:
                    plt.title("Iteration {}".format(iteration + 1))
                if i == model.state_size:
                    plt.xlabel("Time step")
                plot_path(Z, indices=[i], reality=real_Z, legend=False)

    env, cost, real_model = SampleProblems.setup(DT, render=RENDER)

    model = pddp.models.bnn.bnn_dynamics_model_factory(
        env.state_size,
        env.action_size,
        [200, 200],
        real_model.angular_indices,
        real_model.non_angular_indices,
    )(n_particles=100)

    U = (UMAX - UMIN) * torch.rand(N, model.action_size) + UMIN
    controller = pddp.controllers.PDDPController(
        env,
        model,
        cost,
        model_opts={
            "use_predicted_std": False,
            "infer_noise_variables": True
        },
        training_opts={
            "n_iter": 2000,
            "learning_rate": 1e-3,
        },
    )

    controller.train()
    Z, U, state = controller.fit(
        U,
        encoding=ENCODING,
        n_iterations=50,
        on_iteration=on_iteration,
        on_trial=on_trial,
        max_trials=20,
        u_min=UMIN,
        u_max=UMAX)

    plt.figure()
    plot_loss(J_hist)

    if RENDER:
        # Wait for user interaction before trial.
        _ = six.moves.input("Press ENTER to run")

    for i in range(N):
        z = env.get_state().encode(ENCODING)
        u = controller(z, i, ENCODING)
        env.apply(u)

    # Wait for user interaction to close everything.
    _ = six.moves.input("Press ENTER to exit")

    env.close()
