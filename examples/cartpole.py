# coding: utf-8

from __future__ import print_function

import time
import torch
import numpy as np
import matplotlib.pyplot as plt

import pddp
import pddp.examples

torch.set_flush_denormal(True)

N = 25  # Horizon length.
DT = 0.1  # Time step (s).
RENDER = True  # Whether to render the environment or not.
ENCODING = pddp.StateEncoding.STANDARD_DEVIATION_ONLY


def plot_loss(J_hist):
    plt.plot(J_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Total loss")
    plt.title("Loss path")

    plt.show()


def plot_phase(X):
    theta = np.unwrap(X[:, 2])  # Makes for smoother plots.
    theta_dot = X[:, 3]

    plt.xlim(-3 * np.pi, 3 * np.pi)
    plt.ylim(-3 * np.pi, 3 * np.pi)

    plt.plot(theta, theta_dot)
    plt.xlabel("Orientation (rad)")
    plt.ylabel("Angular velocity (rad/s)")

    plt.draw()
    plt.pause(0.001)


def plot_path(Z, encoding=ENCODING, indices=None, std_scale=1.0, legend=True):
    mean_ = pddp.utils.encoding.decode_mean(Z, encoding)
    std_ = pddp.utils.encoding.decode_std(Z, encoding)

    labels = [
        "Position (m)",
        "Velocity (m/s)",
        "Orientation (rad)",
        "Angular velocity (rad/s)",
    ]

    if indices is None:
        indices = list(range(mean_.shape[-1]))

    t = torch.arange(Z.shape[0]).detach().numpy()
    for index in indices:
        mean = mean_[:, index].detach().numpy()
        std = std_[:, index].detach().numpy()

        plt.plot(t, mean, label=labels[index])

        for i in range(1, 4):
            j = std_scale * i
            plt.gca().fill_between(
                t.flat, (mean - j * std).flat, (mean + j * std).flat,
                color="#dddddd",
                alpha=1.0 / i)

    if legend:
        plt.legend()

    plt.ylabel("State")
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    J_hist = []

    plt.figure(figsize=(12, 6))
    plt.ion()
    plt.show()

    def on_trial(trial, X, U):
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.title("Trial {}".format(trial + 1))
        plot_path(X, encoding=pddp.StateEncoding.IGNORE_UNCERTAINTY)

    def on_iteration(iteration, Z, U, J_opt, accepted, converged):
        J_hist.append(J_opt.detach().numpy())

        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("Iteration {}".format(iteration + 1))
        plt.xlabel("Time steps")
        plot_path(Z, legend=False)

    cost = pddp.examples.cartpole.CartpoleCost()
    env = pddp.examples.cartpole.CartpoleEnv(dt=DT, render=RENDER)
    model_class = pddp.examples.cartpole.CartpoleDynamicsModel

    model = pddp.models.bnn.bnn_dynamics_model_factory(
        env.state_size,
        env.action_size,
        [200, 200],
        model_class.angular_indices,
        model_class.non_angular_indices,
    )(n_particles=100)

    controller = pddp.controllers.PDDPController(
        env,
        model,
        cost,
        training_opts={"n_iter": 1000},
    )
    U = torch.randn(N, model.action_size)

    controller.train()
    Z, U = controller.fit(
        U,
        encoding=ENCODING,
        n_iterations=50,
        max_var=0.4,
        on_iteration=on_iteration,
        on_trial=on_trial,
        max_J=0,
        max_trials=20,
    )

    plt.figure()
    plot_loss(J_hist)

    if RENDER:
        # Wait for user interaction before trial.
        _ = input("Press ENTER to run")

    Z_ = torch.empty_like(Z)
    Z_[0] = env.get_state().encode(ENCODING)
    for i, u in enumerate(U):
        env.apply(u)
        Z_[i + 1] = env.get_state().encode(ENCODING)
        time.sleep(DT)


    # Wait for user interaction to close everything.
    _ = input("Press ENTER to exit")

    env.close()