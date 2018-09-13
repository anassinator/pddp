import pddp
import torch
import pddp.examples

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

DT = 0.1  # Time step (s).
RENDER = False  # Whether to render the environment or not.
N = 25  # Horizon length.
ITERATIONS = 100
U_MAX = torch.tensor([10.0])
U_MIN = -U_MAX

# Disable uncertainty since we are dealing with known dynamics.
ENCODING = pddp.StateEncoding.IGNORE_UNCERTAINTY

cost = pddp.examples.cartpole.CartpoleCost()
model = pddp.examples.cartpole.CartpoleDynamicsModel(DT)
env = pddp.examples.cartpole.CartpoleEnv(dt=DT, render=RENDER)

controller = pddp.controllers.iLQRController(env, model, cost)
U = 1e-1 * torch.randn(N, model.action_size)

z0 = env.get_state().encode(ENCODING)
Zs = torch.empty(ITERATIONS + 1, N + 1, z0.shape[0])
Zs[0] = pddp.controllers.ilqr.forward(z0, U, model, cost, ENCODING, True)[0]


def on_iteration(iteration, state, Z, U, J_opt):
    global Zs
    Zs[iteration + 1] = Z.detach()
    if state.is_terminal():
        Zs = Zs[:iteration + 2]


controller.fit(
    U,
    encoding=ENCODING,
    n_iterations=ITERATIONS,
    on_iteration=on_iteration,
    tol=0,
    u_min=U_MIN,
    u_max=U_MAX)


def update(iteration):
    Z = Zs[iteration]

    t = np.arange(N + 1) * DT
    X = pddp.utils.encoding.decode_mean(Z, ENCODING).detach().numpy()

    x = X[:, 0]
    x_dot = X[:, 1]
    theta = np.unwrap(X[:, 2])  # Makes for smoother plots.
    theta_dot = X[:, 3]

    ax.clear()
    # ax.scatter(-2 * np.pi, 0, marker='*', color='r')
    ax.scatter(-np.pi, 0, marker='*', color='r')
    ax.scatter(np.pi, 0, marker='*', color='r')
    ax.plot(theta, theta_dot)
    ax.set_xlim(-3 * np.pi, 3 * np.pi)
    ax.set_ylim(-4 * np.pi, 4 * np.pi)
    ax.set_xlabel("Orientation (rad)")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.set_title("Iteration {}".format(iteration))

    return ax


if __name__ == "__main__":
    anim = FuncAnimation(
        fig, update, frames=np.arange(0, Zs.shape[0]), interval=1000)
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        anim.save("ilqr.gif", dpi=fig.get_dpi(), writer="imagemagick")
    else:
        # Just loop the animation forever.
        plt.show()