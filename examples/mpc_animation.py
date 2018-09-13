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
ITERATIONS = 50
U_MAX = torch.tensor([10.0])
U_MIN = -U_MAX

# Disable uncertainty since we are dealing with known dynamics.
ENCODING = pddp.StateEncoding.IGNORE_UNCERTAINTY

cost = pddp.examples.cartpole.CartpoleCost()
model = pddp.examples.cartpole.CartpoleDynamicsModel(DT)
env = pddp.examples.cartpole.CartpoleEnv(dt=DT, render=RENDER)

controller = pddp.controllers.iLQRController(env, model, cost)
U = 1e-1 * torch.randn(N, model.action_size)

controller.fit(
    U, encoding=ENCODING, n_iterations=1, tol=0, u_min=U_MIN, u_max=U_MAX)


def update(iteration):
    if iteration == 0:
        env.reset()

    z0 = env.get_state().encode(ENCODING)
    u = controller(z0, iteration, ENCODING, mpc=True, u_min=U_MIN, u_max=U_MAX)
    env.apply(u)

    t = np.arange(N + 1) * DT
    Z = controller._Z_nominal.detach()
    X = pddp.utils.encoding.decode_mean(Z, ENCODING).numpy()

    x = X[:, 0]
    x_dot = X[:, 1]
    theta = np.unwrap(X[:, 2])  # Makes for smoother plots.
    theta_dot = X[:, 3]

    ax.clear()
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
        fig, update, frames=np.arange(0, ITERATIONS), interval=100)
    if len(sys.argv) > 1 and sys.argv[1] == "save":
        anim.save("mpc.gif", dpi=fig.get_dpi(), writer="imagemagick")
    else:
        # Just loop the animation forever.
        plt.show()