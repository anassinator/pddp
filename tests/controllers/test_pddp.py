"""Unit tests for pddp.controllers.pddp."""

import torch
import pytest
import numpy as np

from pddp.examples import *
from pddp.costs import QRCost
from pddp.controllers.pddp import *
from pddp.utils.gaussian_variable import GaussianVariable

STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]

COSTS = [
    cartpole.CartpoleCost,
    pendulum.PendulumCost,
    rendezvous.RendezvousCost,
    double_cartpole.DoubleCartpoleCost,
]
ENVS = [
    cartpole.CartpoleEnv,
    pendulum.PendulumEnv,
    rendezvous.RendezvousEnv,
    double_cartpole.DoubleCartpoleEnv,
]
MODELS = [
    cartpole.CartpoleDynamicsModel,
    pendulum.PendulumDynamicsModel,
    rendezvous.RendezvousDynamicsModel,
    double_cartpole.DoubleCartpoleDynamicsModel,
]


def _setup(model_class, cost_class, encoding, N):
    model = model_class(0.1)
    cost = cost_class()

    z0 = GaussianVariable.random(model.state_size).encode(encoding)
    U = torch.randn(N, model.action_size, requires_grad=True)

    return z0, U, model, cost


@pytest.mark.filterwarnings("ignore:exceeded max regularization term")
@pytest.mark.parametrize("N", [1, 3])
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("env_class, model_class, cost_class",
                         zip(ENVS, MODELS, COSTS))
def test_fit(env_class, model_class, cost_class, encoding, N):
    env = env_class()
    _, U, model, cost = _setup(model_class, cost_class, encoding, N)
    controller = PDDPController(env, model, cost)

    controller.eval()
    Z, U, K = controller.fit(U, encoding=encoding)

    controller.train()
    Z, U, K = controller.fit(U, encoding=encoding, max_var=np.inf)