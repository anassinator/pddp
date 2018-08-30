"""Unit tests for pddp.controllers.pddp."""

import torch
import pytest

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
]
ENVS = [
    cartpole.CartpoleEnv,
    pendulum.PendulumEnv,
    rendezvous.RendezvousEnv,
]
MODELS = [
    cartpole.CartpoleDynamicsModel,
    pendulum.PendulumDynamicsModel,
    rendezvous.RendezvousDynamicsModel,
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

    Z, U = controller.fit(U, encoding=encoding)