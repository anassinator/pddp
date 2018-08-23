"""Unit tests for pddp.examples.*.cost."""

import torch
import pytest

from pddp.examples import *
from pddp.utils.gaussian_variable import GaussianVariable
from pddp.utils.encoding import StateEncoding
from pddp.utils.angular import infer_augmented_state_size

STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]

MODELS = [
    cartpole.CartpoleDynamicsModel,
    pendulum.PendulumDynamicsModel,
    rendezvous.RendezvousDynamicsModel,
]

COSTS = [
    cartpole.CartpoleCost,
    pendulum.PendulumCost,
    rendezvous.RendezvousCost,
]


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class, cost_class", zip(MODELS, COSTS))
def test_forward(model_class, cost_class, encoding, terminal):
    cost = cost_class()
    model = model_class(0.1)

    augmented_state_size = infer_augmented_state_size(model.angular_indices,
                                                      model.non_angular_indices)
    z = GaussianVariable.random(augmented_state_size).encode(encoding)
    u = torch.randn(model.action_size, requires_grad=True)

    l = cost(z, u, 0, terminal=terminal, encoding=encoding)

    assert l.dim() == 0
    torch.autograd.grad(l, z, retain_graph=True)

    if not terminal:
        torch.autograd.grad(l, u, retain_graph=True)
