"""Unit tests for pddp.examples.*.cost."""

import torch
import pytest

from pddp.examples import *
from pddp.utils.gaussian_variable import GaussianVariable
from pddp.utils.encoding import StateEncoding

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
    double_cartpole.DoubleCartpoleDynamicsModel,
]

COSTS = [
    cartpole.CartpoleCost,
    pendulum.PendulumCost,
    rendezvous.RendezvousCost,
    double_cartpole.DoubleCartpoleCost,
]


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class, cost_class", zip(MODELS, COSTS))
def test_forward(model_class, cost_class, encoding, terminal):
    cost = cost_class()

    z = GaussianVariable.random(model_class.state_size).encode(encoding)
    u = torch.randn(model_class.action_size, requires_grad=True)

    l = cost(z, u, 0, terminal=terminal, encoding=encoding)

    assert l.dim() == 0
    torch.autograd.grad(l, z, retain_graph=True)

    if not terminal:
        torch.autograd.grad(l, u, retain_graph=True)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class, cost_class", zip(MODELS, COSTS))
def test_gradcheck(model_class, cost_class, encoding, terminal):
    cost = cost_class().double()

    X = [GaussianVariable.random(model_class.state_size) for _ in range(3)]
    Z = torch.stack([x.encode(encoding) for x in X]).double()
    U = torch.randn(3, model_class.action_size).double()

    assert torch.autograd.gradcheck(cost, (Z, U, 0, terminal, encoding))
    assert torch.autograd.gradgradcheck(cost, (Z, U, 0, terminal, encoding))

    assert torch.autograd.gradcheck(cost, (Z[0], U[0], 0, terminal, encoding))
    assert torch.autograd.gradgradcheck(cost,
                                        (Z[0], U[0], 0, terminal, encoding))