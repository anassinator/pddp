"""Unit tests for pddp.examples.*.model."""

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


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class", MODELS)
def test_forward(model_class, encoding):
    model = model_class(0.1)

    z = GaussianVariable.random(model.state_size).encode(encoding)
    u = torch.randn(model.action_size, requires_grad=True)

    z_ = model(z, u, 0, encoding)

    assert z.shape == z_.shape

    for wrt in (z, u):
        for cost in z_:
            torch.autograd.grad(cost, wrt, allow_unused=True, retain_graph=True)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class", MODELS)
def test_gradcheck(model_class, encoding):
    model = model_class(0.1).double()

    X = [GaussianVariable.random(model.state_size) for _ in range(3)]
    Z = torch.stack([x.encode(encoding) for x in X]).double()
    U = torch.randn(3, model.action_size).double()

    assert torch.autograd.gradcheck(model, (Z, U, 0, encoding))
    assert torch.autograd.gradcheck(model, (Z[0], U[0], 0, encoding))
