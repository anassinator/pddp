"""Unit tests for pddp.examples.*.model."""

import torch
import pytest

from pddp.examples import *
from pddp.models import GaussianVariable
from pddp.models.utils.encoding import StateEncoding
from pddp.models.utils.angular import infer_augmented_state_size

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


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class", MODELS)
def test_forward(model_class, encoding):
    model = model_class(0.1)

    augmented_state_size = infer_augmented_state_size(model.angular_indices,
                                                      model.non_angular_indices)
    x = GaussianVariable.random(augmented_state_size)
    u = torch.randn(model.action_size, requires_grad=True)

    z = x.encode(encoding)
    z_ = model(z, u, 0, encoding)

    assert z.shape == z_.shape

    for wrt in (z, u):
        for cost in z_:
            torch.autograd.grad(cost, wrt, allow_unused=True, retain_graph=True)
