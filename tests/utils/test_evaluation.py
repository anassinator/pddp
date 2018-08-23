"""Unit tests for pddp.utils.evaluation."""

import torch
import pytest

from pddp.examples import *
from pddp.costs import QRCost
from pddp.utils.evaluation import *
from pddp.utils.gaussian_variable import GaussianVariable
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


@pytest.mark.parametrize("approximate", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class, cost_class", zip(MODELS, COSTS))
def test_eval_cost(model_class, cost_class, encoding, terminal, approximate):
    model = model_class(0.1)

    N = infer_augmented_state_size(model.angular_indices,
                                   model.non_angular_indices)
    M = model.action_size

    z = GaussianVariable.random(N).encode(encoding)
    u = torch.randn(M, requires_grad=True) if not terminal else None

    cost = cost_class()

    l, l_z, l_u, l_zz, l_uz, l_uu = eval_cost(cost, z, u, 0, terminal, encoding,
                                              approximate)

    assert l.shape == torch.Size([])
    assert l_z.shape == z.shape
    assert l_zz.shape == (z.shape[0], z.shape[0])

    if terminal:
        assert l_u is None
        assert l_uz is None
        assert l_uu is None
    else:
        assert l_u.shape == u.shape
        assert l_uz.shape == (u.shape[0], z.shape[0])
        assert l_uu.shape == (u.shape[0], u.shape[0])


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class", MODELS)
def test_eval_dynamics(model_class, encoding):
    model = model_class(0.1)
    augmented_state_size = infer_augmented_state_size(model.angular_indices,
                                                      model.non_angular_indices)
    x = GaussianVariable.random(augmented_state_size)
    u = torch.randn(model.action_size, requires_grad=True)

    z = x.encode(encoding)
    z_next, d_dz, d_du = eval_dynamics(model, z, u, 0, encoding)

    assert z_next.shape == z.shape
    assert d_dz.shape == (z.shape[0], z.shape[0])
    assert d_du.shape == (z.shape[0], u.shape[0])
