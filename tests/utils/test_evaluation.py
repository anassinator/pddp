"""Unit tests for pddp.utils.evaluation."""

import torch
import pytest

from pddp.examples import *
from pddp.costs import QRCost
from pddp.utils.evaluation import *
from pddp.utils.gaussian_variable import GaussianVariable

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

    N = model.state_size
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

    (batch_l, batch_l_z, batch_l_u,
     batch_l_zz, batch_l_uz, batch_l_uu) = batch_eval_cost(
         cost, z, u, 0, terminal, encoding, approximate)

    assert batch_l.shape == torch.Size([])
    assert batch_l_z.shape == z.shape
    assert batch_l_zz.shape == (z.shape[0], z.shape[0])

    assert batch_l.allclose(l, 1e-3, 1e-3)
    assert batch_l_z.allclose(l_z, 1e-3, 1e-3)
    assert batch_l_zz.allclose(l_zz, 1e-3, 1e-3)

    if terminal:
        assert batch_l_u is None
        assert batch_l_uz is None
        assert batch_l_uu is None
    else:
        assert batch_l_u.shape == u.shape
        assert batch_l_uz.shape == (u.shape[0], z.shape[0])
        assert batch_l_uu.shape == (u.shape[0], u.shape[0])

        assert batch_l_u.allclose(l_u, 1e-3, 1e-3)
        assert batch_l_uz.allclose(l_uz, 1e-3, 1e-3)
        assert batch_l_uu.allclose(l_uu, 1e-3, 1e-3)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("model_class", MODELS)
def test_eval_dynamics(model_class, encoding):
    model = model_class(0.1)

    x = GaussianVariable.random(model.state_size)
    u = torch.randn(model.action_size, requires_grad=True)

    z = x.encode(encoding)
    z_next, d_dz, d_du = eval_dynamics(model, z, u, 0, encoding)

    assert z_next.shape == z.shape
    assert d_dz.shape == (z.shape[0], z.shape[0])
    assert d_du.shape == (z.shape[0], u.shape[0])

    batch_z_next, batch_d_dz, batch_d_du = batch_eval_dynamics(
        model, z, u, 0, encoding)

    assert batch_z_next.shape == z.shape
    assert batch_d_dz.shape == (z.shape[0], z.shape[0])
    assert batch_d_du.shape == (z.shape[0], u.shape[0])

    assert batch_z_next.allclose(z_next, 1e-3, 1e-3)
    assert batch_d_dz.allclose(d_dz, 1e-3, 1e-3)
    assert batch_d_du.allclose(d_du, 1e-3, 1e-3)
