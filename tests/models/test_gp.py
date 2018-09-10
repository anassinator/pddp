"""Unit tests for pddp.models.gp."""

import torch
import pytest

from pddp.models.gp import *
from pddp.utils.autodiff import jacobian
from pddp.utils.evaluation import eval_dynamics
from pddp.utils.gaussian_variable import GaussianVariable
from pddp.utils.encoding import StateEncoding, decode_mean

N = 10
STATE_SIZE = 4
ACTION_SIZE = 2

STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]


def _setup(encoding):
    model = gp_dynamics_model_factory(STATE_SIZE, ACTION_SIZE)()

    X_ = torch.randn(N, STATE_SIZE)
    U_ = torch.randn(N, ACTION_SIZE)
    dX = X_.sin() + 1e-2 * torch.randn_like(X_) - X_
    model.double().fit(X_, U_, dX, n_iter=10, quiet=True)

    X = [GaussianVariable.random(STATE_SIZE) for _ in range(N)]
    Z = torch.stack([x.encode(encoding) for x in X])
    U = torch.randn(N, ACTION_SIZE, requires_grad=True)
    return model, Z, U


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_forward(encoding):
    model, Z, U = _setup(encoding)
    Z_ = model(Z, U, 0, encoding)

    assert Z_.shape == Z.shape
    for z in Z_:
        for cost in z:
            torch.autograd.grad(cost, Z, retain_graph=True)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_gradcheck(encoding):
    model, Z, U = _setup(encoding)

    model.double()
    Z = Z.double()
    U = U.double()

    z = Z[0]
    u = U[0]

    assert torch.autograd.gradcheck(model, (Z, U, 0, encoding))
    assert torch.autograd.gradcheck(model, (z, u, 0, encoding))

    # Verify that batch jacobians are correct.
    z_next = model(z, u, 0, encoding)
    F_z = jacobian(z_next, z)
    F_u = jacobian(z_next, u)
    z_next_, F_z_, F_u_ = eval_dynamics(model, z, u, 0, encoding)

    assert z_next.allclose(z_next_)
    assert F_z.allclose(F_z_, 1e-3)
    assert F_u.allclose(F_u_, 1e-3)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_forward(benchmark, encoding):
    model, Z, U = _setup(encoding)
    benchmark(model, Z, U, 0, encoding)
