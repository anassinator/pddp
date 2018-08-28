"""Unit tests for pddp.costs.quadratic."""

import torch
import pytest

from pddp.costs.quadratic import *
from pddp.utils.encoding import StateEncoding
from pddp.utils.autodiff import grad, jacobian
from pddp.utils.gaussian_variable import GaussianVariable

N = 50
M = 20

STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_qrcost(encoding, terminal):
    Q = torch.randn(N, N)
    R = torch.randn(M, M)
    Q_terminal = torch.randn(N, N)
    x_goal = torch.randn(N)
    u_goal = torch.randn(M)
    cost = QRCost(Q, R, Q_terminal, x_goal, u_goal)

    x = GaussianVariable.random(N)
    z = x.encode(encoding)
    u = torch.randn(M, requires_grad=True) if not terminal else None

    l = cost(z, u, 0, terminal, encoding)

    assert l.shape == torch.Size([])

    l_x = grad(l, x.mean(), create_graph=True)
    l_xx = jacobian(l_x, x.mean())

    Q_ = Q_terminal + Q_terminal.t() if terminal else Q + Q.t()
    assert l_xx.isclose(Q_, 1e-3).all()

    if not terminal:
        l_u = grad(l, u, create_graph=True)
        l_ux = jacobian(l_u, x.mean())
        l_uu = jacobian(l_u, u)

        assert l_uu.isclose(R + R.t(), 1e-3).all()

        # Test batch.
        Z = z.repeat(N, 1)
        U = u.repeat(N, 1)
        L = cost(Z, U, 0, terminal, encoding)


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_gradcheck(encoding, terminal):
    # Smaller dimensionality for speed.
    state_size = 5
    action_size = 2
    batch_size = 3

    Q = torch.randn(state_size, state_size)
    R = torch.randn(action_size, action_size)
    Q_terminal = torch.randn(state_size, state_size)
    x_goal = torch.randn(state_size)
    u_goal = torch.randn(action_size)
    cost = QRCost(Q, R, Q_terminal, x_goal, u_goal).double()

    X = [GaussianVariable.random(state_size) for _ in range(batch_size)]
    Z = torch.stack([x.encode(encoding) for x in X]).double()
    U = torch.randn(batch_size, action_size).double()

    assert torch.autograd.gradcheck(cost, (Z, U, 0, terminal, encoding))
    assert torch.autograd.gradgradcheck(cost, (Z, U, 0, terminal, encoding))

    assert torch.autograd.gradcheck(cost, (Z[0], U[0], 0, terminal, encoding))
    assert torch.autograd.gradgradcheck(cost,
                                        (Z[0], U[0], 0, terminal, encoding))


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_qrcost(benchmark, encoding, terminal):
    Q = torch.randn(N, N)
    R = torch.randn(M, M)
    Q_terminal = torch.randn(N, N)
    x_goal = torch.randn(N)
    u_goal = torch.randn(M)
    cost = QRCost(Q, R, Q_terminal, x_goal, u_goal)

    Z = torch.stack(
        [GaussianVariable.random(N).encode(encoding) for _ in range(10)])
    U = torch.randn(10, M)
    I = torch.arange(10)

    benchmark(cost, Z, U, I, terminal=terminal, encoding=encoding)
