"""Unit tests for pddp.costs.base.AggregateCost."""

import torch
import pytest
import operator

from pddp.utils.gaussian_variable import GaussianVariable
from pddp.costs import AggregateCost, Cost, QRCost
from pddp.utils.encoding import StateEncoding

N = 5
M = 2

STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]
BINARY_OPS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.pow,
]
UNARY_OPS = [operator.neg]

COSTS = [
    QRCost(torch.eye(N), torch.eye(M)),
    QRCost(torch.rand(N, N), torch.rand(M, M)),
    -QRCost(torch.rand(N, N), torch.eye(M)),
]
NON_COSTS = [torch.tensor(-1.0), torch.tensor(2.0)]


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("cost2", COSTS + NON_COSTS)
@pytest.mark.parametrize("cost1", COSTS)
@pytest.mark.parametrize("op", BINARY_OPS)
def test_binary_op(op, cost1, cost2, terminal, encoding):
    x = GaussianVariable.random(N)
    z = x.encode(encoding)
    u = torch.randn(M, requires_grad=True) if not terminal else None

    cost = op(cost1, cost2)
    l = cost(z, u, 0, terminal, encoding)
    l1 = cost1(z, u, 0, terminal, encoding)
    if isinstance(cost2, Cost):
        l2 = cost2(z, u, 0, terminal, encoding)
    else:
        l2 = cost2

    assert l.shape == torch.Size([])
    if torch.isnan(l):
        assert torch.isnan(op(l1, l2))
    else:
        assert l.isclose(op(l1, l2), 1e-3)

    torch.autograd.grad(l, z, retain_graph=True)

    if not terminal:
        torch.autograd.grad(l, u, retain_graph=True)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("cost", COSTS)
@pytest.mark.parametrize("op", UNARY_OPS)
def test_unary_op(op, cost, terminal, encoding):
    x = GaussianVariable.random(N)
    z = x.encode(encoding)
    u = torch.randn(M)
    u = torch.randn(M, requires_grad=True) if not terminal else None

    cost_ = op(cost)
    l = cost(z, u, 0, terminal, encoding)
    l_ = cost_(z, u, 0, terminal, encoding)

    assert l.shape == torch.Size([])
    assert l_.isclose(op(l), 1e-3)

    torch.autograd.grad(l, z, retain_graph=True)

    if not terminal:
        torch.autograd.grad(l, u, allow_unused=True, retain_graph=True)
