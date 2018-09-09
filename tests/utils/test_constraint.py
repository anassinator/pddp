"""Unit tests for pddp.utils.constraint."""

import torch
import pytest

from pddp.utils.constraint import *

N = 100
BOUNDS = [
    (-1.0, 1.0),
    (-torch.ones(N), torch.ones(N)),
    (-torch.rand(N), torch.rand(N)),
]


@pytest.mark.parametrize("min_bounds, max_bounds", BOUNDS)
def test_constrain(min_bounds, max_bounds):
    u = 10 * torch.randn(N, requires_grad=True)
    u_ = constrain(u, min_bounds, max_bounds)
    assert u_.shape == u.shape
    assert (u_ >= (min_bounds - 1e-6)).all()
    assert (u_ <= (max_bounds + 1e-6)).all()

    # Verify differentiable.
    for i in range(N):
        torch.autograd.grad(u_[i], u, retain_graph=True)


@pytest.mark.parametrize("min_bounds, max_bounds", BOUNDS)
def test_boxqp(min_bounds, max_bounds):
    min_bounds = torch.tensor(min_bounds)
    if min_bounds.dim() < 1:
        min_bounds = min_bounds.unsqueeze(0)
    max_bounds = torch.tensor(min_bounds)
    if max_bounds.dim() < 1:
        max_bounds = max_bounds.unsqueeze(0)

    u0 = torch.tensor((max_bounds + min_bounds) / 2.0)
    Q = torch.randn(u0.shape[0], u0.shape[0])
    Q = Q.t().mm(Q)
    c = torch.randn(u0.shape[0])
    u, result, Ufree, free = boxqp(u0, Q, c, min_bounds, max_bounds)
    assert u.shape == u.shape
    assert (u >= (min_bounds - 1e-6)).all()
    assert (u <= (max_bounds + 1e-6)).all()
    assert result != 0
