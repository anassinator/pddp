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
