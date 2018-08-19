"""Unit tests for pddp.controllers.utils.autodiff."""

import torch

from pddp.controllers.utils.autodiff import *


def test_grad():
    x = torch.ones(5, requires_grad=True)
    y = x.sum()
    dy_dx = grad(y, x)
    assert dy_dx.shape == x.shape
    assert (dy_dx == x).all()


def test_jacobian():

    def f(x):
        return torch.stack(
            [
                x.sum(dim=-1),
                (x**2).sum(dim=-1),
                (x**3).sum(dim=-1),
            ], dim=-1)

    x = 2 * torch.ones(5, requires_grad=True)
    y = f(x)
    J = jacobian(y, x)

    assert J.shape == (y.shape[0], x.shape[0])
    assert (J[0] == torch.ones_like(x)).all()
    assert (J[1] == 2 * x).all()
    assert (J[2] == 3 * x**2).all()

    J_ = batch_jacobian(f, x)
    assert (J == J_).all()
