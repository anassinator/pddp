"""Unit tests for pddp.models.gaussian_variable."""

import torch
import pytest

from pddp.models.gaussian_variable import *


def _test_gaussian_variable(g):
    assert (g.var().isclose(g.covar().diag())).all()
    assert (g.std().isclose(g.covar().diag().sqrt())).all()
    assert (g.std().isclose(g.var().sqrt())).all()
    assert (g.var().isclose(g.std().pow(2))).all()


def _test_grad(g, wrt):
    for cost in (g.mean(), g.covar().view(-1), g.var(), g.std()):
        for c in cost:
            torch.autograd.grad(c, wrt, allow_unused=True, retain_graph=True)


def test_from_covar():
    N = 5
    mean = torch.randn(N, requires_grad=True)
    covar = torch.rand(N, N, requires_grad=True)
    g = GaussianVariable(mean, covar=covar)

    _test_gaussian_variable(g)
    _test_grad(g, g.mean())
    _test_grad(g, g.covar())


def test_from_var():
    N = 5
    mean = torch.randn(N, requires_grad=True)
    var = torch.rand(N, requires_grad=True)
    g = GaussianVariable(mean, var=var)

    _test_gaussian_variable(g)
    _test_grad(g, g.mean())
    _test_grad(g, g.covar())


def test_from_std():
    N = 5
    mean = torch.randn(N, requires_grad=True)
    std = torch.rand(N, requires_grad=True)
    g = GaussianVariable(mean, std=std)

    _test_gaussian_variable(g)
    _test_grad(g, g.mean())
    _test_grad(g, g.covar())
