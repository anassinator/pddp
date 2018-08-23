"""Unit tests for pddp.models.bnn."""

import torch
import pytest

from pddp.models.bnn import *
from pddp.utils.gaussian_variable import GaussianVariable
from pddp.utils.encoding import StateEncoding

N = 20
STATE_SIZE = 4
ACTION_SIZE = 2

STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_forward(encoding):
    model = bnn_dynamics_model_factory(STATE_SIZE, ACTION_SIZE, [10, 10])()

    X = [GaussianVariable.random(STATE_SIZE) for _ in range(N)]
    Z = torch.stack([x.encode(encoding) for x in X])
    U = torch.randn(N, ACTION_SIZE)

    Z_ = model(Z, U, 0, encoding)

    assert Z_.shape == Z.shape
    for z in Z_:
        for cost in z:
            torch.autograd.grad(cost, Z, retain_graph=True)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_forward(benchmark, encoding):
    model = bnn_dynamics_model_factory(STATE_SIZE, ACTION_SIZE, [10, 10])()

    X = [GaussianVariable.random(STATE_SIZE) for _ in range(N)]
    Z = torch.stack([x.encode(encoding) for x in X])
    U = torch.randn(N, ACTION_SIZE)

    benchmark(model, Z, U, 0, encoding)
