"""Unit tests for pddp.models.utils.trajectory."""

import torch

from pddp.models import GaussianVariable
from pddp.models.utils.trajectory import *


def test_mean_trajectory():
    N = 100
    state_size = 5

    X = [GaussianVariable.random(state_size) for i in range(N)]

    X_ = mean_trajectory(X)
    assert X_.shape == (N, state_size)
    for x, mean in zip(X, X_):
        assert (mean == x.mean()).all()


def test_sample_trajectory():
    N = 100
    state_size = 5

    X = [GaussianVariable.random(state_size) for i in range(N)]

    X_ = sample_trajectory(X)
    assert X_.shape == (N, state_size)


def test_trajectory_to_training_data():
    N = 100
    state_size = 5
    action_size = 2

    X = torch.randn(N + 1, state_size)
    U = torch.randn(N, action_size)

    X_, dX = trajectory_to_training_data(X, U)
    assert X_.shape == (N, state_size + action_size)
    assert dX.shape == (N, state_size)