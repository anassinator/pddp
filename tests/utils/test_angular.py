"""Unit tests for pddp.utils.angular."""

import torch
import pytest
import numpy as np

from pddp.utils.angular import *

# Organized as angular_indices, non_angular_indices pairs.
INDICES = [
    (torch.tensor([]).long(), torch.tensor([0, 1, 2, 3, 4]).long()),
    (torch.tensor([2]).long(), torch.tensor([0, 1, 3, 4]).long()),
    (torch.tensor([1, 3]).long(), torch.tensor([0, 2, 4]).long()),
    (torch.tensor([0, 1, 2, 3, 4]).long(), torch.tensor([]).long()),
]
AUGMENTED_SIZES = [5, 6, 7, 10]


@pytest.mark.parametrize("indices, augmented_size", zip(INDICES,
                                                        AUGMENTED_SIZES))
def test_infer_augmented_state_size(indices, augmented_size):
    assert infer_augmented_state_size(*indices) == augmented_size


@pytest.mark.parametrize("angular_indices, non_angular_indices", INDICES)
def test_complementary_indices(angular_indices, non_angular_indices):
    other = complementary_indices(angular_indices, 5)
    assert (other == non_angular_indices).all()

    other = complementary_indices(non_angular_indices, 5)
    assert (other == angular_indices).all()


@pytest.mark.parametrize("indices, augmented_size", zip(INDICES,
                                                        AUGMENTED_SIZES))
def test_augment_reduce_state(indices, augmented_size):
    X = torch.randn(30, 5, requires_grad=True)
    Y = augment_state(X, *indices)
    assert Y.shape[1] == augmented_size

    X_ = reduce_state(Y, *indices)
    assert torch.isclose(X % (2 * np.pi), X_ % (2 * np.pi), 1e-3).all()

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            torch.autograd.grad(Y[i, j], X, retain_graph=True)

    for i in range(X_.shape[0]):
        for j in range(X_.shape[1]):
            J, = torch.autograd.grad(X_[i, j], X, retain_graph=True)
            assert J[i, j].isclose(torch.tensor(1.0), 1e-3)
