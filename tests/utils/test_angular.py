"""Unit tests for pddp.utils.angular."""

import torch
import pytest
import numpy as np

from pddp.utils.angular import *
from pddp.utils.gaussian_variable import GaussianVariable
from pddp.utils.encoding import StateEncoding, infer_encoded_state_size

# Organized as angular_indices, non_angular_indices pairs.
INDICES = [
    (torch.tensor([]).long(), torch.tensor([0, 1, 2, 3, 4]).long()),
    (torch.tensor([2]).long(), torch.tensor([0, 1, 3, 4]).long()),
    (torch.tensor([1, 3]).long(), torch.tensor([0, 2, 4]).long()),
    (torch.tensor([1, 3, 4]).long(), torch.tensor([0, 2]).long()),
    (torch.tensor([1, 2, 3, 4]).long(), torch.tensor([0]).long()),
    (torch.tensor([0, 1, 2, 3, 4]).long(), torch.tensor([]).long()),
]
AUGMENTED_SIZES = [5, 6, 7, 8, 9, 10]
STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]


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

    # Verify they match with the encoded augmentation implementation.
    Z = torch.cat([X, torch.zeros_like(X)], dim=-1)
    Y_ = augment_encoded_state(
        Z, *indices, encoding=StateEncoding.VARIANCE_ONLY)
    assert Y_[..., :augmented_size].allclose(Y)

    torch.autograd.gradcheck(augment_state,
                             (X.double(), indices[0], indices[1]))
    torch.autograd.gradcheck(reduce_state, (Y.double(), indices[0], indices[1]))
    torch.autograd.gradcheck(
        augment_encoded_state,
        (Z.double(), indices[0], indices[1], StateEncoding.VARIANCE_ONLY))


@pytest.mark.parametrize("indices, augmented_size", zip(INDICES,
                                                        AUGMENTED_SIZES))
@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_augment_encoded_state(encoding, indices, augmented_size):
    state_size = 5
    N = 3
    X = [GaussianVariable.random(state_size) for i in range(N)]
    Z = torch.stack([x.encode(encoding).double() for x in X])

    angular_indices, non_angular_indices = indices
    Z_ = augment_encoded_state(Z, angular_indices, non_angular_indices,
                               encoding)
    assert Z_.shape[0] == N
    assert Z_.shape[1] == infer_encoded_state_size(augmented_size, encoding)

    torch.autograd.gradcheck(
        augment_encoded_state,
        (Z, angular_indices, non_angular_indices, encoding))

    torch.autograd.gradgradcheck(
        augment_encoded_state,
        (Z, angular_indices, non_angular_indices, encoding))