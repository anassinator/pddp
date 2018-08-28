# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""State encoding utilities."""

from __future__ import division

import torch
import numpy as np


class StateEncoding(object):

    """State encoding types."""

    # Encode full covariance matrix.
    FULL_COVARIANCE_MATRIX = 0

    # Encode only the upper triangular elements of the Cholesky factorization of
    # the covariance matrix (default).
    UPPER_TRIANGULAR_CHOLESKY = DEFAULT = 1

    # Encode the variance only.
    VARIANCE_ONLY = 2

    # Encode the standard deviation only.
    STANDARD_DEVIATION_ONLY = 3

    # Encode the mean only without any uncertainty.
    IGNORE_UNCERTAINTY = 4


def infer_encoded_state_size(x, encoding=StateEncoding.DEFAULT):
    """Computes the expected encoded state size from a state distribution.

    Args:
        x (Tensor<state_size>): Mean state distribution.
        encoding (int): StateEncoding enum.

    Returns:
        Encoded state size (int).
    """
    state_size = x.shape[-1]
    if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
        return state_size + state_size**2
    elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
        return int(0.5 * (3 * state_size + state_size**2))
    elif encoding == StateEncoding.VARIANCE_ONLY:
        return 2 * state_size
    elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
        return 2 * state_size
    elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
        return state_size
    else:
        raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))


def infer_state_size(z, encoding=StateEncoding.DEFAULT):
    """Computes the original state size from an encoded state.

    Args:
        z (Tensor<..., n>): Encoded state vector.
        encoding (int): StateEncoding enum.

    Returns:
        State size (int).
    """
    n = z.shape[-1]
    if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
        # n = state_size + state_size^2
        return int(0.5 * (-1 + np.sqrt(1 + 4 * n)))
    elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
        # n = (3 * state_size + state_size^2) / 2
        return int(0.5 * (-3 + np.sqrt(9 + 8 * n)))
    elif encoding == StateEncoding.VARIANCE_ONLY:
        # n = 2 * state_size
        return n // 2
    elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
        # n = 2 * state_size
        return n // 2
    elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
        return n
    else:
        raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))


def encode(M, C=None, V=None, S=None, encoding=StateEncoding.DEFAULT):
    """Encodes a state distribution (in batches).

    Note: At least one of C, V, or S must be specified properly.

    Args:
        M (Tensor<..., state_size>): Mean vector(s).
        C (Tensor<..., state_size, state_size>): Covariance matrices.
        V (Tensor<..., state_size>): Variance vector(s).
        S (Tensor<..., state_size>): Standard deviation vector(s).
        encoding (int): StateEncoding enum.

    Returns:
        Encoded state vector(s) (Tensor<..., state_size>).
    """
    state_size = M.shape[-1]
    tensor_opts = {
        "device": M.device,
        "dtype": M.dtype,
        "requires_grad": M.requires_grad
    }
    encoded_state_size = infer_encoded_state_size(M, encoding)

    if M.dim() == 1:
        Z = torch.zeros(encoded_state_size, **tensor_opts)
    else:
        Z = torch.zeros(M.shape[0], encoded_state_size, **tensor_opts)

    Z[..., :state_size] += M

    if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
        C = _C_from(C, V, S)
        if C.dim() == 2:
            C = C.view(-1)
        elif C.dim() == 3:
            C = C.view(-1, state_size * state_size)
        else:
            raise NotImplementedError("Expected a 2D or 3D tensor")

        Z[..., state_size:] += C
    elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
        C = _C_from(C, V, S)

        if C.dim() == 2:
            L = C.potrf()
        elif C.dim() == 3:
            # TODO: Remove for-loop.
            L = torch.zeros_like(C, requires_grad=M.requires_grad)
            for c, l in zip(C, L):
                l += c.potrf()
        else:
            raise NotImplementedError("Expected a 2D or 3D tensor")

        triu_x, triu_y = np.triu_indices(state_size)
        Z[..., state_size:] += L[..., triu_x, triu_y]
    elif encoding == StateEncoding.VARIANCE_ONLY:
        Z[..., state_size:] += _V_from(C, V, S)
    elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
        Z[..., state_size:] += _S_from(C, V, S)
    elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
        # Nothing to do.
        pass
    else:
        raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))

    return Z


def decode_mean(Z, encoding=StateEncoding.DEFAULT, state_size=None):
    """Computes the original state mean from a z-encoded state in batches.

    Args:
        Z (Tensor<..., n>): Encoded state vector(s).
        encoding (int): StateEncoding enum.
        state_size (int): Original state size, default: inferred.

    Returns:
        Mean vector(s) (Tensor<..., state_size>).
    """
    mean, _, _ = _split(Z, encoding, state_size)
    return mean


def decode_covar(Z, encoding=StateEncoding.DEFAULT, state_size=None):
    """Computes the original state covariances from a z-encoded state
    in batches.

    Args:
        Z (Tensor<..., n>): Encoded state vector(s).
        encoding (int): StateEncoding enum.
        state_size (int): Original state size, default: inferred.

    Returns:
        Covariance(s) (Tensor<..., state_size, state_size>).
    """
    _, other, state_size = _split(Z, encoding, state_size)

    if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
        if Z.dim() == 1:
            return other.view(state_size, state_size)
        elif Z.dim() == 2:
            return other.view(Z.shape[0], state_size, state_size)
        else:
            raise NotImplementedError("Expected a 1D or 2D tensor")
    elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
        L = _L_from_flat_triu(other, state_size)
        return L.transpose(dim0=-2, dim1=-1).matmul(L)
    elif encoding == StateEncoding.VARIANCE_ONLY:
        if other.dim() == 1:
            return other.diag()
        elif other.dim() == 2:
            # TODO: Remove for-loop.
            return torch.stack([x.diag() for x in other])
        else:
            raise NotImplementedError("Expected a 1D or 2D tensor")
    elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
        if other.dim() == 1:
            return other.diag().pow(2)
        elif other.dim() == 2:
            # TODO: Remove for-loop.
            return torch.stack([x.diag().pow(2) for x in other])
        else:
            raise NotImplementedError("Expected a 1D or 2D tensor")
    elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
        # Hard-code a unit normal distribution.
        C = 1e-6 * torch.eye(
            state_size, state_size, dtype=Z.dtype, device=Z.device)

        if Z.dim() == 1:
            pass
        elif Z.dim() == 2:
            C = C.expand(Z.shape[0], state_size, state_size)
        else:
            raise NotImplementedError("Expected a 1D or 2D tensor")

        if Z.requires_grad:
            C.requires_grad_()

        return C
    else:
        raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))


def decode_var(Z, encoding=StateEncoding.DEFAULT, state_size=None):
    """Computes the original state variances from a z-encoded state in batches.

    Args:
        Z (Tensor<..., n>): Encoded state vector(s).
        encoding (int): StateEncoding enum.
        state_size (int): Original state size, default: inferred.

    Returns:
        Variance(s) (Tensor<..., state_size>).
    """
    _, other, state_size = _split(Z, encoding, state_size)

    if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
        return _batch_diag_from_flat(other, state_size)
    elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
        return _batch_diag_from_flat_triu_cholesky(other, state_size)
    elif encoding == StateEncoding.VARIANCE_ONLY:
        return other
    elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
        return other.pow(2)
    elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
        # Hard-code a unit normal distribution.
        V = 1e-6 * torch.ones(state_size, dtype=Z.dtype, device=Z.device)

        if Z.dim() == 1:
            pass
        elif Z.dim() == 2:
            V = V.expand(Z.shape[0], state_size)
        else:
            raise NotImplementedError("Expected a 1D or 2D tensor")

        if Z.requires_grad:
            V.requires_grad_()

        return V
    else:
        raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))


def decode_std(Z, encoding=StateEncoding.DEFAULT, state_size=None):
    """Computes the original state standard deviations from a z-encoded state
    in batches.

    Args:
        Z (Tensor<..., n>): Encoded state vector(s).
        encoding (int): StateEncoding enum.
        state_size (int): Original state size, default: inferred.

    Returns:
        Standard deviation(s) (Tensor<..., state_size>).
    """
    _, other, state_size = _split(Z, encoding, state_size)

    if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
        return _batch_diag_from_flat(other, state_size).sqrt()
    elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
        return _batch_diag_from_flat_triu_cholesky(other, state_size).sqrt()
    elif encoding == StateEncoding.VARIANCE_ONLY:
        return other.sqrt()
    elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
        return other
    elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
        # Hard-code a unit normal distribution.
        S = 1e-3 * torch.ones(state_size, dtype=Z.dtype, device=Z.device)

        if Z.dim() == 1:
            pass
        elif Z.dim() == 2:
            S = S.expand(Z.shape[0], state_size)
        else:
            raise NotImplementedError("Expected a 1D or 2D tensor")

        if Z.requires_grad:
            S.requires_grad_()

        return S
    else:
        raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))


def _split(Z, encoding=StateEncoding.DEFAULT, state_size=None):
    """Splits a z-encoded vector or a batch of z-encoded vectors into means and
    the remainder.

    Args:
        Z (Tensor<..., n>): Encoded state vector(s).
        encoding (int): StateEncoding enum.
        state_size (int): Original state size, default: inferred.

    Returns:
        Tuple of:
            Mean vector(s) (Tensor<..., state_size>),
            Remaining encoded data (Tensor<..., n - state_size>),
            Inferred state size (int).
    """
    if state_size is None:
        # Infer the state size.
        state_size = infer_state_size(Z, encoding)

    n_other = Z.shape[-1] - state_size
    if n_other > 0:
        mean, other = torch.split(Z, [state_size, n_other], dim=-1)
    else:
        mean = Z
        other = torch.empty(
            0, dtype=Z.dtype, device=Z.device, requires_grad=Z.requires_grad)

    return mean, other, state_size


def _C_from(C=None, V=None, S=None):
    """Converts a measure of uncertainty into covariance matrices.

    Note: At least one of C, V, or S must be specified properly.

    Args:
        C (Tensor<..., state_size, state_size>): Covariance matrices.
        V (Tensor<..., state_size>): Variance vector(s).
        S (Tensor<..., state_size>): Standard deviation vector(s).

    Returns:
        Covariance matrices (Tensor<..., state_size, state_size>).
    """
    if C is not None:
        return C

    V = _V_from(C, V, S)
    if V.dim() == 1:
        return V.diag()
    elif V.dim() == 2:
        n, m = V.shape
        C = torch.zeros(n, m, m, dtype=V.dtype, device=V.device)
        diag_x, diag_y = np.diag_indices(m)
        C[..., diag_x, diag_y] = V

        if V.requires_grad:
            C.requires_grad_()

        return C
    else:
        raise NotImplementedError("Expected either a 1D or 2D tensor")


def _V_from(C=None, V=None, S=None):
    """Converts measures of uncertainty into variances.

    Note: At least one of C, V, or S must be specified properly.

    Args:
        C (Tensor<..., state_size, state_size>): Covariance matrix.
        V (Tensor<..., state_size>): Variance vector(s).
        S (Tensor<..., state_size>): Standard deviation vector(s).

    Returns:
        Variance vector(s) (Tensor<..., state_size>).
    """
    if V is not None:
        return V

    if S is not None:
        return S.pow(2)

    if C is not None:
        n = C.shape[-1]
        diag_x, diag_y = np.diag_indices(n)
        return C[..., diag_x, diag_y]

    raise ValueError("At least one of C, V, S must be specified")


def _S_from(C=None, V=None, S=None):
    """Converts a measure of uncertainty into a standard deviation.

    Note: At least one of C, V, or S must be specified properly.

    Args:
        C (Tensor<..., state_size, state_size>): Covariance matrix.
        V (Tensor<..., state_size>): Variance vector(s).
        S (Tensor<..., state_size>): Standard deviation vector(s).

    Returns:
        Standard deviation vector(s) (Tensor<..., state_size>).
    """
    if S is not None:
        return S

    if V is not None:
        return V.sqrt()

    if C is not None:
        V = _V_from(C, V, S)
        return V.sqrt()

    raise ValueError("At least one of C, V, S must be specified")


def _L_from_flat_triu(X, size):
    """Decodes a flattened upper triangular matrix.

    Args:
        X (Tensor<..., n>): Flattened upper triangular vector(s).
        size (int): Size of a single dimension of the unflattened matrix.

    Returns:
        The unflattened upper triangular matrix (Tensor<..., size, size).
    """
    if X.dim() == 1:
        L = torch.zeros(size, size, dtype=X.dtype, device=X.device)
    elif X.dim() == 2:
        L = torch.zeros(X.shape[0], size, size, dtype=X.dtype, device=X.device)
    else:
        raise NotImplementedError

    triu_x, triu_y = np.triu_indices(size)
    L[..., triu_x, triu_y] = X

    if X.requires_grad:
        L.requires_grad_()

    return L


def _batch_diag_from_flat(X, size):
    """Decodes the diagonal elements of a flattened matrix.

    Args:
        X (Tensor<..., n>): Flattened vector(s).
        size (int): Size of a single dimension of the unflattened matrix.

    Returns:
        The diagonal elements of the matrix (Tensor<..., size>).
    """
    return X[..., range(0, X.shape[-1], size + 1)]


def _batch_diag_from_flat_triu_cholesky(X, size):
    """Decodes the diagonal elements of a flattened upper triangular Cholesky
    decomposed matrix.

    Args:
        X (Tensor<..., n>): Flattened upper triangular Cholesky decomposed
            vector(s).
        size (int): Size of a single dimension of the unflattened matrix.

    Returns:
        The diagonal elements of the recomposed matrix (Tensor<..., size>).
    """
    L = _L_from_flat_triu(X, size)
    return L.pow(2).sum(dim=-2)
