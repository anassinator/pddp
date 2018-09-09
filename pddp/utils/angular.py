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
"""Angular state utilities."""

from __future__ import division

import torch
import numpy as np

from .encoding import (StateEncoding, decode_mean, decode_var, decode_covar,
                       encode)


def complementary_indices(indices, size):
    """Computes the complementary indices of an index vector.

    Args:
        indices (Tensor<n>): Indices.
        size (int): Size of the index space.

    Returns:
        Complementary indices vector (Tensor<m>).
    """
    all_indices = torch.arange(0, size).long()
    if len(indices) == 0:
        return all_indices
    elif len(indices) == len(all_indices):
        return torch.tensor([]).long()

    other_indices = (1 - torch.eq(indices, all_indices.view(-1, 1)))
    other_indices = other_indices.prod(dim=1).nonzero()[:, 0]
    return other_indices


def augment_encoded_state(z,
                          angular_indices,
                          non_angular_indices,
                          encoding=StateEncoding.DEFAULT,
                          state_size=None):
    """Augments an encoded state vector by replacing all angular states and
    corresponding uncertainties with their complex representation.

    Args:
        z (Tensor<..., encoded_state_size>): Encoded state vector.
        angular_indices (Tensor<n>): Column indices of angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            `angular_indices`.
        encoding (int): StateEncoding enum.
        state_size (int): Original state size, default: inferred.

    Returns:
        Augmented encoded state vector
        (Tensor<..., augmented_encoded_state_size>).
    """
    if encoding == StateEncoding.IGNORE_UNCERTAINTY:
        return augment_state(z, angular_indices, non_angular_indices)

    mean = decode_mean(z, encoding, state_size)

    if encoding in (StateEncoding.FULL_COVARIANCE_MATRIX,
                    StateEncoding.UPPER_TRIANGULAR_CHOLESKY):
        covar = decode_covar(z, encoding, state_size)
        M, C = _augment_covar(mean, covar, angular_indices, non_angular_indices)
        return encode(M, C=C, encoding=encoding)

    if encoding in (StateEncoding.VARIANCE_ONLY,
                    StateEncoding.STANDARD_DEVIATION_ONLY):
        var = decode_var(z, encoding, state_size)
        M, V = _augment_var(mean, var, angular_indices, non_angular_indices)
        return encode(M, V=V, encoding=encoding)

    raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))


def _augment_var(m, v, angular_indices, non_angular_indices):
    """Augments mean and variance vectors by replacing all angular states and
    corresponding variances with their complex representation.

    Args:
        m (Tensor<..., state_size>): Mean vector(s).
        v (Tensor<..., state_size>): Variance vector(s).
        angular_indices (Tensor<n>): Column indices of angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            `angular_indices`.

    Returns:
        Tuple of:
            M (Tensor<..., augmented_state_size>): Augmented mean vector(s).
            V (Tensor<..., augmented_state_size>): Augmented variance vector(s).
    """
    # Based on https://github.com/mcgillmrl/kusanagi
    if len(angular_indices) == 0:
        return m, v

    tensor_opts = {"dtype": m.dtype, "device": m.device}

    D = m.shape[-1]
    n_angles = len(angular_indices)
    Da = 2 * n_angles
    Dna = len(non_angular_indices)

    if m.dim() == 1:
        Ma = torch.empty(Da, **tensor_opts)
        Va = torch.empty(Da, **tensor_opts)
    elif m.dim() == 2:
        N = m.shape[0]
        Ma = torch.empty(N, Da, **tensor_opts)
        Va = torch.empty(N, Da, **tensor_opts)
    else:
        raise NotImplementedError("Unsupported number of dimensions")

    # Compute the mean.
    mi = m[..., angular_indices]
    vi = v[..., angular_indices]
    vii = vi.unsqueeze(-2)
    vii = torch.stack([vii.diag() for vii in vi]) if m.dim() == 2 else vi.diag()

    exp_vi_h = (-0.5 * vi).exp()
    Ma[..., ::2] = exp_vi_h * mi.sin()
    Ma[..., 1::2] = exp_vi_h * mi.cos()

    # Compute the entries in the augmented variance vectors.
    lq = -0.5 * (vi.unsqueeze(-1) + vi.unsqueeze(-2))
    q = lq.exp()
    exp_lq_p_vi = (lq + vii).exp()
    exp_lq_m_vi = (lq - vii).exp()
    U3 = (exp_lq_p_vi - q) * (mi.unsqueeze(-1) - mi.unsqueeze(-2)).cos()
    U4 = (exp_lq_m_vi - q) * (mi.unsqueeze(-1) + mi.unsqueeze(-2)).cos()

    diag_idx = torch.arange(n_angles)
    U3 = U3[..., diag_idx, diag_idx]
    U4 = U4[..., diag_idx, diag_idx]

    Va[..., ::2] = U3 - U4
    Va[..., 1::2] = U3 + U4
    Va = 0.5 * Va

    # Construct mean vectors.
    Mna = m[..., non_angular_indices]
    M = torch.cat([Mna, Ma], dim=-1)

    # Construct the corresponding variance vectors.
    Vna = v[..., non_angular_indices]
    V = torch.cat([Vna, Va], dim=-1)

    return M, V


def _augment_covar(m, c, angular_indices, non_angular_indices):
    """Augments mean vectors and covariances by replacing all angular states and
    corresponding covariances with their complex representation.

    Args:
        m (Tensor<..., state_size>): Mean vector(s).
        c (Tensor<..., state_size>): Covariance matrices.
        angular_indices (Tensor<n>): Column indices of angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            `angular_indices`.

    Returns:
        Tuple of:
            M (Tensor<..., augmented_state_size>): Augmented mean vector(s).
            C (Tensor<..., augmented_state_size>): Augmented covariance matrices.
    """
    # Based on https://github.com/mcgillmrl/kusanagi
    tensor_opts = {"dtype": m.dtype, "device": m.device}

    D = m.shape[-1]
    Da = len(angular_indices) * 2
    Dna = len(non_angular_indices)

    if m.dim() == 1:
        Ma = torch.zeros(Da, **tensor_opts)
        Va = torch.zeros(Da, Da, **tensor_opts)
        Ca = torch.zeros(D, Da, **tensor_opts)
        C = torch.zeros(Dna + Da, Dna + Da, **tensor_opts)
    elif m.dim() == 2:
        N = m.shape[0]
        Ma = torch.zeros(N, Da, **tensor_opts)
        Va = torch.zeros(N, Da, Da, **tensor_opts)
        Ca = torch.zeros(N, D, Da, **tensor_opts)
        C = torch.zeros(N, Dna + Da, Dna + Da, **tensor_opts)
    else:
        raise NotImplementedError("Unsupported number of dimensions")

    # Compute the mean.
    mi = m[..., angular_indices]
    ci = c[..., angular_indices, :][..., :, angular_indices]
    cii = c[..., angular_indices, angular_indices]
    exp_cii_h = (-0.5 * cii).exp()

    Ma[..., ::2] = exp_cii_h * mi.sin()
    Ma[..., 1::2] = exp_cii_h * mi.cos()

    # Compute the entries in the augmented covariance matrix.
    lq = -0.5 * (cii.unsqueeze(-1) + cii.unsqueeze(-2))
    q = lq.exp()
    exp_lq_p_ci = (lq + ci).exp()
    exp_lq_m_ci = (lq - ci).exp()
    U1 = (exp_lq_p_ci - q) * (mi.unsqueeze(-1) - mi.unsqueeze(-2)).sin()
    U2 = (exp_lq_m_ci - q) * (mi.unsqueeze(-1) + mi.unsqueeze(-2)).sin()
    U3 = (exp_lq_p_ci - q) * (mi.unsqueeze(-1) - mi.unsqueeze(-2)).cos()
    U4 = (exp_lq_m_ci - q) * (mi.unsqueeze(-1) + mi.unsqueeze(-2)).cos()

    Va[..., ::2, ::2] = U3 - U4
    Va[..., 1::2, 1::2] = U3 + U4
    Va[..., ::2, 1::2] = U1 + U2
    Va[..., 1::2, ::2] = Va[..., ::2, 1::2].transpose(-1, -2)
    Va = 0.5 * Va

    # Inv times input output covariance.
    Is = 2 * torch.arange(len(angular_indices))
    Ic = Is + 1
    Ca[..., angular_indices, Is] = Ma[..., 1::2]
    Ca[..., angular_indices, Ic] = -Ma[..., ::2]

    # Construct mean vectors.
    Mna = m[..., non_angular_indices]
    M = torch.cat([Mna, Ma], dim=-1)

    # Construct the corresponding covariance matrices.
    Vna = c[..., non_angular_indices, :][..., :, non_angular_indices]
    C[..., :Dna, :Dna] = Vna
    C[..., Dna:, Dna:] = Va

    # Fill in the cross covariances.
    C[..., :Dna, Dna:] = (
        c.unsqueeze(-1) * Ca.unsqueeze(-2)).sum(-3)[..., non_angular_indices, :]
    C[..., Dna:, :Dna] = C[..., :Dna, Dna:].transpose(-1, -2)

    return M, C


def augment_state(x, angular_indices, non_angular_indices):
    """Augments state vector by replacing all angular states with their complex
    representation at the end of the vector.

    Args:
        x (Tensor<..., state_size>): Original state vector.
        angular_indices (Tensor<n>): Column indices of angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            `angular_indices`.

    Returns:
        Augmented state vector (Tensor<..., augmented_state_size>) as
        [non_angular_states, sin(angular_states), cos(angular_states)].
    """
    if len(angular_indices) == 0:
        return x
    elif len(non_angular_indices) == 0:
        return torch.cat([x.sin(), x.cos()], dim=-1)

    angles = x.index_select(-1, angular_indices)
    others = x.index_select(-1, non_angular_indices)
    return torch.cat([others, angles.sin(), angles.cos()], dim=-1)


def reduce_state(x_, angular_indices, non_angular_indices):
    """Reduces an augmented state vector.

    Args:
        x_ (Tensor<..., augmented_state_size>): Augmented state vector.
        angular_indices (Tensor<n>): Column indices of original angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            `angular_indices`.

    Returns:
        Original state vector (Tensor<..., state_size>).
    """
    n_angles = len(angular_indices)
    if n_angles == 0:
        return x_

    n_others = len(non_angular_indices)
    if n_others == 0:
        sin_angles, cos_angles = x_.split([n_angles, n_angles], dim=-1)
        angles = torch.atan2(sin_angles, cos_angles)
        return angles

    others, sin_angles, cos_angles = x_.split(
        [n_others, n_angles, n_angles], dim=-1)
    angles = torch.atan2(sin_angles, cos_angles)

    if x_.dim() == 1:
        x = torch.empty(n_angles + n_others, dtype=x_.dtype, device=x_.device)
    else:
        x = torch.empty(
            x_.shape[0], n_angles + n_others, dtype=x_.dtype, device=x_.device)

    x[..., angular_indices] = angles
    x[..., non_angular_indices] = others

    return x


def infer_augmented_state_size(angular_indices, non_angular_indices):
    """Computes the augmented state vector size.

    Args:
        angular_indices (Tensor<n>): Column indices of angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            angular_indices.

    Return:
        Augmented state vector size (int).
    """
    return len(non_angular_indices) + 2 * len(angular_indices)


def infer_reduced_state_size(angular_indices, non_angular_indices):
    """Computes the reduced state vector size.

    Args:
        angular_indices (Tensor<n>): Column indices of angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            angular_indices.

    Return:
        Reduced state vector size (int).
    """
    return len(non_angular_indices) + len(angular_indices)