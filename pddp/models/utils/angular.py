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


def augment_state(x, angular_indices, non_angular_indices):
    """Augments state vector by replacing all angular states with their complex
    representation at the end of the vector.

    Args:
        x (Tensor<N, state_size>): Original state vector.
        angular_indices (Tensor<n>): Column indices of angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            `angular_indices`.

    Returns:
        Augmented state vector (Tensor<N, augmented_state_size>) as
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
        x_ (Tensor<N, augmented_state_size>): Augmented state vector.
        angular_indices (Tensor<n>): Column indices of original angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            `angular_indices`.

    Returns:
        Original state vector (Tensor<N, state_size>).
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
        x = torch.empty(n_angles + n_others, dtype=x_.dtype)
    else:
        x = torch.empty(x_.shape[0], n_angles + n_others, dtype=x_.dtype)

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
