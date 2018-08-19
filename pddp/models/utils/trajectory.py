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
"""Trajectory utilities."""

import torch


def mean_trajectory(X):
    """Extracts the mean trajectory from a state trajectory distribution.

    Args:
        X (List<GaussianVariable<state_size>, N>): State distribution
            trajectory.

    Returns:
        Mean state trajectory (Tensor<N, state_size>).
    """
    N = len(X)
    if N == 0:
        raise ValueError("Trajectory cannot be empty")

    X_ = torch.empty(N, X[0].shape[0], dtype=X[0].dtype)
    for i in range(N):
        X_[i] = X[i].mean()
    return X_


def sample_trajectory(X):
    """Samples a trajectory from a state trajectory distribution.

    Args:
        X (List<GaussianVariable<state_size>, N>): State distribution
            trajectory.

    Returns:
        Sampled state trajectory (Tensor<N, state_size>).
    """
    N = len(X)
    if N == 0:
        raise ValueError("Trajectory cannot be empty")

    X_ = torch.empty(N, X[0].shape[0], dtype=X[0].dtype)
    for i in range(N):
        X_[i] = X[i].sample()
    return X_


def trajectory_to_training_data(X, U):
    """Converts a state-action trajectory into training data.

    Args:
        X (Tensor<N+1, state_size>): State trajectory.
        U (Tensor<N, action_size>): Action trajectory.
    Returns:
        Tuple of:
            X_ (Tensor<N, state_size + action_size>): State-action pairs.
            dX (Tensor<N, state_size>): State difference.
    """
    X_ = torch.cat([X[:-1], U], dim=-1)
    dX = X[:-1] - X[1:]
    return X_, dX
