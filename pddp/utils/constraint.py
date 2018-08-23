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
"""Constraint utilities."""

import torch
from .encoding import StateEncoding


def constrain(u, min_bounds, max_bounds):
    """Constrains the action through a tanh() squash function.

    Args:
        u (Tensor<action_size>): Action vector.
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Constrained action vector (Tensor<action_size>).
    """
    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    return diff * u.tanh() + mean


def constrain_env(env, min_bounds, max_bounds):
    """Constrains the action space of the environment through a squash function.

    Args:
        env (Env): Environment.
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.
    """

    def apply_fn(u):
        """Applies an action to the environment.

        Args:
            u (Tensor<action_size>): Action vector.
        """
        u = constrain(u, min_bounds, max_bounds)
        return _apply_fn(u)

    # Monkey-patch the env.
    _apply_fn = env.apply
    env.apply = apply_fn


def constrain_model(model, min_bounds, max_bounds):
    """Constrains the action space of the model through a squash function.

    Args:
        model (DynamicsModel): Dynamics model.
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.
    """

    def forward_fn(z, u, i, encoding=StateEncoding.DEFAULT, **kwargs):
        """Dynamics model function.

        Args:
            z (Tensor<..., encoded_state_size>): Encoded state distribution.
            u (Tensor<..., action_size>): Action vector(s).
            i (Tensor<...>): Time index.
            encoding (int): StateEncoding enum.

        Returns:
            Next encoded state distribution (Tensor<..., encoded_state_size>).
        """
        u = constrain(u, min_bounds, max_bounds)
        return _forward_fn(z, u, i, encoding=encoding, **kwargs)

    # Monkey-patch the model.
    _forward_fn = model.forward
    model.forward = forward_fn
