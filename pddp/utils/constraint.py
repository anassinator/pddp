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


def constrain_env(min_bounds, max_bounds):
    """Decorator that constrains the action space of an environment through a
    squash function.

    Args:
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Decorator that constrains an Env.
    """

    def decorator(cls):

        def apply_fn(self, u):
            """Applies an action to the environment.

            Args:
                u (Tensor<action_size>): Action vector.
            """
            u = constrain(u, min_bounds, max_bounds)
            return _apply_fn(self, u)

        # Monkey-patch the env.
        _apply_fn = cls.apply
        cls.apply = apply_fn

        return cls

    return decorator


def constrain_model(min_bounds, max_bounds):
    """Decorator that constrains the action space of a dynamics model through a
    squash function.

    Args:
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Decorator that constrains a DynamicsModel.
    """

    def decorator(cls):

        def init_fn(self, *args, **kwargs):
            """Constructs a DynamicsModel."""
            _init_fn(self, *args, **kwargs)
            self.max_bounds = torch.nn.Parameter(
                torch.tensor(max_bounds).expand(cls.action_size),
                requires_grad=False)
            self.min_bounds = torch.nn.Parameter(
                torch.tensor(max_bounds).expand(cls.action_size),
                requires_grad=False)

        def forward_fn(self, z, u, i, encoding=StateEncoding.DEFAULT, **kwargs):
            """Dynamics model function.

            Args:
                z (Tensor<..., encoded_state_size>): Encoded state distribution.
                u (Tensor<..., action_size>): Action vector(s).
                i (Tensor<...>): Time index.
                encoding (int): StateEncoding enum.

            Returns:
                Next encoded state distribution
                    (Tensor<..., encoded_state_size>).
            """
            u = constrain(u, min_bounds, max_bounds)
            return _forward_fn(self, z, u, i, encoding=encoding, **kwargs)

        def constrain_fn(self, u):
            """Constrains an action through a squash function.

            Args:
                u (Tensor<..., action_size>): Action vector(s).

            Returns:
                Constrained action vector(s) (Tensor<..., action_size>).
            """
            return constrain(u, min_bounds, max_bounds)

        # Monkey-patch the model.
        _init_fn = cls.__init__
        _forward_fn = cls.forward
        cls.__init__ = init_fn
        cls.forward = forward_fn
        cls.constrain = constrain_fn

        return cls

    return decorator
