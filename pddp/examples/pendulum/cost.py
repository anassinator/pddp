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
"""Pendulum cost."""

import torch

from .model import PendulumDynamicsModel

from ...costs import QRCost
from ...utils.encoding import StateEncoding
from ...utils.angular import (augment_encoded_state, augment_state,
                              infer_augmented_state_size)


class PendulumCost(QRCost):

    """Pendulum cost."""

    def __init__(self, pendulum_length=0.5):
        """Constructs a PendulumCost.

        Args:
            pendulum_length (float): Pole length [m].
        """
        model = PendulumDynamicsModel

        augmented_state_size = infer_augmented_state_size(
            model.angular_indices, model.non_angular_indices)

        # We minimize the distance between the tip of the pendulum and the goal.
        # Don't penalize instantaneous velocities as much.
        # Note: we are operating on the augmented state vectors here:
        #   [theta', sin(theta), cos(theta)]
        Q = torch.zeros(augmented_state_size, augmented_state_size)
        Q[0, 0] = 0.1
        Q[0, 1] = Q[1, 0] = pendulum_length
        Q[1, 1] = Q[1, 1] = pendulum_length**2
        R = 0.1 * torch.eye(model.action_size)

        # Goal is not all zeroes after augmenting the state.
        x_goal = augment_state(
            torch.zeros(model.state_size), model.angular_indices,
            model.non_angular_indices)

        super(PendulumCost, self).__init__(Q, R, x_goal=x_goal)

    def forward(self,
                z,
                u,
                i,
                terminal=False,
                encoding=StateEncoding.DEFAULT,
                **kwargs):
        """Cost function.

        Args:
            z (Tensor<..., encoded_state_size): Encoded state distribution.
            u (Tensor<..., action_size>): Action vector.
            i (Tensor<...>): Time index.
            terminal (bool): Whether the cost is terminal. If so, u should be
                `None`.
            encoding (int): StateEncoding enum.

        Returns:
            The expectation of the cost (Tensor<...>).
        """
        model = PendulumDynamicsModel

        z = augment_encoded_state(z, model.angular_indices,
                                  model.non_angular_indices, encoding,
                                  model.state_size)

        return super(PendulumCost, self).forward(z, u, i, terminal, encoding,
                                                 **kwargs)