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
"""Cartpole cost."""

import torch

from .model import CartpoleDynamicsModel

from ...costs import QRCost
from ...utils.encoding import StateEncoding
from ...utils.angular import (augment_encoded_state, augment_state,
                              infer_augmented_state_size)


class CartpoleCost(QRCost):

    """Cartpole cost."""

    def __init__(self, pole_length=0.5):
        """Constructs a CartpoleCost.

        Args:
            pole_length (float): Pole length [m].
        """
        model = CartpoleDynamicsModel
        augmented_state_size = infer_augmented_state_size(
            model.angular_indices, model.non_angular_indices)

        Q_term = torch.eye(augmented_state_size)
        Q = torch.zeros_like(Q_term)

        # We minimize the distance between the tip of the pole and the goal.
        # Note: we are operating on the augmented state vectors here:
        #   [x, x', theta', sin(theta), cos(theta)]
        Q[0, 0] = 1.0
        Q[1, 1] = Q[2, 2] = 0.0
        Q[0, 3] = Q[3, 0] = pole_length
        Q[3, 3] = Q[4, 4] = pole_length**2
        R = 0.1 * torch.eye(model.action_size)

        # Goal is not all zeroes after augmenting the state.
        x_goal = augment_state(
            torch.zeros(model.state_size), model.angular_indices,
            model.non_angular_indices)

        super(CartpoleCost, self).__init__(Q, R, x_goal=x_goal)

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
        model = CartpoleDynamicsModel

        z = augment_encoded_state(z, model.angular_indices,
                                  model.non_angular_indices, encoding,
                                  model.state_size)

        return super(CartpoleCost, self).forward(z, u, i, terminal, encoding,
                                                 **kwargs)