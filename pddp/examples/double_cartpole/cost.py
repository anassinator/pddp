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
"""Double cartpole cost."""

import torch
import numpy as np

from .model import DoubleCartpoleDynamicsModel

from ...costs import QRCost
from ...utils.encoding import StateEncoding
from ...utils.angular import (augment_encoded_state, augment_state,
                              infer_augmented_state_size)


class DoubleCartpoleCost(QRCost):

    """Double cartpole cost."""

    def __init__(self, pole1_length=0.6, pole2_length=0.6):
        """Constructs a DoubleCartpoleCost.

        Args:
            pole1_length (float): First link length [m].
            pole2_length (float): Second link length [m].
        """
        model = DoubleCartpoleDynamicsModel
        augmented_state_size = infer_augmented_state_size(
            model.angular_indices, model.non_angular_indices)

        # We minimize the distance between the tip of the pole and the goal.
        # Note: we are operating on the augmented state vectors here:
        #   [x, x', theta1', theta2',
        #    sin(theta1), cos(theta1),
        #    sin(theta2), cos(theta2)]
        Q_term = 100 * torch.eye(augmented_state_size)
        Q = torch.zeros(augmented_state_size, augmented_state_size)
        cost_dims = np.hstack([
            0,
            np.arange(augmented_state_size - 2 * len(model.angular_indices),
                      augmented_state_size)
        ])[:, None]
        C = torch.tensor([[1, -pole1_length, 0, -pole2_length, 0],
                          [0, 0, pole1_length, 0, pole2_length]])
        Q[cost_dims, cost_dims.T] = C.t().mm(C)

        R = 0.1 * torch.eye(model.action_size)

        # Goal is not all zeroes after augmenting the state.
        x_goal = augment_state(
            torch.zeros(model.state_size), model.angular_indices,
            model.non_angular_indices)

        super(DoubleCartpoleCost, self).__init__(
            Q, R, Q_term=Q_term, x_goal=x_goal)

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
        model = DoubleCartpoleDynamicsModel

        z = augment_encoded_state(z, model.angular_indices,
                                  model.non_angular_indices, encoding,
                                  model.state_size)

        return super(DoubleCartpoleCost, self).forward(z, u, i, terminal,
                                                       encoding, **kwargs)
