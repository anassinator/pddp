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
"""Multi-vehicle rendezvous cost."""

import torch

from .model import RendezvousDynamicsModel

from ...costs import QRCost
from ...models.utils.angular import augment_state, infer_augmented_state_size


class RendezvousCost(QRCost):

    """Multi-vehicle rendezvous cost."""

    def __init__(self):
        """Constructs a RendezvousCost."""
        model = RendezvousDynamicsModel

        # In order to approach the two vehicles to each other, Q is set up to
        # penalize differences in positions as ||x_0 - x_1||^2 while penalizing
        # non-zero velocities.
        Q = torch.eye(model.state_size)
        Q[0, 2] = Q[2, 0] = -1
        Q[1, 3] = Q[3, 1] = -1

        R = 0.1 * torch.eye(model.action_size)

        # Goal is not all zeroes after augmenting the state.
        super(RendezvousCost, self).__init__(Q, R)
