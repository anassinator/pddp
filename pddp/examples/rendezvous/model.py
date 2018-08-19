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
"""Multi-vehicle rendezvous dynamics model."""

import tqdm
import torch
import numpy as np
from torch.nn import Parameter
from ...models.base import DynamicsModel
from ...utils.classproperty import classproperty
from ...models.utils.encoding import StateEncoding, decode_covar, decode_mean, encode


class RendezvousDynamicsModel(DynamicsModel):

    """Multi-vehicle rendezvous dynamics model.

    Note:
        state: [x_0, y_0, x_1, y_1, x_0_dot, y_0_dot, x_1_dot, y_1_dot]
        action: [F_x_0, F_y_0, F_x_1, F_y_1]
    """

    def __init__(self, dt, m=1.0, alpha=0.1):
        """Constructs RendezvousDynamicsModel.

        Args:
            dt (float): Time step [s].
            m: Vehicle mass [kg].
            alpha: Friction coefficient.
        """
        super(RendezvousDynamicsModel, self).__init__()
        self.dt = Parameter(torch.tensor(dt), requires_grad=False)
        self.m = Parameter(torch.tensor(m), requires_grad=True)
        self.alpha = Parameter(torch.tensor(alpha), requires_grad=True)

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        return 4

    @classproperty
    def state_size(cls):
        """Augmented state size (int)."""
        return 8

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.arange(8).long()

    def fit(self, X_, dX, quiet=False, tqdm_class=tqdm.tqdm, **kwargs):
        """Fits the dynamics model.

        Args:
            X_ (Tensor<N, state_size + action_size>): State-action pair
                trajectory.
            dX (Tensor<N, action_size>): Encoded next state distribution.
            quiet (bool): Whether to print anything to screen or not.
            tqdm_class (class): `tqdm` class for progress bar output. Can be
                completely silenced by setting `quiet=True`.
        """
        # No need: this is an exact dynamics model.
        pass

    def forward(self, z, u, i, encoding=StateEncoding.DEFAULT, **kwargs):
        """Dynamics model function.

        Args:
            z (Tensor<..., encoded_state_size>): Encoded state distribution.
            u (Tensor<..., action_size>): Action vector(s).
            i (int): Time index.
            encoding (int): StateEncoding enum.

        Returns:
            Next encoded state distribution (Tensor<..., encoded_state_size>).
        """
        dt = self.dt
        m = self.m
        alpha = self.alpha

        x = decode_mean(z, encoding)
        covar = decode_covar(z, encoding)

        # Define acceleration.
        def acceleration(x_dot, u):
            x_dot_dot = x_dot * (1 - alpha * dt / m) + u * dt / m
            return x_dot_dot

        mean = torch.stack(
            [
                x[..., 0] + x[..., 4] * dt,
                x[..., 1] + x[..., 5] * dt,
                x[..., 2] + x[..., 6] * dt,
                x[..., 3] + x[..., 7] * dt,
                x[..., 4] + acceleration(x[..., 4], u[..., 0]) * dt,
                x[..., 5] + acceleration(x[..., 5], u[..., 1]) * dt,
                x[..., 6] + acceleration(x[..., 6], u[..., 2]) * dt,
                x[..., 7] + acceleration(x[..., 7], u[..., 3]) * dt,
            ],
            dim=-1)

        return encode(mean, C=covar, encoding=encoding)
