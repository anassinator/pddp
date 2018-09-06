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
"""Cartpole dynamics model."""

import torch

from torch.nn import Parameter

from ...models.base import DynamicsModel
from ...utils.constraint import constrain_model
from ...utils.classproperty import classproperty
from ...utils.angular import augment_state, reduce_state
from ...utils.encoding import StateEncoding, decode_var, decode_mean, encode


class CartpoleDynamicsModel(DynamicsModel):

    """Cartpole dynamics model.

    Note:
        state: [x, x', theta, theta']
        action: [F]
        theta: 0 is pointing up and increasing clockwise.
    """

    def __init__(self, dt, mc=0.5, mp=0.5, l=0.5, mu=0.1, g=-9.80665):
        """Constructs a CartpoleDynamicsModel.

        Args:
            dt (float): Time step [s].
            mc (float): Cart mass [kg].
            mp (float): Pendulum mass [kg].
            l (float): Pendulum length [m].
            mu (float): Coefficient of friction [dimensionless].
            g (float): Gravity acceleration [m/s^2].
        """
        super(CartpoleDynamicsModel, self).__init__()
        self.dt = Parameter(torch.tensor(dt), requires_grad=False)
        self.mc = Parameter(torch.tensor(mc), requires_grad=True)
        self.mp = Parameter(torch.tensor(mp), requires_grad=True)
        self.l = Parameter(torch.tensor(l), requires_grad=True)
        self.mu = Parameter(torch.tensor(mu), requires_grad=True)
        self.g = Parameter(torch.tensor(g), requires_grad=True)

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        return 1

    @classproperty
    def state_size(cls):
        """State size (int)."""
        return 4

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([2]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.tensor([0, 1, 3]).long()

    def fit(self, X, U, dX, quiet=False, **kwargs):
        """Fits the dynamics model.

        Args:
            X (Tensor<N, state_size>): State trajectory.
            U (Tensor<N, action_size>): Action trajectory.
            dX (Tensor<N, state_size>): Next state trajectory.
            quiet (bool): Whether to print anything to screen or not.
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
        mc = self.mc
        mp = self.mp
        l = self.l
        mu = self.mu
        g = self.g

        mean = decode_mean(z, encoding)
        var = decode_var(z, encoding)

        x = mean[..., 0]
        x_dot = mean[..., 1]
        theta = mean[..., 2]
        theta_dot = mean[..., 3]
        F = u.flatten()

        sin_theta = theta.sin()
        cos_theta = theta.cos()

        a0 = mp * l * theta_dot**2 * sin_theta
        a1 = g * sin_theta
        a2 = F - mu * x_dot
        a3 = 4 * (mc + mp) - 3 * mp * cos_theta**2

        theta_dot_dot = -3 * (a0 * cos_theta + 2 * (
            (mc + mp) * a1 + a2 * cos_theta)) / (l * a3)
        x_dot_dot = (2 * a0 + 3 * mp * a1 * cos_theta + 4 * a2) / a3

        mean = torch.stack(
            [
                x + x_dot * dt,
                x_dot + x_dot_dot.view(x_dot.shape) * dt,
                theta + theta_dot * dt,
                theta_dot + theta_dot_dot.view(theta_dot.shape) * dt,
            ],
            dim=-1)

        return encode(mean, V=var, encoding=encoding)


ConstrainedCartpoleDynamicsModel = \
    constrain_model(-10.0, 10.0)(CartpoleDynamicsModel)
