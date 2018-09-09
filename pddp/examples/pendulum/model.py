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
"""Pendulum dynamics model."""

import torch
import numpy as np
from torch.nn import Parameter
from ...models.base import DynamicsModel
from ...utils.constraint import constrain_model
from ...utils.classproperty import classproperty
from ...utils.encoding import StateEncoding, decode_var, decode_mean, encode
from ...utils.angular import augment_state, complementary_indices, reduce_state


@constrain_model(-2.5, 2.5)
class PendulumDynamicsModel(DynamicsModel):

    """Pendulum dynamics model.

    Note:
        state: [theta, theta']
        action: [torque]
        theta: 0 is pointing up and increasing counter-clockwise.
    """

    def __init__(self, dt, m=1.0, l=1.0, mu=0.1, g=9.80665):
        """Constructs PendulumDynamicsModel.

        Args:
            dt (float): Time step [s].
            m (float): Pendulum mass [kg].
            l (float): Pendulum length [m].
            g (float): Gravity acceleration [m/s^2].
        """
        super(PendulumDynamicsModel, self).__init__()
        self.dt = Parameter(torch.tensor(dt), requires_grad=False)
        self.m = Parameter(torch.tensor(m), requires_grad=True)
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
        return 2

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([0]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.tensor([1]).long()

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
        m = self.m
        l = self.l
        mu = self.mu
        g = self.g

        mean = decode_mean(z, encoding)
        var = decode_var(z, encoding)

        theta = mean[..., 0]
        theta_dot = mean[..., 1]
        torque = u.flatten()

        # Define acceleration.
        temp = m * l
        theta_dot_dot = torque - mu * theta_dot - 0.5 * temp * g * theta.sin()
        theta_dot_dot = 3 * theta_dot_dot / (temp * l)

        mean = torch.stack(
            [
                theta + theta_dot.view(theta.shape) * dt,
                theta_dot + theta_dot_dot.view(theta_dot.shape) * dt,
            ],
            dim=-1)
        return encode(mean, V=var, encoding=encoding)
