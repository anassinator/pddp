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
from ...utils.encoding import StateEncoding, decode_covar, decode_mean, encode
from ...utils.angular import augment_state, complementary_indices, reduce_state


@constrain_model(-1, 1)
class PendulumDynamicsModel(DynamicsModel):

    """Friction-less pendulum dynamics model.

    Note:
        state: augment_state([theta, theta']) = [theta', sin(theta), cos(theta)]
        action: [torque]
        theta: 0 is pointing up and increasing counter-clockwise.
    """

    def __init__(self, dt, m=1.0, l=0.5, g=9.80665):
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
        self.g = Parameter(torch.tensor(g), requires_grad=True)

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        return 1

    @classproperty
    def state_size(cls):
        """Augmented state size (int)."""
        return 3

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([0]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.tensor([1]).long()

    def fit(self, dataset, quiet=False, **kwargs):
        """Fits the dynamics model.

        Args:
            dataset
                (Dataset<Tensor<N, state_size + action_size>,
                         Tensor<N, action_size>>):
                Dataset of state-action pair trajectory and next state
                trajectory.
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
        g = self.g

        mean = decode_mean(z, encoding)
        covar = decode_covar(z, encoding)
        state = reduce_state(mean, self.angular_indices,
                             self.non_angular_indices)

        theta = state[..., 0]
        theta_dot = state[..., 1]
        torque = u.flatten()

        # Define acceleration.
        theta_dot_dot = (-3.0 * g / (2 * l) * (theta + np.pi).sin() +
                         3.0 / (m * l**2) * torque)

        # Constrain the angular velocity.
        next_theta_dot = theta_dot + theta_dot_dot.view(theta_dot.shape) * dt

        mean = torch.stack(
            [
                theta + theta_dot.view(theta.shape) * dt,
                next_theta_dot,
            ], dim=-1)
        mean_ = augment_state(mean, self.angular_indices,
                              self.non_angular_indices)
        return encode(mean_, C=covar, encoding=encoding)
