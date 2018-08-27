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
from ...utils.classproperty import classproperty
from ...utils.encoding import StateEncoding, decode_covar, decode_mean, encode
from ...utils.angular import augment_state, reduce_state


class CartpoleDynamicsModel(DynamicsModel):

    """Friction-less cartpole dynamics model.

    Note:
        state: augment_state([x, x', theta, theta'])
               = [x, x', theta', sin(theta), cos(theta)]
        action: [F]
        theta: 0 is pointing up and increasing clockwise.
    """

    def __init__(self, dt, mc=1.0, mp=0.1, l=1.0, g=9.80665):
        """Constructs a CartpoleDynamicsModel.

        Args:
            dt (float): Time step [s].
            mc (float): Cart mass [kg].
            mp (float): Pendulum mass [kg].
            l (float): Pendulum length [m].
            g (float): Gravity acceleration [m/s^2].
        """
        super(CartpoleDynamicsModel, self).__init__()
        self.dt = Parameter(torch.tensor(dt), requires_grad=False)
        self.mc = Parameter(torch.tensor(mc), requires_grad=True)
        self.mp = Parameter(torch.tensor(mp), requires_grad=True)
        self.l = Parameter(torch.tensor(l), requires_grad=True)
        self.g = Parameter(torch.tensor(g), requires_grad=True)

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        return 1

    @classproperty
    def state_size(cls):
        """Augmented state size (int)."""
        return 5

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([2]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.tensor([0, 1, 3]).long()

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
        mc = self.mc
        mp = self.mp
        l = self.l
        g = self.g

        mean = decode_mean(z, encoding)
        covar = decode_covar(z, encoding)

        state = reduce_state(mean, self.angular_indices,
                             self.non_angular_indices)
        x = state[..., 0]
        x_dot = state[..., 1]
        theta = state[..., 2]
        theta_dot = state[..., 3]
        F = u.flatten()

        # Define dynamics model as per Razvan V. Florian's
        # "Correct equations for the dynamics of the cart-pole system".
        # Friction is neglected.

        # Eq. (23)
        sin_theta = theta.sin()
        cos_theta = theta.cos()
        temp = (F + mp * l * theta_dot**2 * sin_theta) / (mc + mp)
        numerator = g * sin_theta - cos_theta * temp
        denominator = l * (4.0 / 3.0 - mp * cos_theta**2 / (mc + mp))
        theta_dot_dot = numerator / denominator

        # Eq. (24)
        x_dot_dot = temp - mp * l * theta_dot_dot * cos_theta / (mc + mp)

        mean = torch.stack(
            [
                x + x_dot * dt,
                x_dot + x_dot_dot.view(x_dot.shape) * dt,
                theta + theta_dot * dt,
                theta_dot + theta_dot_dot.view(theta_dot.shape) * dt,
            ],
            dim=-1)
        mean_ = augment_state(mean, self.angular_indices,
                              self.non_angular_indices)
        return encode(mean_, C=covar, encoding=encoding)
