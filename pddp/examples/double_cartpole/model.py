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
"""Double cartpole dynamics model."""

import torch

from torch.nn import Parameter

from ...models.base import DynamicsModel
from ...utils.classproperty import classproperty
from ...utils.angular import augment_state, reduce_state
from ...utils.encoding import StateEncoding, decode_var, decode_mean, encode


class DoubleCartpoleDynamicsModel(DynamicsModel):

    """Double cartpole dynamics model.

    Note:
        state: [x, x', theta1, theta1', theta2, theta2']
        action: [F]
        theta: 0 is pointing up and increasing clockwise.
    """

    def __init__(self,
                 dt,
                 mc=0.5,
                 mp1=0.5,
                 mp2=0.5,
                 l1=0.6,
                 l2=0.6,
                 mu=0.1,
                 g=9.80665):
        """Constructs a DoubleCartpoleDynamicsModel.

        Args:
            dt (float): Time step [s].
            mc (float): Cart mass [kg].
            mp1 (float): First link mass [kg].
            mp2 (float): Second link mass [kg].
            l1 (float): First link length [m].
            l2 (float): Second link length [m].
            mu (float): Coefficient of friction [dimensionless].
            g (float): Gravity acceleration [m/s^2].
        """
        super(DoubleCartpoleDynamicsModel, self).__init__()
        self.dt = Parameter(torch.tensor(dt), requires_grad=False)
        self.mc = Parameter(torch.tensor(mc), requires_grad=True)
        self.mp1 = Parameter(torch.tensor(mp1), requires_grad=True)
        self.mp2 = Parameter(torch.tensor(mp2), requires_grad=True)
        self.l1 = Parameter(torch.tensor(l1), requires_grad=True)
        self.l2 = Parameter(torch.tensor(l2), requires_grad=True)
        self.mu = Parameter(torch.tensor(mu), requires_grad=True)
        self.g = Parameter(torch.tensor(g), requires_grad=True)

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        return 1

    @classproperty
    def state_size(cls):
        """State size (int)."""
        return 6

    @classproperty
    def angular_indices(cls):
        """Column indices of angular states (Tensor)."""
        return torch.tensor([2, 4]).long()

    @classproperty
    def non_angular_indices(cls):
        """Column indices of non-angular states (Tensor)."""
        return torch.tensor([0, 1, 3, 5]).long()

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
        mc = self.mc if z.dim() == 1 else self.mc.repeat(z.shape[0])
        mp1 = self.mp1 if z.dim() == 1 else self.mp1.repeat(z.shape[0])
        mp2 = self.mp2 if z.dim() == 1 else self.mp2.repeat(z.shape[0])
        l1 = self.l1 if z.dim() == 1 else self.l1.repeat(z.shape[0])
        l2 = self.l2 if z.dim() == 1 else self.l2.repeat(z.shape[0])
        mu = self.mu if z.dim() == 1 else self.mu.repeat(z.shape[0])
        g = self.g if z.dim() == 1 else self.g.repeat(z.shape[0])

        mean = decode_mean(z, encoding)
        var = decode_var(z, encoding)

        x = mean[..., 0]
        x_dot = mean[..., 1]
        theta1 = mean[..., 2]
        theta1_dot = mean[..., 3]
        theta2 = mean[..., 4]
        theta2_dot = mean[..., 5]
        F = u[..., 0]

        sin_theta1 = theta1.sin()
        cos_theta1 = theta1.cos()

        sin_theta2 = theta2.sin()
        cos_theta2 = theta2.cos()

        sin_dtheta = (theta1 - theta2).sin()
        cos_dtheta = (theta1 - theta2).cos()

        a0 = mp2 + 2 * mc
        a1 = mc * l2
        a2 = l1 * theta1_dot**2
        a3 = a1 * theta2_dot**2

        # yapf: disable
        A = torch.stack([
            torch.stack([
                2 * (mp1 + mp2 + mc),
                -a0 * l1 * cos_theta1,
                -a1 * cos_theta2
            ], dim=-1),
            torch.stack([
                -3 * a0 * cos_theta1,
                (2 * a0 + 2 * mc) * l1,
                3 * a1 * cos_dtheta
            ], dim=-1),
            torch.stack([
                -3 * cos_theta2,
                3 * l1 * cos_dtheta,
                2 * l2
            ], dim=-1),
        ], dim=-1).transpose(-2, -1)
        b = torch.stack([
            torch.stack([
                2 * F - 2 * mu * x_dot - a0 * a2 * sin_theta1 - a3 * sin_theta2
            ], dim=-1),
            torch.stack([
                3 * a0 * g * sin_theta1 - 3 * a3 * sin_dtheta
            ], dim=-1),
            torch.stack([
                3 * a2 * sin_dtheta + 3 * g * sin_theta2
            ], dim=-1),
        ], dim=-1).transpose(-2, -1)
        # yapf: enable

        sol = torch.gesv(b, A)[0].transpose(-2, -1)

        # For symplectic integration.
        new_x_dot = x_dot + sol[..., 0].view(x_dot.shape) * dt
        new_theta1_dot = theta1_dot + sol[..., 1].view(theta1_dot.shape) * dt
        new_theta2_dot = theta2_dot + sol[..., 2].view(theta2_dot.shape) * dt

        mean = torch.stack(
            [
                x + new_x_dot * dt,
                new_x_dot,
                theta1 + new_theta1_dot * dt,
                new_theta1_dot,
                theta2 + new_theta2_dot * dt,
                new_theta2_dot,
            ],
            dim=-1)

        return encode(mean, V=var, encoding=encoding)