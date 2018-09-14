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
"""Base dynamics model."""

import torch

from enum import IntEnum

from ..utils.encoding import (StateEncoding, decode_covar, decode_var,
                              decode_mean, encode)
from ..utils.classproperty import classproperty


class Integrator(IntEnum):
    FW_EULER = 0
    MIDPOINT = 1
    RUNGE_KUTTA = 2


class DynamicsModel(torch.nn.Module):

    """Base dynamics model."""

    def reset_parameters(self, initializer=torch.nn.init.normal_):
        """Resets all parameters that require gradients with random values.

        Args:
            initializer (callable): In-place function to initialize module
                parameters.

        Returns:
            self.
        """
        for p in self.parameters():
            if p.requires_grad:
                initializer(p)
        return self

    @classproperty
    def action_size(cls):
        """Action size (int)."""
        raise NotImplementedError

    @classproperty
    def state_size(cls):
        """State size (int)."""
        raise NotImplementedError

    def fit(self, X, U, dX, quiet=False, **kwargs):
        """Fits the dynamics model.

        Args:
            X (Tensor<N, state_size>): State trajectory.
            U (Tensor<N, action_size>): Action trajectory.
            dX (Tensor<N, state_size>): Next state trajectory.
            quiet (bool): Whether to print anything to screen or not.
        """
        raise NotImplementedError

    def dynamics(self, z, u, i, **kwargs):
        """Dynamics model function.

        Args:
            z (Tensor<..., encoded_state_size>): Encoded state distribution.
            u (Tensor<..., action_size>): Action vector(s).
            i (Tensor<...>): Time index.

        Returns:
            derivatives of current state wrt to time (Tensor<..., encoded_state_size>).
        """
        raise NotImplementedError

    def forward(self,
                z,
                u,
                i,
                encoding=StateEncoding.DEFAULT,
                int_method=Integrator.RUNGE_KUTTA,
                **kwargs):
        """Dynamics model function.

        Args:
            z (Tensor<..., encoded_state_size>): Encoded state distribution.
            u (Tensor<..., action_size>): Action vector(s).
            i (int): Time index.
            encoding (int): StateEncoding enum.

        Returns:
            Next encoded state distribution (Tensor<..., encoded_state_size>).
        """
        mean = decode_mean(z, encoding)
        var = decode_var(z, encoding)
        C = decode_covar(z, encoding)

        if int_method == Integrator.FW_EULER:
            dmean = self.dynamics(mean, u, i)
            mean = mean + dmean * self.dt
        elif int_method == Integrator.MIDPOINT:
            dmean = self.dynamics(mean, u, i)
            mid = mean + dmean * self.dt / 2
            dmid = self.dynamics(mid, u, i)
            mean = mean + dmid * self.dt
        elif int_method == Integrator.RUNGE_KUTTA:
            d1 = self.dynamics(mean, u, i)
            d2 = self.dynamics(mean + d1 * self.dt / 2, u, i)
            d3 = self.dynamics(mean + d2 * self.dt / 2, u, i)
            d4 = self.dynamics(mean + d3 * self.dt, u, i)
            mean = mean + (d1 + 2 * d2 + 2 * d3 + d4) * (self.dt / 6)

        try:
            return encode(mean, C=C, encoding=encoding)
        except RuntimeError:
            return encode(mean, var=var, encoding=encoding)
