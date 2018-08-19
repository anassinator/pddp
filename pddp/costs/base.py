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
"""Base cost functions."""

import torch
from ..models.utils.encoding import StateEncoding, decode_covar, decode_mean


class Cost(torch.nn.Module):

    """Base cost function."""

    def __add__(self, other):
        """Adds two cost functions together.

        Args:
            other (Cost or Tensor): Other cost function.

        Returns:
            Sum of costs.
        """
        return AggregateCost(self, other, torch.add)

    def __div__(self, other):
        """Divides two cost functions with each other.

        Args:
            other (Cost or Tensor): Other cost function.

        Returns:
            Quotient of costs.
        """
        return AggregateCost(self, other, torch.div)

    def __mul__(self, other):
        """Multiplies two cost functions together.

        Args:
            other (Cost or Tensor): Other cost function.

        Returns:
            Product of costs.
        """
        return AggregateCost(self, other, torch.mul)

    def __neg__(self):
        """Negates this cost.

        Returns:
            Opposite of this cost.
        """
        return AggregateCost(self, -1, torch.mul)

    def __pow__(self, other):
        """Takes the power of two cost functions.

        Args:
            other (Cost or Tensor): Other cost function.

        Returns:
            Power of costs.
        """
        return AggregateCost(self, other, torch.pow)

    def __sub__(self, other):
        """Subtracts two cost functions from each other.

        Args:
            other (Cost or Tensor): Other cost function.

        Returns:
            Difference of costs.
        """
        return AggregateCost(self, other, torch.sub)

    def __truediv__(self, other):
        """Divides two cost functions with each other with true division.

        Args:
            other (Cost or Tensor): Other cost function.

        Returns:
            Quotient of costs.
        """
        return AggregateCost(self, other, torch.div)

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
        raise NotImplementedError


class AggregateCost(Cost):

    """Aggregate cost function.

    This enables compounding two cost functions easily.

    Instantaneous cost:
        E[L(x, u)] = op(E[L_1(x, u)], E[L_2(x, u)])

    Terminal cost:
        E[L(x)] = op(E[L_1(x)], E[L_2(x)])
    """

    def __init__(self, first, second, op):
        """Constructs an AggregateCost.

        Args:
            first (Cost): First cost.
            second (Cost): Second cost.
            op (function): Operation to apply on both costs.
        """
        super(AggregateCost, self).__init__()
        self.first = first
        self.second = second
        self.op = op

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
        if isinstance(self.first, Cost):
            first = self.first(z, u, i, terminal, encoding, **kwargs)
        else:
            first = self.first

        if isinstance(self.second, Cost):
            second = self.second(z, u, i, terminal, encoding, **kwargs)
        else:
            second = self.second

        return self.op(first, second)
