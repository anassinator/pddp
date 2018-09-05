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
"""Quadratic cost functions."""

import torch
import numpy as np

from .base import Cost
from ..utils.encoding import StateEncoding, decode_covar, decode_mean


class QRCost(Cost):

    r"""Quadratic cost function.

    Instantaneous cost:
        E[L(x, u)] = tr(Q \Sigma)
                   + (\mu - x_goal)^T Q (\mu - x_goal)
                   + (u - u_goal)^T R (u - u_goal)

    Terminal cost:
        E[L(x)] = tr(Q \Sigma) + (\mu - x_goal)^T Q (\mu - x_goal)
    """

    def __init__(self,
                 Q,
                 R,
                 Q_term=None,
                 x_goal=torch.tensor(0.0),
                 u_goal=torch.tensor(0.0)):
        """Constructs a QRCost.

        Args:
            Q (Tensor<state_size, state_size>): Q matrix.
            R (Tensor<action_size, action_size>): R matrix.
            Q_term (Tensor<state_size, state_size>): Terminal Q matrix,
                default: Q.
            x_goal (Tensor<state_size>): Goal state, default: 0.0.
            u_goal (Tensor<action_size>): Goal action, default: 0.0.
        """
        super(QRCost, self).__init__()
        Q_term = Q if Q_term is None else Q_term

        self.Q = torch.nn.Parameter(Q, requires_grad=False)
        self.R = torch.nn.Parameter(R, requires_grad=False)
        self.Q_term = torch.nn.Parameter(Q_term, requires_grad=False)

        self.x_goal = torch.nn.Parameter(x_goal, requires_grad=False)
        self.u_goal = torch.nn.Parameter(u_goal, requires_grad=False)

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
        Q = self.Q_term if terminal else self.Q

        mean = decode_mean(z, encoding)
        x_ = mean - self.x_goal
        y = x_.matmul(Q).matmul(x_.t() if x_.dim() > 1 else x_)
        cost = y.diag() if x_.dim() > 1 else y

        if not terminal:
            u_ = (u - self.u_goal)
            v = u_.matmul(self.R).matmul(u_.t() if u_.dim() > 1 else u_)
            cost += v.diag() if u_.dim() > 1 else v

        if encoding != StateEncoding.IGNORE_UNCERTAINTY:
            # Batch compute trace.
            covar = decode_covar(z, encoding)
            Y = Q.matmul(covar)
            ix, iy = np.diag_indices(covar.shape[-1])
            cost += Y[..., ix, iy].sum(dim=-1)

        return cost
