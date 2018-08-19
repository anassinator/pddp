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
"""Dynamics model and cost evaluation utilities."""

import torch
from ...models import GaussianVariable
from ...models.utils.encoding import StateEncoding


def eval_cost(cost,
              z,
              u,
              i,
              terminal=False,
              encoding=StateEncoding.DEFAULT,
              approximate=False,
              **kwargs):
    """Evaluates the cost function and its first-order derivatives.

    Args:
        cost (Cost): Cost function.
        z (Tensor<encoded_state_size>): Encoded state vector.
        u (Tensor<action_size>): Action vector.
        i (int): Time index.
        terminal (bool): Whether the cost is terminal. If so, `u` should be
            `None`.
        encoding (int): StateEncoding enum.
        approximate (bool): Whether to approximate the Hessians or not.
        **kwargs: Additional key-word arguments to pass to `cost()`.

    Returns:
        Tuple of:
            Cost expectation (Tensor<0>).
            Partial derivative of the cost w.r.t. z
                (Tensor<encoded_state_size>).
            Partial derivative of the cost w.r.t. u (Tensor<action_size>).
            First-order derivative of the cost w.r.t. z
                (Tensor<encoded_state_size, encoded_state_size>).
            First-order derivative of the cost w.r.t. u
                (Tensor<encoded_state_size, action_size>).
            Second-order derivative of the cost w.r.t. z, z
                (Tensor<encoded_state_size, encoded_state_size>).
            Second-order derivative of the cost w.r.t. u, z
                (Tensor<action_size, encoded_state_size>).
            Second-order derivative of the cost w.r.t. u, u
                (Tensor<action_size, action_size>).
    """
    if approximate:
        l = cost(z, u, i, terminal=terminal, encoding=encoding, **kwargs)
        l_z, = torch.autograd.grad(l, z, retain_graph=True)
        l_zz = l_z.view(-1, 1).mm(l_z.view(1, -1))

        if terminal:
            l_u = None
            l_uz = None
            l_uu = None
        else:
            l_u, = torch.autograd.grad(l, u, retain_graph=True)
            l_uz = l_u.view(-1, 1).mm(l_z.view(1, -1))
            l_uu = l_u.view(-1, 1).mm(l_u.view(1, -1))

        return l, l_z, l_u, l_zz, l_uz, l_uu

    encoded_state_size = z.shape[-1]

    z_rep = z.repeat(encoded_state_size, 1)
    u_rep = u.repeat(encoded_state_size, 1) if not terminal else None

    l_rep = cost(
        z_rep, u_rep, i, terminal=terminal, encoding=encoding, **kwargs)
    l = l_rep[0]

    l_z_rep, = torch.autograd.grad(
        l_rep,
        z_rep,
        torch.ones(encoded_state_size, dtype=z.dtype),
        create_graph=True)
    l_z = l_z_rep[0]

    l_zz, = torch.autograd.grad(
        l_z_rep,
        z_rep,
        torch.eye(encoded_state_size, dtype=z.dtype),
        allow_unused=True,
        retain_graph=True)

    if terminal:
        l_u = None
        l_uz = None
        l_uu = None
    else:
        action_size = u.shape[-1]

        z_rep = z.repeat(action_size, 1)
        u_rep = u.repeat(action_size, 1) if not terminal else None
        l_rep = cost(
            z_rep, u_rep, i, terminal=terminal, encoding=encoding, **kwargs)

        l_u_rep, = torch.autograd.grad(
            l_rep,
            u_rep,
            torch.ones(action_size, dtype=z.dtype),
            create_graph=True)
        l_u = l_u_rep[0]

        l_uz, = torch.autograd.grad(
            l_u_rep,
            z_rep,
            torch.eye(action_size, dtype=z.dtype),
            allow_unused=True,
            retain_graph=True)

        if l_uz is None:
            l_uz = torch.zeros(action_size, encoded_state_size, dtype=z.dtype)

        l_uu, = torch.autograd.grad(
            l_u_rep,
            u_rep,
            torch.eye(action_size, dtype=z.dtype),
            allow_unused=True,
            retain_graph=True)

    return l, l_z, l_u, l_zz, l_uz, l_uu


def eval_dynamics(model, z, u, i, encoding=StateEncoding.DEFAULT, **kwargs):
    """Evaluates the dynamics model and its first-order derivatives.

    Args:
        model (DynamicsModel): Dynamics model.
        z (Tensor<encoded_state_size>): Encoded state vector.
        u (Tensor<action_size>): Action vector.
        i (int): Time index.
        **kwargs: Additional key-word arguments to pass to `model()`.

    Returns:
        Tuple of:
            Encoded next state vector (Tensor<encoded_state_size>).
            Jacobian of the model w.r.t. z
                (Tensor<encoded_state_size, encoded_state_size>).
            Jacobian of the model w.r.t. u
                (Tensor<encoded_state_size, action_size>).
    """
    encoded_state_size = z.shape[-1]
    action_size = u.shape[-1]

    z_rep = z.repeat(encoded_state_size, 1)
    u_rep = u.repeat(encoded_state_size, 1)
    z_next_rep = model(z_rep, u_rep, i, encoding, **kwargs)

    # Parallelized jacobian.
    d_dz, = torch.autograd.grad(
        z_next_rep,
        z_rep,
        torch.eye(encoded_state_size, dtype=z.dtype),
        allow_unused=True,
        retain_graph=True)
    d_du, = torch.autograd.grad(
        z_next_rep,
        u_rep,
        torch.eye(encoded_state_size, dtype=z.dtype),
        allow_unused=True,
        retain_graph=True)

    z_next = z_next_rep[0]
    return z_next, d_dz, d_du
