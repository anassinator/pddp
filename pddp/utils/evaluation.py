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
from .encoding import StateEncoding
from .gaussian_variable import GaussianVariable


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
    encoded_state_size = z.shape[-1]
    action_size = u.shape[-1] if not terminal else 0
    zu = torch.cat([z, u], -1) if not terminal else z

    if approximate:
        l = cost(
            zu[:encoded_state_size],
            zu[encoded_state_size:],
            i,
            terminal=terminal,
            encoding=encoding,
            **kwargs)
        l_zu, = torch.autograd.grad(l, zu, retain_graph=False)
        l_z = l_zu[:encoded_state_size].detach()
        l_u = l_zu[encoded_state_size:].detach()
        l_zz = l_z.view(-1, 1).mm(l_z.view(1, -1))

        if terminal:
            l_u = None
            l_uz = None
            l_uu = None
        else:
            l_uz = l_u.view(-1, 1).mm(l_z.view(1, -1))
            l_uu = l_u.view(-1, 1).mm(l_u.view(1, -1))

        cost.zero_grad()

        return l, l_z, l_u, l_zz, l_uz, l_uu

    tensor_opts = {"device": z.device, "dtype": z.dtype}

    zu_rep = zu.repeat(encoded_state_size + action_size, 1)
    z_rep = zu_rep[:, :encoded_state_size]
    u_rep = zu_rep[:, encoded_state_size:] if not terminal else None

    l_rep = cost(
        z_rep, u_rep, i, terminal=terminal, encoding=encoding, **kwargs)
    l = l_rep[0]

    l_zu_rep, = torch.autograd.grad(
        l_rep,
        zu_rep,
        torch.ones(zu.shape[0], **tensor_opts),
        create_graph=True)
    l_z = l_zu_rep[0, :encoded_state_size].detach()

    l_zuzu, = torch.autograd.grad(
        l_zu_rep,
        zu_rep,
        torch.eye(zu.shape[0], **tensor_opts),
        allow_unused=True)
    l_zz = l_zuzu[:encoded_state_size, :encoded_state_size].detach()

    if terminal:
        l_u = None
        l_uz = None
        l_uu = None
    else:
        l_u = l_zu_rep[0, encoded_state_size:].detach()
        l_uu = l_zuzu[encoded_state_size:, encoded_state_size:].detach()
        l_uz = l_zuzu[encoded_state_size:, :encoded_state_size].detach()

    cost.zero_grad()

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
    tensor_opts = {"device": z.device, "dtype": z.dtype}

    zu = torch.cat([z, u], -1)
    zu_rep = zu.expand(encoded_state_size, -1)
    z_next_rep = model(zu_rep[:, :encoded_state_size],
                       zu_rep[:, encoded_state_size:], i, encoding, **kwargs)

    # Parallelized jacobian.
    d_dzu, = torch.autograd.grad(
        z_next_rep,
        zu_rep,
        torch.eye(encoded_state_size, **tensor_opts),
        allow_unused=True,
        retain_graph=False)

    z_next = z_next_rep[0].detach()
    d_dz = d_dzu[:, :encoded_state_size].detach()
    d_du = d_dzu[:, encoded_state_size:].detach()

    model.zero_grad()

    return z_next, d_dz, d_du