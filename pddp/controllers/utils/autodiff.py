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
"""Auto-differentiation utilities."""

import torch


def grad(y, x, **kwargs):
    """Evaluates the gradient of y w.r.t x safely.

    Args:
        y (Tensor<0>): Tensor to differentiate.
        x (Tensor<n>): Tensor to differentiate with respect to.
        **kwargs: Additional key-word arguments to pass to
            `torch.autograd.grad()`.

    Returns:
        Gradient (Tensor<n>).
    """
    dy_dx, = torch.autograd.grad(
        y, x, retain_graph=True, allow_unused=True, **kwargs)

    # The gradient is None if disconnected.
    dy_dx = dy_dx if dy_dx is not None else torch.zeros_like(x)
    dy_dx.requires_grad_()

    return dy_dx


def jacobian(y, x, **kwargs):
    """Evaluates the jacobian of y w.r.t x safely.

    Args:
        y (Tensor<m>): Tensor to differentiate.
        x (Tensor<n>): Tensor to differentiate with respect to.
        **kwargs: Additional key-word arguments to pass to `grad()`.

    Returns:
        Jacobian (Tensor<m, n>).
    """
    J = [grad(y[i], x, **kwargs) for i in range(y.shape[0])]
    J = torch.stack(J)
    J.requires_grad_()
    return J


def batch_jacobian(f, x, m=None, **kwargs):
    """Compute the jacobian of f(x) w.r.t. x in batch.

    Args:
        f (Tensor<..., n> -> Tensor<..., m>): Function to differentiate.
        x (Tensor<n>): Tensor to differentiate with respect to.
        m (int): Expected output dimensions.

    Returns:
        Jacobian (Tensor<m, n>).
    """
    if m is None:
        y = f(x)
        m = y.shape[-1]

    x_rep = x.repeat(m, 1)
    x_rep = torch.tensor(x_rep, requires_grad=True)
    y_rep = f(x_rep)

    dy_dx, = torch.autograd.grad(
        y_rep,
        x_rep,
        torch.eye(m, dtype=x.dtype),
        allow_unused=True,
        retain_graph=True,
        **kwargs)

    if dy_dx is None:
        return torch.zeros(m, x.shape[-1], dtype=x.dtype, requires_grad=True)

    return dy_dx
