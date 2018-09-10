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
"""Gaussian process kernels."""

import torch


class Kernel(torch.nn.Module):

    """Base kernel."""

    def __add__(self, other):
        """Sums two kernels together.
        Args:
            other (Kernel): Other kernel.
        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.add)

    def __mul__(self, other):
        """Multiplies two kernels together.
        Args:
            other (Kernel): Other kernel.
        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.mul)

    def __sub__(self, other):
        """Subtracts two kernels from each other.
        Args:
            other (Kernel): Other kernel.
        Returns:
            AggregateKernel.
        """
        return AggregateKernel(self, other, torch.sub)

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        raise NotImplementedError


class AggregateKernel(Kernel):

    """An aggregate kernel."""

    def __init__(self, first, second, op):
        """Constructs an AggregateKernel.
        Args:
            first (Kernel): First kernel.
            second (Kernel): Second kernel.
            op (Function): Operation to apply.
        """
        super(Kernel, self).__init__()
        self.first = first
        self.second = second
        self.op = op

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        first = self.first(xi, xj, *args, **kwargs)
        second = self.second(xi, xj, *args, **kwargs)
        return self.op(first, second)


class RBFKernel(Kernel):

    """Radial-Basis Function Kernel."""

    def __init__(self, log_length_scale, log_sigma_s=None):
        """Constructs an RBFKernel.
        Args:
            log_length_scale (Tensor): Inverse log length scale.
            log_sigma_s (Tensor): Log signal standard deviation.
        """
        super(Kernel, self).__init__()
        self.log_length_scale = torch.nn.Parameter(log_length_scale)
        self.log_sigma_s = torch.nn.Parameter(
            torch.randn(1) if log_sigma_s is None else log_sigma_s)

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        length_scale = (-2 * self.log_length_scale).exp()

        # TODO: Remove for-loop.
        M = torch.stack([l.diag() for l in length_scale])
        var_s = (2 * self.log_sigma_s).exp()
        dist = mahalanobis_squared(xi, xj, M)
        return var_s[:, None, None] * (-0.5 * dist).exp()


class WhiteNoiseKernel(Kernel):

    """White noise kernel."""

    def __init__(self, log_sigma_n=None, eps=1e-6):
        """Constructs a WhiteNoiseKernel.
        Args:
            log_sigma_n (Tensor): Log noise standard deviation.
        """
        super(Kernel, self).__init__()
        self.log_sigma_n = torch.nn.Parameter(
            torch.randn(1) if log_sigma_n is None else log_sigma_n)

    def forward(self, xi, xj, *args, **kwargs):
        """Covariance function.
        Args:
            xi (Tensor): First matrix.
            xj (Tensor): Second matrix.
        Returns:
            Covariance (Tensor).
        """
        var_n = (2 * self.log_sigma_n).exp()
        return var_n[:, None, None]


def mahalanobis_squared(xi, xj, VI=None):
    """Computes the pair-wise squared mahalanobis distance matrix as:
        (xi - xj)^T V^-1 (xi - xj)
    Args:
        xi (Tensor): xi input matrix.
        xj (Tensor): xj input matrix.
        VI (Tensor): The inverse of the covariance matrix, default: identity
            matrix.
    Returns:
        Weighted matrix of all pair-wise distances (Tensor).
    """
    if xi.dim() == 2:
        if VI is None:
            xi_VI = xi
            xj_VI = xj
        else:
            xi_VI = xi.mm(VI)
            xj_VI = xj.mm(VI)

        D = (xi_VI * xi).sum(dim=-1).unsqueeze(0).t() \
          + (xj_VI * xj).sum(dim=-1).unsqueeze(0) \
          - 2 * xi_VI.mm(xj.t())
    else:
        if VI is None:
            xi_VI = xi
            xj_VI = xj
        else:
            xi_VI = xi.bmm(VI)
            xj_VI = xj.bmm(VI)

        D = (xi_VI * xi).sum(dim=-1).unsqueeze(-2).transpose(-2, -1) \
          + (xj_VI * xj).sum(dim=-1).unsqueeze(-2) \
          - 2 * xi_VI.bmm(xj.transpose(-2, -1))
    return D