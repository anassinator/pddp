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
"""Gaussian random variable."""

import torch
from .encoding import (StateEncoding, encode, decode_covar, decode_mean,
                       decode_var, decode_std)


class GaussianVariable(object):

    """Lazily-evaluated multivariate gaussian random variable."""

    def __init__(self, mean, covar=None, var=None, std=None):
        """Constructs a GaussianVariable<n> instance.

        Note: At least one of covar, var and std must be provided. The remaining
        will be computed automatically lazily.

        Args:
            mean (Tensor<n>): Mean column vector.
            covar (Tensor<n, n>): Full covariance matrix.
            var (Tensor<n>): Variance vector.
            std (Tensor<n>): Standard deviation vector.
        """
        self._mean = mean
        self._covar = covar
        self._var = var
        self._std = std
        self._is_from_covar = covar is not None

    def __repr__(self):
        """String representation of GaussianVariable."""
        return "GaussianVariable({})".format(self.shape)

    @property
    def device(self):
        """Gaussian variable device."""
        return self._mean.device

    @property
    def dtype(self):
        """Gaussian variable dtype."""
        return self._mean.dtype

    @property
    def requires_grad(self):
        """Whether the gaussian variable requires gradients."""
        return self._mean.requires_grad

    @property
    def shape(self):
        """Gaussian variable mean shape."""
        return self._mean.shape

    def mean(self):
        """Mean column vector (Tensor<n>)."""
        return self._mean

    def covar(self):
        """Full covariance matrix (Tensor<n, n>)."""
        if self._covar is None:
            if self._var is not None:
                self._covar = torch.diag(self._var)
            elif self._std is not None:
                self._covar = torch.diag(self._std.pow(2))
            else:
                raise NotImplementedError("Cannot compute covariance")

        return self._covar

    def var(self):
        """Variance vector (Tensor<n>)."""
        if self._var is None:
            if self._covar is not None:
                self._var = self._covar.diag()
            elif self._std is not None:
                self._var = self._std.pow(2)
            else:
                raise NotImplementedError("Cannot compute variance")

        return self._var

    def std(self):
        """Standard deviation vector (Tensor<n>)."""
        if self._std is None:
            if self._var is not None:
                self._std = self._var.sqrt()
            elif self._covar is not None:
                self._std = self._covar.diag().sqrt()
            else:
                raise NotImplementedError("Cannot compute standard deviation")

        return self._std

    def sample(self, sample_shape=torch.Size([])):
        """Generates a sample_shape shaped sample.

        Args:
            sample_shape (Size): Shape of sample.

        Returns:
            Sampled state (Tensor<state_size> or
            Tensor<sample_shape, state_size>).
        """
        if self._is_from_covar:
            n = torch.distributions.MultivariateNormal(self.mean(),
                                                       self.covar())
        else:
            n = torch.distributions.Normal(self.mean(), self.std())
        return n.sample(sample_shape)

    def encode(self, encoding=StateEncoding.DEFAULT):
        """Encodes itself to a vector.

        Args:
            encoding (int): StateEncoding enum.

        Returns:
            Encoded state vector (Tensor<encoded_state_size>).
        """
        if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
            return encode(self.mean(), C=self.covar(), encoding=encoding)
        elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
            return encode(self.mean(), C=self.covar(), encoding=encoding)
        elif encoding == StateEncoding.VARIANCE_ONLY:
            return encode(self.mean(), V=self.var(), encoding=encoding)
        elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
            return encode(self.mean(), S=self.std(), encoding=encoding)
        elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
            return encode(self.mean(), V=self.var(), encoding=encoding)

        raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))

    @classmethod
    def decode(cls, z, encoding=StateEncoding.DEFAULT, state_size=None):
        """Computes the original state distribution from a z-encoded state.

        Args:
            z (Tensor<encoded_state_size>): Encoded state vector.
            encoding (int): StateEncoding enum.
            state_size (int): Original state size, default: inferred.

        Returns:
            State distribution (GaussianVariable<state_size>).
        """
        mean = decode_mean(z, encoding, state_size)
        if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
            covar = decode_covar(z, encoding, state_size)
            return GaussianVariable(mean, covar=covar)
        elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
            covar = decode_covar(z, encoding, state_size)
            return GaussianVariable(mean, covar=covar)
        elif encoding == StateEncoding.VARIANCE_ONLY:
            var = decode_var(z, encoding, state_size)
            return GaussianVariable(mean, var=var)
        elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
            std = decode_std(z, encoding, state_size)
            return GaussianVariable(mean, std=std)
        elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
            var = decode_var(z, encoding, state_size)
            return GaussianVariable(mean, var=var)

        raise NotImplementedError("Unknown StateEncoding: {}".format(encoding))

    def detach(self):
        """Returns a new GaussianVariable detached from the current graph."""
        other = self.clone()
        other._mean = other._mean.detach()
        if other._covar is not None:
            other._covar = other._covar.detach()
        if other._var is not None:
            other._var = other._var.detach()
        if other._std is not None:
            other._std = other._std.detach()
        return other

    def requires_grad_(self, requires_grad=True):
        """Change if autograd should record operations on this tensor: sets this
        tensor's `requires_grad` attribute in-place. Returns this variable."""
        self._mean.requires_grad_(requires_grad)
        if self._covar is not None:
            self._covar.requires_grad_(requires_grad)
        if self._var is not None:
            self._var.requires_grad_(requires_grad)
        if self._std is not None:
            self._std.requires_grad_(requires_grad)
        return self

    def clone(self):
        """Clones the GaussianVariable.

        Returns:
            GaussianVariable.
        """
        mean = self._mean.clone()
        covar = self._covar.clone() if self._covar is not None else None
        var = self._var.clone() if self._var is not None else None
        std = self._std.clone() if self._std is not None else None
        return GaussianVariable(mean, covar, var, std)

    def to(self, *args, **kwargs):
        """Performs Tensor dtype and/or device conversion to all internal
        tensors.

        Returns:
            GaussianVariable.
        """
        other._mean = other._mean.to(*args, **kwargs)
        if other._covar is not None:
            other._covar = other._covar.to(*args, **kwargs)
        if other._var is not None:
            other._var = other._var.to(*args, **kwargs)
        if other._std is not None:
            other._std = other._std.to(*args, **kwargs)
        return other

    def cpu(self):
        """Returns a copy of this object in CPU memory."""
        return self.to(device="cpu")

    def cuda(self, *args, **kwargs):
        """Returns a copy of this object in CUDA memory."""
        other = self.clone()
        other._mean = other._mean.cuda(*args, **kwargs)
        if other._covar is not None:
            other._covar = other._covar.cuda(*args, **kwargs)
        if other._var is not None:
            other._var = other._var.cuda(*args, **kwargs)
        if other._std is not None:
            other._std = other._std.cuda(*args, **kwargs)
        return other

    def double(self):
        """Equivalent to `self.to(torch.float64)`."""
        return self.to(torch.float64)

    def float(self):
        """Equivalent to `self.to(torch.float32)`."""
        return self.to(torch.float32)

    def half(self):
        """Equivalent to `self.to(torch.float16)`."""
        return self.to(torch.float16)

    @classmethod
    def random(cls, n, reg=1e-1, requires_grad=True):
        """Constructs a random valid GaussianVariable of size n.

        Args:
            n (int): Size of the mean.
            reg (float): Regularization term to guarantee positive-definiteness
                of the covariance matrix.
            requires_grad (bool): Whether a gradient is required in underlying
                tensors.

        Returns:
            GaussianVariable<n>.
        """
        mean = torch.randn(n, requires_grad=requires_grad)
        L = torch.randn(n, n, requires_grad=requires_grad)
        covar = L.t().mm(L) + reg * torch.eye(n)
        return GaussianVariable(mean, covar=covar)
