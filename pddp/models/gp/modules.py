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
"""Gaussian process modules."""

import torch
import warnings

import numpy as np

with warnings.catch_warnings():
    # Ignore potential warning when in Jupyter environment.
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange

from .kernels import RBFKernel

from ..base import DynamicsModel

from ...utils.constraint import constrain
from ...utils.angular import augment_state
from ...utils.classproperty import classproperty
from ...utils.encoding import StateEncoding, decode_mean, encode


class GaussianProcess(torch.nn.Module):

    """Gaussian process regressor."""

    def __init__(self, kernel, log_sigma_n=None):
        """Constructs a GaussianProcess.
        Args:
            kernel (Kernel): Kernel.
            log_sigma_n (Tensor): Log noise standard deviation.
        """
        super(GaussianProcess, self).__init__()
        self.kernel = kernel
        self.log_sigma_n = torch.nn.Parameter(
            torch.randn(1) if log_sigma_n is None else log_sigma_n)
        self._is_set = False
        self._reg = 1e-12

    def _update_k(self):
        """Updates the K matrix."""
        K = self.kernel(self.X, self.X)
        K = 0.5 * (K + K.transpose(-2, -1))
        self.register_buffer("K", K)

    def _update_k_inv(self):
        """Updates the inverse K matrix."""
        var_n = (2 * self.log_sigma_n).exp().clamp(1e-5, 1e5)
        K = self.K + var_n[:, None, None] * torch.eye(self.K.shape[-1])

        # TODO: Use batch potrf().
        self.register_buffer("L", torch.stack([k.potrf() for k in K]))
        self.register_buffer("K_inv", torch.stack([k.inverse() for k in K]))

    def increase_reg(self):
        self._reg *= 10

    def set_data(self, X, Y, normalize_y=True):
        """Set the training data.
        Args:
            X (Tensor): Training inputs.
            Y (Tensor): Training outputs.
            normalize_y (bool): Whether to normalize the outputs.
        """
        Y = Y.unsqueeze(0).permute(2, 1, 0)
        self.register_buffer("non_normalized_Y", Y)

        if normalize_y:
            Y_mean = Y.mean(dim=0)
            Y_std = Y.std(dim=0)
            Y = (Y - Y_mean) / Y_std

        self.register_buffer("X", X.expand(Y.shape[0], -1, -1))
        self.register_buffer("Y", Y)
        self._update_k()
        self._is_set = True

    def loss(self):
        """Computes the loss as the negative marginal log likelihood."""
        if not self._is_set:
            raise RuntimeError("You must call set_data() first")

        Y = self.Y
        self._update_k_inv()
        K_inv = self.K_inv

        # Compute the log likelihood.
        ix, iy = np.diag_indices(self.L.shape[-1])
        L_diag = self.L[..., ix, iy]
        log_likelihood_dims = -0.5 * Y.transpose(-2, -1).bmm(K_inv).bmm(Y).sum()
        log_likelihood_dims -= L_diag.log().sum()
        log_likelihood_dims -= (
            self.L.shape[0] * self.L.shape[-1] / 2.0 * np.log(2 * np.pi))
        log_likelihood = log_likelihood_dims.sum()

        return -log_likelihood

    def forward(self, x, return_mean=True, return_var=False, **kwargs):
        """Computes the GP estimate.
        Args:
            x (Tensor): Inputs.
            return_mean (bool): Whether to return the mean.
            return_var (bool): Whether to return the variance.
        Returns:
            Tensor or tuple of Tensors.
            The order of the tuple if all outputs are requested is:
                (mean, variance).
        """
        if not self._is_set:
            raise RuntimeError("You must call set_data() first")

        X = self.X
        Y = self.Y
        K_inv = self.K_inv

        # Kernel functions.
        K_ss = self.kernel(x, x)
        K_s = self.kernel(x, X)

        # Compute mean.
        outputs = []
        if return_mean:
            # Non-normalized for scale.
            mean = K_s.bmm(K_inv.bmm(self.non_normalized_Y))
            mean = mean.squeeze(-1).transpose(-2, -1)
            outputs.append(mean)

        # Compute variance.
        if return_var:
            ix, iy = np.diag_indices(K_ss.shape[-1])
            covar = K_ss - K_s.bmm(K_inv).bmm(K_s.transpose(-2, -1))
            var = covar[..., ix, iy].transpose(-2, -1)
            outputs.append(var.clamp(0, np.inf))

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)


def gp_dynamics_model_factory(state_size,
                              action_size,
                              angular_indices=None,
                              non_angular_indices=None,
                              constrain_min=None,
                              constrain_max=None,
                              **kwargs):
    """A GPDynamicsModel factory.

    Args:
        state_size (int): Augmented state size.
        action_size (int): Action size.
        angular_indices (Tensor<n>): Column indices of angular states.
        non_angular_indices (Tensor<m>): Complementary indices of
            `angular_indices`.
        constrain_min (Tensor<action_size>): Minimum action bounds.
        constrain_max (Tensor<action_size>): Maximum action bounds.
        **kwargs (dict): Additional key-word arguments to pass to
            `bayesian_model()`.

    Returns:
        GPDynamicsModel class.
    """
    angular = angular_indices is not None and non_angular_indices is not None
    should_constrain = constrain_min is not None and constrain_max is not None

    class GPDynamicsModel(DynamicsModel):

        """Gaussian process dynamics model."""

        def __init__(self):
            super(GPDynamicsModel, self).__init__()

        @classproperty
        def action_size(cls):
            """Action size (int)."""
            return action_size

        @classproperty
        def state_size(cls):
            """State size (int)."""
            return state_size

        def _constrain(self, u):
            if should_constrain:
                return constrain(u, constrain_min, constrain_max)
            return u

        def fit(self,
                X,
                U,
                dX,
                n_iter=500,
                learning_rate=1e-4,
                normalize=True,
                quiet=False,
                **kwargs):
            """Fits the dynamics model.

            Args:
                X (Tensor<N, state_size>): State trajectory.
                U (Tensor<N, action_size>): Action trajectory.
                dX (Tensor<N, state_size>): Next state trajectory.
                n_iter (int): Number of iterations.
                learning_rate (float): Learning rate.
                normalize (bool): Whether to normalize the dataset.
                quiet (bool): Whether to print anything to screen or not.
            """
            if angular:
                X = augment_state(X, angular_indices, non_angular_indices)

            U = self._constrain(U)
            X_ = torch.cat([X, U], dim=-1)

            log_length_scale = X_.std(dim=0).log().repeat(state_size, 1)
            log_sigma_s = dX.std(dim=0).log()
            log_sigma_n = 0.1 * dX.std(dim=0).log()

            kernel = RBFKernel(log_length_scale, log_sigma_s)
            self.add_module("model", GaussianProcess(kernel, log_sigma_n))

            X_ = X_.repeat(state_size, 1, 1)
            self.model.set_data(X_, dX, normalize_y=normalize)

            params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.Adam(params, learning_rate)

            with trange(n_iter, desc="GP", disable=quiet) as pbar:

                def closure():
                    loss = None
                    while loss is None:
                        try:
                            optimizer.zero_grad()
                            loss = self.model.loss()
                        except RuntimeError:
                            self.model.increase_reg()

                    pbar.set_postfix({"loss": loss.detach().cpu().numpy()})
                    loss.backward(retain_graph=True)
                    return loss

                for _ in pbar:
                    closure()
                    optimizer.step()

        def forward(self,
                    z,
                    u,
                    i,
                    encoding=StateEncoding.DEFAULT,
                    identical_inputs=False,
                    **kwargs):
            """Dynamics model function.

            Args:
                z (Tensor<..., encoded_state_size>): Encoded state distribution.
                u (Tensor<..., action_size>): Action vector(s).
                i (Tensor<...>): Time index.
                encoding (int): StateEncoding enum.
                identical_inputs (bool): Whether the batched inputs can be
                    assumed identical or not.

            Returns:
                Next encoded state distribution
                    (Tensor<..., encoded_state_size>).
            """
            mean = decode_mean(z, encoding)
            if angular:
                Z = augment_encoded_state(Z, angular_indices,
                                          non_angular_indices)
                X_ = decode_mean(Z, encoding)
            else:
                X_ = mean

            U = self._constrain(u)
            X_ = torch.cat([X_, U], dim=-1)
            if mean.dim() == 1:
                X_ = X_.unsqueeze(0)

            X_ = X_.expand(state_size, -1, -1)

            output_mean, output_var = self.model(X_, return_var=True)

            # TODO: Moment match.
            M = mean + output_mean
            V = output_var
            Z = encode(M, V=V, encoding=encoding)
            if mean.dim() == 1:
                Z = Z.squeeze(0)

            return Z

    return GPDynamicsModel