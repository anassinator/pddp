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
import gpytorch
import warnings

with warnings.catch_warnings():
    # Ignore potential warning when in Jupyter environment.
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange

from torch import optim
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import RBFKernel, MultitaskKernel
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.random_variables import (GaussianRandomVariable,
                                       MultitaskGaussianRandomVariable)

from ..base import DynamicsModel

from ...utils.constraint import constrain
from ...utils.classproperty import classproperty
from ...utils.encoding import StateEncoding, decode_mean, encode
from ...utils.angular import (augment_state, augment_state)


def gpytorch_dynamics_model_factory(state_size,
                                    action_size,
                                    angular_indices=None,
                                    non_angular_indices=None,
                                    constrain_min=None,
                                    constrain_max=None,
                                    **kwargs):
    """A GPytorchDynamicsModel factory.

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
        GpytorchDynamicsModel class.
    """
    angular = angular_indices is not None and non_angular_indices is not None
    should_constrain = constrain_min is not None and constrain_max is not None

    augmented_state_size = state_size
    if angular:
        augmented_state_size = infer_augmented_state_size(
            angular_indices, non_angular_indices)
    input_size = augmented_state_size + action_size

    class SingletaskGPModel(gpytorch.models.ExactGP):

        """GPytorch single-task gaussian process model."""

        def __init__(self, train_x, train_y, likelihood):
            super(SingletaskGPModel, self).__init__(train_x, train_y,
                                                    likelihood)
            self.mean_module = ConstantMean()
            self.covar_module = RBFKernel(ard_num_dims=input_size)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return GaussianRandomVariable(mean_x, covar_x)

    class MultitaskGPModel(gpytorch.models.ExactGP):

        """GPytorch multi-task gaussian process model."""

        def __init__(self, train_x, train_y, likelihood):
            super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = MultitaskMean(ConstantMean(), n_tasks=state_size)
            self.data_covar_module = RBFKernel(ard_num_dims=input_size)
            self.covar_module = MultitaskKernel(
                self.data_covar_module, n_tasks=state_size, rank=1)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultitaskGaussianRandomVariable(mean_x, covar_x)

    class GPytorchDynamicsModel(DynamicsModel):

        """Gaussian process dynamics model."""

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
                n_iter=50,
                learning_rate=1e-1,
                quiet=False,
                **kwargs):
            """Fits the dynamics model.

            Args:
                X (Tensor<N, state_size>): State trajectory.
                U (Tensor<N, action_size>): Action trajectory.
                dX (Tensor<N, state_size>): Next state trajectory.
                n_iter (int): Number of iterations.
                learning_rate (float): Learning rate.
                quiet (bool): Whether to print anything to screen or not.
            """
            if angular:
                X = augment_state(X, angular_indices, non_angular_indices)

            U = self._constrain(U)
            X_ = torch.cat([X, U], dim=-1)

            # TODO: Figure out why learning dX instead breaks things.
            Y = X + dX

            if state_size > 1:
                likelihood = MultitaskGaussianLikelihood(
                    n_tasks=state_size).train()
                model = MultitaskGPModel(X_, Y, likelihood).train()
            else:
                Y = Y.squeeze(-1)
                likelihood = GaussianLikelihood().train()
                model = SingletaskGPModel(X_, Y, likelihood).train()

            params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = torch.optim.Adam(params, learning_rate, amsgrad=True)

            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            with trange(n_iter, desc="GP", disable=quiet) as pbar:
                for _ in pbar:
                    optimizer.zero_grad()

                    output = model(X_)
                    loss = -mll(output, Y)
                    pbar.set_postfix({"loss": loss.detach().cpu().numpy()})

                    loss.backward()
                    optimizer.step()

            self.model = model.eval()
            self.likelihood = likelihood.eval()

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

            output = self.likelihood(self.model(X_))
            output_mean = output.mean()
            output_var = output.var()

            if state_size == 1:
                output_mean = output_mean.unsqueeze(-1)
                output_var = output_var.unsqueeze(-1)

            # TODO: Moment match.
            M = output_mean
            V = output_var
            Z = encode(M, V=V, encoding=encoding)
            if mean.dim() == 1:
                Z = Z.squeeze(0)

            return Z

    return GPytorchDynamicsModel