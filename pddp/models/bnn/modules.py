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
"""Bayesian neural network modules."""

import torch
import inspect
import warnings
import numpy as np

with warnings.catch_warnings():
    # Ignore potential warning when in Jupyter environment.
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm

from functools import partial
from torch.nn import Parameter
from collections import Iterable, OrderedDict
from torch.utils.data import DataLoader, TensorDataset

from ..base import DynamicsModel
from .losses import gaussian_log_likelihood
from ...utils.classproperty import classproperty
from ...utils.encoding import StateEncoding, decode_mean, decode_std, encode
from ...utils.angular import (augment_encoded_state, augment_state,
                              infer_augmented_state_size, reduce_state)


def bnn_dynamics_model_factory(state_size,
                               action_size,
                               hidden_features,
                               angular_indices=None,
                               non_angular_indices=None,
                               **kwargs):
    """A BNNDynamicsModel factory.

    Args:
        state_size (int): Augmented state size.
        action_size (int): Action size.
        hidden_features (list<int>): Ordered list of hidden dimensions.
        **kwargs (dict): Additional key-word arguments to pass to
            `bayesian_model()`.

    Returns:
        BNNDynamicsModel class.
    """
    angular = angular_indices is not None and non_angular_indices is not None
    augmented_state_size = state_size
    if angular:
        augmented_state_size = infer_augmented_state_size(
            angular_indices, non_angular_indices)

    class BNNDynamicsModel(DynamicsModel):

        """Bayesian neural network dynamics model."""

        def __init__(self, n_particles=100):
            """Constructs a BNNDynamicsModel.

            Args:
                n_particles (int): Number of particles.
            """
            super(BNNDynamicsModel, self).__init__()

            self.model = bayesian_model(augmented_state_size + action_size,
                                        2 * state_size, hidden_features,
                                        **kwargs)

            self.n_particles = n_particles

            # Normalization parameters.
            self.register_buffer("X_mean", torch.tensor(0.0))
            self.register_buffer("X_std", torch.tensor(1.0))
            self.register_buffer("X_std_inv", torch.tensor(1.0))
            self.register_buffer("dX_mean", torch.tensor(0.0))
            self.register_buffer("dX_std", torch.tensor(1.0))
            self.register_buffer("dX_std_inv", torch.tensor(1.0))
            self.eps1 = []
            self.eps2 = []

        @classproperty
        def action_size(cls):
            """Action size (int)."""
            return action_size

        @classproperty
        def state_size(cls):
            """State size (int)."""
            return state_size

        def resample(self):
            """Resamples model."""
            self.eps = []
            self.model.resample()

        def _normalize_input(self, x_):
            return (x_ - self.X_mean) * self.X_std_inv

        def _scale_output(self, mean, log_std):
            mean = mean * self.dX_std + self.dX_mean
            log_std = log_std + self.dX_std.log()
            return mean, log_std

        def fit(self,
                X,
                U,
                dX,
                n_iter=500,
                batch_size=128,
                reg_scale=1.0,
                learning_rate=1e-4,
                likelihood=gaussian_log_likelihood,
                resample=True,
                normalize=True,
                quiet=False,
                **kwargs):
            """Fits the dynamics model.

            Args:
                X (Tensor<N, state_size>): State trajectory.
                U (Tensor<N, action_size>): Action trajectory.
                dX (Tensor<N, state_size>): Next state trajectory.
                n_iter (int): Number of iterations.
                batch_size (int): Batch size of each iteration.
                reg_scale (float): Regularization scale.
                learning_rate (float): Learning rate.
                likelihood (callable): Likelihood function.
                resample (bool): Whether to resample during training or not.
                normalize (bool): Whether to normalize or not.
                quiet (bool): Whether to print anything to screen or not.
            """
            if angular:
                X = augment_state(X, angular_indices, non_angular_indices)

            X_ = torch.cat([X, U], dim=-1)
            N = X_.shape[0]

            if normalize:
                self.X_mean.data = X_.mean(dim=0).detach()
                self.X_std.data = X_.std(dim=0).detach()
                self.X_std_inv.data = self.X_std.reciprocal()
                self.dX_mean.data = dX.mean(dim=0).detach()
                self.dX_std.data = dX.std(dim=0).detach()
                self.dX_std_inv.data = self.dX_std.reciprocal()

            params = filter(lambda p: p.requires_grad, self.parameters())
            optimizer = torch.optim.Adam(params, learning_rate, amsgrad=True)

            dataset = TensorDataset(X_, dX)
            dataloader = DataLoader(
                dataset, batch_size=batch_size, shuffle=True)
            datagen = _cycle(dataloader, n_iter)
            with tqdm(datagen, total=n_iter, desc="BNN", disable=quiet) as pbar:
                for x_, dx in pbar:
                    optimizer.zero_grad()

                    x_ = self._normalize_input(x_)
                    output = self.model(x_, resample=resample)

                    mean, log_std = output.split(
                        [state_size, state_size], dim=-1)
                    mean, log_std = self._scale_output(mean, log_std)

                    loss = -likelihood(dx, mean, log_std.exp()).mean()
                    reg_loss = self.model.regularization() / N
                    loss += reg_scale * reg_loss
                    pbar.set_postfix({"loss": loss.detach().cpu().numpy()})

                    loss.backward()
                    optimizer.step()

        def forward(self,
                    z,
                    u,
                    i,
                    encoding=StateEncoding.DEFAULT,
                    resample=False,
                    return_samples=False,
                    sample_input_distribution=True,
                    use_predicted_std=False,
                    **kwargs):
            """Dynamics model function.

            Args:
                z (Tensor<..., encoded_state_size>): Encoded state distribution.
                u (Tensor<..., action_size>): Action vector(s).
                i (Tensor<...>): Time index.
                encoding (int): StateEncoding enum.
                resample (bool): Whether to force resample.
                return_samples (bool): Whether to return all samples instead of
                    the encoded state distribution.
                sample_input_distribution (bool): Whether to sample particles
                    from the input distribution or simply use their mean.
                use_predicted_std (bool): Whether to use the predicted standard
                    deviations in the inference or not.

            Returns:
                Next encoded state distribution
                    (Tensor<..., encoded_state_size>) or next samples
                    (Tensor<n_particles, ..., state_size>).
            """
            if angular:
                original_mean = decode_mean(z, encoding)
                z = augment_encoded_state(z, angular_indices,
                                          non_angular_indices, encoding,
                                          state_size)

            mean = decode_mean(z, encoding)
            std = decode_std(z, encoding)
            x = mean.expand(self.n_particles, *mean.shape)

            if sample_input_distribution:
                if resample or len(self.eps1) < i+1:
                    if x.dim() == 3:
                        # This is required to make batched jacobians correct as the
                        # batches are in the second dimension and should share the same
                        # samples.
                        eps = torch.randn_like(x[:, 0, :])
                    else:
                        eps = torch.randn_like(x)
                    self.eps1.append(eps)

                eps = self.eps1[i]
                if x.dim() == 3:
                    eps = eps.unsqueeze(1).repeat(1, x.shape[1], 1)
                x = x + std * eps

            u_ = u.expand(self.n_particles, *u.shape)
            x_ = torch.cat([x, u_], dim=-1)

            # Normalize.
            x_ = self._normalize_input(x_)

            if x_.dim() == 3:
                # Shuffle the dimensions for the proper masks to be used in
                # batch jacobians.
                x_ = x_.permute(1, 0, 2)

            output = self.model(x_, resample=resample)
            if output.dim() == 3:
                # Unshuffle the dimensions.
                output = output.permute(1, 0, 2)

            dx, log_std = output.split([state_size, state_size], dim=-1)
            dx, log_std = self._scale_output(dx, log_std)

            if use_predicted_std:
                if resample or len(self.eps1) < i+1:
                    if dx.dim() == 3:
                        # This is required to make batched jacobians correct as the
                        # batches are in the second dimension and should share the
                        # same samples.
                        eps = torch.randn_like(dx[:, 0, :])
                    else:
                        eps = torch.randn_like(dx)
                    self.eps2.append(eps)

                eps = self.eps2[i]
                if x.dim() == 3:
                    eps = eps.unsqueeze(1).repeat(1, dx.shape[1], 1)
                dx = dx + log_std.exp() * eps

            if angular:
                # We need the reduced state.
                mean = original_mean

            if return_samples:
                if angular:
                    x = reduce_state(x, angular_indices, non_angular_indices)
                return x + dx

            M = mean + dx.mean(dim=0)
            S = dx.std(dim=0)
            return encode(M, S=S, encoding=encoding)

    return BNNDynamicsModel


def _cycle(iterable, total):
    """Cycles through an iterable until a total number of iterations is reached.

    Args:
        iterable (iterable): Non-exhaustive iterable to cycle through.
        total (int): Total number of iterations to go through.

    Yields:
        Element from iterable.
    """
    i = 0
    while True:
        for x in iterable:
            i += 1
            yield x
            if i == total:
                return


class BDropout(torch.nn.Dropout):

    """Binary dropout with regularization and resampling.

    See: Gal Y., Ghahramani Z., "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning", 2016.
    """

    def __init__(self, rate=0.1, reg=1.0, **kwargs):
        """Constructs a BDropout.

        Args:
            rate (float): Dropout rate.
            reg (float): Regularization scale.
        """
        super(BDropout, self).__init__(**kwargs)
        self.register_buffer("rate", torch.tensor(rate))
        self.p = 1 - self.rate
        self.register_buffer("reg", torch.tensor(reg))
        self.register_buffer("noise", torch.bernoulli(self.p))

    def regularization(self, weight, bias):
        """Computes the regularization cost.

        Args:
            weight (Tensor): Weight tensor.
            bias (Tensor): Bias tensor.

        Returns:
            Regularization cost (Tensor).
        """
        self.p = 1 - self.rate
        weight_reg = self.p * (weight**2).sum()
        bias_reg = (bias**2).sum() if bias is not None else 0
        return self.reg * (weight_reg + bias_reg)

    def resample(self):
        """Resamples the dropout noise."""
        self._update_noise(self.noise)

    def _update_noise(self, x):
        """Updates the dropout noise.

        Args:
            x (Tensor): Input.
        """
        self.p = 1 - self.rate
        self.noise.data = torch.bernoulli(self.p.expand(x.shape))

    def forward(self, x, resample=False, mask_dims=2, **kwargs):
        """Computes the binary dropout.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.
            mask_dims (int): Number of dimensions to sample noise for
                (0 for all).

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        if resample:
            return x * torch.bernoulli(self.p.expand(x.shape))
        elif sample_shape != self.noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self._update_noise(sample)

        # We never need these gradients in evaluation mode.
        noise = self.noise if self.training else self.noise.detach()
        return x * noise

    def extra_repr(self):
        """Formats module representation.

        Returns:
            Module representation (str).
        """
        return "rate={}".format(self.rate)


class CDropout(BDropout):

    """Concrete dropout with regularization and resampling.

    See: Gal Y., Hron, J., Kendall, A. "Concrete Dropout", 2017.
    """

    def __init__(self, temperature=0.1, rate=0.5, reg=1.0, **kwargs):
        """Constructs a CDropout.

        Args:
            temperature (float): Temperature.
            rate (float): Initial dropout rate.
            reg (float): Regularization scale.
        """
        super(CDropout, self).__init__(rate, reg, **kwargs)
        self.temperature = Parameter(
            torch.tensor(temperature), requires_grad=False)

        # We need to constrain p to [0, 1], so we train logit(p).
        self.logit_p = Parameter(-torch.log(self.p.reciprocal() - 1.0))

    def regularization(self, weight, bias):
        """Computes the regularization cost.

        Args:
            weight (Tensor): Weight tensor.
            bias (Tensor): Bias tensor.

        Returns:
            Regularization cost (Tensor).
        """
        self.p.data = self.logit_p.sigmoid()
        reg = super(CDropout, self).regularization(weight, bias)
        reg -= -(1 - self.p) * (1 - self.p).log() - self.p * self.p.log()
        return reg

    def _update_noise(self, x):
        """Updates the dropout noise.

        Args:
            x (Tensor): Input.
        """
        self.noise.data = torch.rand_like(x)

    def forward(self, x, resample=False, mask_dims=2, **kwargs):
        """Computes the concrete dropout.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.
            mask_dims (int): Number of dimensions to sample noise for
                (0 for all).

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        noise = self.noise
        if resample:
            noise = torch.rand_like(x)
        elif sample_shape != noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self._update_noise(sample)
            noise = self.noise

        self.p.data = self.logit_p.sigmoid()
        concrete_p = self.logit_p + noise.log() - (1 - noise).log()
        concrete_noise = (concrete_p / self.temperature).sigmoid()

        if not self.training:
            # We never need these gradients in evaluation mode.
            concrete_noise = concrete_noise.detach()

        return x * concrete_noise

    def extra_repr(self):
        """Formats module representation.

        Returns:
            Module representation (str).
        """
        return "temperature={}".format(self.temperature)


class BSequential(torch.nn.Sequential):

    """Extension of Sequential module with regularization and resampling."""

    def resample(self):
        """Resample all child modules."""
        for child in self.children():
            if isinstance(child, BDropout):
                child.resample()

    def regularization(self):
        """Computes the total regularization cost of all child modules.

        Returns:
            Total regularization cost (Tensor).
        """
        reg = torch.tensor(0.0)
        children = list(self._modules.values())
        for i, child in enumerate(children):
            if isinstance(child, BSequential):
                reg += child.regularization()
            elif isinstance(child, BDropout):
                for next_child in children[i:]:
                    if hasattr(next_child, "weight") and hasattr(
                            next_child, "bias"):
                        reg += child.regularization(next_child.weight,
                                                    next_child.bias)
                        break
        return reg

    def forward(self, x, resample=False, **kwargs):
        """Computes the model.

        Args:
            x (Tensor): Input.
            resample (bool): Whether to force resample.

        Returns:
            Output (Tensor).
        """
        for module in self._modules.values():
            if isinstance(module, (BDropout, BSequential)):
                x = module(x, resample=resample, **kwargs)
            else:
                x = module(x)
        return x


def bayesian_model(in_features,
                   out_features,
                   hidden_features,
                   nonlin=torch.nn.ReLU,
                   output_nonlin=None,
                   weight_initializer=partial(
                       torch.nn.init.xavier_normal_,
                       gain=torch.nn.init.calculate_gain("relu")),
                   bias_initializer=partial(
                       torch.nn.init.uniform_, a=-0.1, b=0.1),
                   initial_p=0.5,
                   dropout_layers=CDropout,
                   input_dropout=None):
    """Constructs and initializes a Bayesian neural network with dropout.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        hidden_features (list<int>): Ordered list of hidden dimensions.
        nonlin (Module): Activation function for all hidden layers.
        output_nonlin (Module): Activation function for output layer.
        weight_initializer (callable): Function to initialize all module
            weights to pass to module.apply().
        bias_initializer (callable): Function to initialize all module
            biases to pass to module.apply().
        initial_p (float): Initial dropout probability.
        dropout_layers (Dropout or list<Dropout>): Dropout type to apply to
            hidden layers.
        input_dropout (Dropout): Dropout to apply to input layer.

    Returns:
        Bayesian neural network (BSequential).
    """
    dims = [in_features] + hidden_features
    if not isinstance(dropout_layers, Iterable):
        dropout_layers = [dropout_layers] * len(hidden_features)

    modules = OrderedDict()

    # Input layer.
    if inspect.isclass(input_dropout):
        input_dropout = input_dropout()
    if input_dropout is not None:
        modules["drop_in"] = input_dropout

    # Hidden layers.
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        drop_i = dropout_layers[i]
        if inspect.isclass(drop_i):
            drop_i = drop_i(p=initial_p)

        modules["fc_{}".format(i)] = torch.nn.Linear(din, dout)
        if drop_i is not None:
            modules["drop_{}".format(i)] = drop_i
        modules["nonlin_{}".format(i)] = nonlin()

    # Output layer.
    modules["fc_out"] = torch.nn.Linear(dims[-1], out_features)
    if output_nonlin is not None:
        modules["nonlin_out"] = output_nonlin()

    def init(module):
        if callable(weight_initializer) and hasattr(module, "weight"):
            weight_initializer(module.weight)
        if callable(bias_initializer) and hasattr(module, "bias"):
            if module.bias is not None:
                bias_initializer(module.bias)

    # Initialize weights and biases.
    net = BSequential(modules)
    net.apply(init)

    return net
