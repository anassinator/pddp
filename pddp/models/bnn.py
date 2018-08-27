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
"""Bayesian neural network dynamics models."""

import torch
import inspect
import warnings
import numpy as np

with warnings.catch_warnings():
    # Ignore potential warning when in Jupyter environment.
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange

from functools import partial
from torch.nn import Parameter
from collections import Iterable, OrderedDict

from .base import DynamicsModel
from ..utils.classproperty import classproperty
from ..utils.encoding import StateEncoding, decode_mean, decode_std, encode


def bnn_dynamics_model_factory(state_size, action_size, hidden_features,
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

    class BNNDynamicsModel(DynamicsModel):

        """Bayesian neural network dynamics model."""

        def __init__(self, reg_scale=1e-2, n_particles=10):
            """Constructs a BNNDynamicsModel.

            Args:
                reg_scale (float): Regularization scale.
                n_particles (int): Number of particles.
            """
            super(BNNDynamicsModel, self).__init__()
            self.model = bayesian_model(state_size + action_size, state_size,
                                        hidden_features, **kwargs)
            self.reg_scale = reg_scale
            self.n_particles = n_particles

        @classproperty
        def action_size(cls):
            """Action size (int)."""
            return action_size

        @classproperty
        def state_size(cls):
            """Augmented state size (int)."""
            return state_size

        def resample(self):
            """Resamples model."""
            self.model.resample()

        def fit(self,
                dataset,
                n_iter=500,
                batch_size=128,
                reg_scale=1.0,
                resample=True,
                quiet=False,
                **kwargs):
            """Fits the dynamics model.

            Args:
                dataset
                    (Dataset<Tensor<N, state_size + action_size>,
                             Tensor<N, action_size>>):
                    Dataset of state-action pair trajectory and next state
                    trajectory.
                n_iter (int): Number of iterations.
                batch_size (int): Batch size of each iteration.
                reg_scale (float): Regularization scale.
                resample (bool): Whether to resample during training or not.
                quiet (bool): Whether to print anything to screen or not.
            """
            params = filter(lambda p: p.requires_grad, self.parameters())
            optimizer = torch.optim.Adam(params, 1e-4)

            X_, dX = dataset.tensors
            N = X_.shape[0]
            pbar = trange(n_iter, desc="BNN", disable=quiet)
            for _ in pbar:
                optimizer.zero_grad()
                output = self.model(X_, resample=resample)

                loss = -_gaussian_log_likelihood(dX, output).mean()
                reg_loss = self.model.regularization() / N
                loss += self.reg_scale * reg_loss
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

            Returns:
                Next encoded state distribution
                    (Tensor<..., encoded_state_size>) or next samples
                    (Tensor<n_particles, ..., state_size>).
            """
            # Moment match.
            # Sample manually to preserve gradients.
            mean = decode_mean(z, encoding)
            std = decode_std(z, encoding)
            x = mean.expand(self.n_particles, *mean.shape)
            x = x + std * torch.randn_like(x)

            u_ = u.expand(self.n_particles, *u.shape)
            x_ = torch.cat([x, u_], dim=-1)
            if x_.dim() == 3:
                x_ = x_.permute(1, 0, 2)
            dx = self.model(x_, resample=resample)
            if dx.dim() == 3:
                dx = dx.permute(1, 0, 2)

            if return_samples:
                return x + dx

            M = mean + dx.mean(dim=0)
            S = dx.std(dim=0)
            return encode(M, S=S, encoding=encoding)

    return BNNDynamicsModel


def _gaussian_log_likelihood(targets, pred_means, pred_stds=None):
    """Computes the gaussian log marginal likelihood.

    Args:
        targets (Tensor): Target values.
        pred_means (Tensor): Predicted means.
        pred_stds (Tensor): Predicted standard deviations.

    Returns:
        Gaussian log marginal likelihood (Tensor).
    """
    deltas = pred_means - targets
    if pred_stds is not None:
        lml = -((deltas / pred_stds)**2).sum(dim=-1) * 0.5 \
              - pred_stds.log().sum(dim=-1) \
              - np.log(2 * np.pi) * 0.5
    else:
        lml = -(deltas**2).sum(dim=-1) * 0.5

    return lml


class BDropout(torch.nn.Dropout):

    """Binary dropout with regularization and resampling.

    See: Gal Y., Ghahramani Z., "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning", 2016.
    """

    def __init__(self, p=0.5, reg=1.0, **kwargs):
        """Constructs a BDropout.

        Args:
            p (float): Dropout probability.
            reg (float): Regularization scale.
        """
        super(BDropout, self).__init__(**kwargs)
        self.register_buffer("rate", torch.tensor(p))
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

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        if sample_shape != self.noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self._update_noise(sample)
        elif resample:
            return x * torch.bernoulli(self.p.expand(x.shape))

        return x * self.noise

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

    def __init__(self, temperature=0.1, p=0.5, reg=1.0, **kwargs):
        """Constructs a CDropout.

        Args:
            temperature (float): Temperature.
            p (float): Initial dropout probability.
            reg (float): Regularization scale.
        """
        super(CDropout, self).__init__(p, reg, **kwargs)
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

        Returns:
            Output (Tensor).
        """
        sample_shape = x.shape[-mask_dims:]
        noise = self.noise
        if sample_shape != noise.shape:
            sample = x.view(-1, *sample_shape)[0]
            self._update_noise(sample)
            noise = self.noise
        elif resample:
            noise = torch.rand_like(x)

        self.p.data = self.logit_p.sigmoid()
        concrete_p = self.logit_p + noise.log() - (1 - noise).log()
        concrete_noise = (concrete_p / self.temperature).sigmoid()

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
                       torch.nn.init.uniform_, a=-1.0, b=1.0),
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
