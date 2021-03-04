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
"""Particle utilities."""

import torch

from .encoding import StateEncoding, decode_mean, decode_covar_sqrt, encode


def particulate_model(cls, n_particles=100):
    """Converts a DynamicsModel into a ParticleDynamicsModel with
    moment-matching.

    Args:
        cls (class): DynamicsModel class.
        n_particles (int): Number of particles to use.

    Returns:
        ParticlesDynamicsModel<cls> class.
    """
    class ParticleDynamicsModel(cls):
        """Particle dynamics model."""
        def __init__(self, *args, **kwargs):
            """Constructs a ParticleDynamicsModel."""
            super(ParticleDynamicsModel, self).__init__(*args, **kwargs)
            self.eps = {}
            self.output = {}
            self.n_particles = n_particles

        def resample(self):
            """Resamples the model's particles."""
            super(ParticleDynamicsModel, self).resample()
            self.eps = {}
            self.output = {}

        def forward(self,
                    z,
                    u,
                    i,
                    encoding=StateEncoding.DEFAULT,
                    identical_inputs=False,
                    resample=False,
                    infer_noise_variables=True,
                    **kwargs):
            """Dynamics model function.

            Args:
                z (Tensor<..., encoded_state_size>): Encoded state distribution.
                u (Tensor<..., action_size>): Action vector(s).
                i (int): Time index.
                encoding (int): StateEncoding enum.

            Returns:
                Next encoded state distribution
                (Tensor<..., encoded_state_size>).
            """
            mean = decode_mean(z, encoding)

            L = decode_covar_sqrt(z, encoding)
            X = mean.expand(self.n_particles, *mean.shape)
            U = u.expand(self.n_particles, *u.shape)

            if resample or i not in self.eps:
                if X.dim() == 3:
                    # This is required to make batched jacobians correct as
                    # the batches are in the second dimension and should
                    # share the same samples.
                    eps = torch.randn_like(X[:, 0, :])
                else:
                    eps = torch.randn_like(X)
                self.eps[i] = (eps - eps.mean(0)) / eps.std(0)

            should_expand = i == 0
            if infer_noise_variables and i > 0:
                deltas = self.output[i - 1] - mean
                if not identical_inputs and X.dim() == 3:
                    deltas = deltas.permute(1, 0, 2)
                    eps = deltas.transpose(-2, -1).solve(L.transpose(-2, -1))[0]
                    eps = eps.permute(2, 0, 1).detach()
                else:
                    if X.dim() == 3:
                        deltas = deltas[:, 0]
                        L_ = L[0]
                    else:
                        L_ = L
                    eps = torch.trtrs(deltas.t(), L_, transpose=True)[0].t().detach()
                    should_expand = True
            else:
                eps = self.eps[i]
                should_expand = True

            if X.dim() == 3:
                if should_expand:
                    eps = eps.unsqueeze(1).repeat(1, X.shape[1], 1)
                X = X + (eps[:, :, :, None] * L[None, :, :, :]).sum(-2)
            else:
                X = X + eps.mm(L)

            output = super(ParticleDynamicsModel, self).forward(X,
                                                                U,
                                                                i,
                                                                encoding=StateEncoding.IGNORE_UNCERTAINTY,
                                                                identical_inputs=identical_inputs,
                                                                resample=resample,
                                                                **kwargs)
            self.output[i] = output

            M = output.mean(dim=0)
            C = particles_covar(output)
            try:
                return encode(M, C=C, encoding=encoding)
            except RuntimeError:
                import pdb
                pdb.set_trace()

    return ParticleDynamicsModel


def particles_covar(x):
    """Computes the covariance of a set of particles.

    Args:
        x (Tensor<..., n_particles, state_size>): Particle set.

    Returns:
        Covariance matrix (Tensor<..., state_size, state_size>).
    """
    deltas = x - x.mean(dim=0)
    if deltas.dim() == 3:
        deltas = deltas.permute(1, 0, 2)
        return deltas.transpose(1, 2).bmm(deltas) / (x.shape[0] - 1)
    return deltas.t().mm(deltas) / (x.shape[0] - 1)
