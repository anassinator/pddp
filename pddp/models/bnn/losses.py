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
"""Bayesian neural network losses."""

import numpy as np


def gaussian_log_likelihood(targets, pred_means, pred_stds=None):
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