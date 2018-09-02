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

from .modules import (bnn_dynamics_model_factory, BDropout, CDropout,
                      BSequential, bayesian_model)
from .losses import gaussian_log_likelihood

__all__ = [
    "BDropout", "BSequential", "CDropout", "bayesian_model",
    "bnn_dynamics_model_factory", "gaussian_log_likelihood"
]
