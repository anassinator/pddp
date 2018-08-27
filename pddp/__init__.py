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
"""Probabilistic Differential Dynamic Programming library."""

from .__version__ import __version__

from . import controllers, costs, envs, models, utils

# Re-export some stuff for better visibility.
from .utils.encoding import StateEncoding
from .utils.gaussian_variable import GaussianVariable

__all__ = [
    "controllers", "costs", "envs", "models", "utils", "GaussianVariable",
    "StateEncoding"
]
