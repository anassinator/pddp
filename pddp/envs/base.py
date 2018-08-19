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
"""Base environment."""

import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Env():

    """Base environment."""

    def __enter__(self):
        """Enters the environment's context."""
        return self

    def __exit__(self, type, value, traceback):
        """Exits the environment's context."""
        self.close()

    @property
    @abc.abstractmethod
    def action_size(self):
        """Action size (int)."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def state_size(self):
        """State size (int)."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, u):
        """Applies an action to the environment.

        Args:
            u (Tensor<action_size>): Action vector.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_state(self):
        """Gets the current state of the environment.

        Returns:
            State distribution (GaussianVariable<state_size>).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """Resets the environment."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """Stops the current environment session."""
        raise NotImplementedError
