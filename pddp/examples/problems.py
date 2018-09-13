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
"""Sample problem setups."""

from enum import IntEnum
from . import cartpole, double_cartpole, pendulum, rendezvous


class SampleProblems(IntEnum):

    """Sample problem enum."""

    CARTPOLE = 1
    DOUBLE_CARTPOLE = 2
    PENDULUM = 3
    RENDEZVOUS = 4

    def setup(self, dt, render=False, **kwargs):
        """Sets up an example environment.

        Args:
            dt (float): Time step [s].
            render (bool): Whether to render the environment or not.
            **kwargs: Additional keyword environments to pass to the model
                constructor.

        Returns:
            Tuple of the environment (Env), the cost (Cost), and the model
            (DynamicsModel).
        """
        env_class = self.get_env_class()
        cost_class = self.get_cost_class()
        model_class = self.get_model_class()

        model = model_class(dt, **kwargs)
        cost = cost_class()
        env = env_class(dt=dt, model=model_class(dt, **kwargs), render=render)

        return env, cost, model

    def get_env_class(self):
        """Returns the corresponding Env class."""
        if self == self.CARTPOLE:
            return cartpole.CartpoleEnv
        elif self == self.DOUBLE_CARTPOLE:
            return double_cartpole.DoubleCartpoleEnv
        elif self == self.PENDULUM:
            return pendulum.PendulumEnv
        elif self == self.RENDEZVOUS:
            return rendezvous.RendezvousEnv

        raise NotImplementedError("Unsupported example: {}".format(self))

    def get_cost_class(self):
        """Returns the corresponding Cost class."""
        if self == self.CARTPOLE:
            return cartpole.CartpoleCost
        elif self == self.DOUBLE_CARTPOLE:
            return double_cartpole.DoubleCartpoleCost
        elif self == self.PENDULUM:
            return pendulum.PendulumCost
        elif self == self.RENDEZVOUS:
            return rendezvous.RendezvousCost

        raise NotImplementedError("Unsupported example: {}".format(self))

    def get_model_class(self):
        """Returns the corresponding DynamicsModel class."""
        if self == self.CARTPOLE:
            return cartpole.CartpoleDynamicsModel
        elif self == self.DOUBLE_CARTPOLE:
            return double_cartpole.DoubleCartpoleDynamicsModel
        elif self == self.PENDULUM:
            return pendulum.PendulumDynamicsModel
        elif self == self.RENDEZVOUS:
            return rendezvous.RendezvousDynamicsModel

        raise NotImplementedError("Unsupported example: {}".format(self))