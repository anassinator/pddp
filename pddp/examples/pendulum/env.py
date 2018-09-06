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
"""Pendulum environment."""

import time
import torch
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from ...envs import GymEnv
from .model import PendulumDynamicsModel
from ...utils.encoding import StateEncoding
from ...utils.gaussian_variable import GaussianVariable
from ...utils.angular import augment_state, reduce_state


class PendulumEnv(GymEnv):

    """Pendulum environment.

    Note: This environment is preconstrained if the model is.
    """

    def __init__(self, model=None, dt=0.1, render=False):
        """Constructs a PendulumEnv.

        Args:
            model (PendulumDynamicsModel): Ground truth dynamics model.
            dt (float): Time difference (s). Only used if model isn't defined.
            render (bool): Whether to render the environment or not.
        """
        self.dt = dt
        if model is None:
            model = PendulumDynamicsModel(dt)

        gym_env = _PendulumEnv(model.eval())
        super(PendulumEnv, self).__init__(gym_env, render=render)

    def apply(self, u):
        """Applies an action to the environment.

        Args:
            u (Tensor<action_size>): Action vector.
        """
        super(PendulumEnv, self).apply(u)
        if self._render:
            time.sleep(self.dt)


class _PendulumEnv(gym.Env):

    """Open AI gym pendulum environment.

    Based on the OpenAI gym Pendulum-v0 environment, but with more
    custom dynamics for a better ground truth.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
    }

    def __init__(self, model):
        self.model = model

        high = np.array([np.finfo(np.float32).max])
        self.action_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        high = np.array([np.pi, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(
            low=-high, high=high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        tensor_opts = {"dtype": torch.get_default_dtype()}
        x = self.state.astype(np.float32)
        x_next = self.model(
            torch.tensor(x, **tensor_opts),
            torch.tensor(action, **tensor_opts),
            0,
            encoding=StateEncoding.IGNORE_UNCERTAINTY)
        self.state = x_next.detach().cpu().numpy()
        reward = 0.0
        done = False

        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([np.pi, 0.0])
        self.state += 1e-2 * np.random.random(self.state.shape)
        return self.state

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)

            rod = rendering.make_capsule(1.0, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)

            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)

        theta, _ = self.state
        self.pole_transform.set_rotation(theta + np.pi / 2)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
