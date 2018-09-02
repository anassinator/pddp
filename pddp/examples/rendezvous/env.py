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
"""Multi-vehicle rendezvous environment."""

import torch
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from ...envs import GymEnv
from .model import RendezvousDynamicsModel
from ...utils.encoding import StateEncoding
from ...utils.angular import augment_state, reduce_state


class RendezvousEnv(GymEnv):

    """Multi-vehicle rendezvous environment.

    Note: This environment is preconstrained if the model is.
    """

    def __init__(self, model=None, dt=0.05, render=False):
        """Constructs a RendezvousEnv.

        Args:
            model (RendezvousDynamicsModel): Ground truth dynamics model.
            dt (float): Time difference (s). Only used if model isn't defined.
            render (bool): Whether to render the environment or not.
        """
        if model is None:
            model = RendezvousDynamicsModel(dt)

        gym_env = _RendezvousEnv(model)
        super(RendezvousEnv, self).__init__(gym_env, render=render)


class _RendezvousEnv(gym.Env):

    """Open AI gym multi-vehicle rendezvous environment."""

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 30,
    }

    def __init__(self, model):
        self.model = model

        high = np.array([np.finfo(np.float32).max] * 4)
        self.action_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        high = np.array([np.finfo(np.float32).max] * 8)
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
        self.state = np.array([-10.0, -10.0, 10.0, 10.0, 0.0, -5.0, 5.0, 0.0])
        self.state += 1e-2 * np.random.random(self.state.shape)
        return self.state

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-15.0, 15.0, -15.0, 15.0)

            vehicle_0 = rendering.make_circle(0.5)
            vehicle_0.set_color(1.0, 0.0, 0.0)
            self.vehicle_0_transform = rendering.Transform()
            vehicle_0.add_attr(self.vehicle_0_transform)
            self.viewer.add_geom(vehicle_0)

            vehicle_1 = rendering.make_circle(0.5)
            vehicle_1.set_color(0.0, 0.0, 1.0)
            self.vehicle_1_transform = rendering.Transform()
            vehicle_1.add_attr(self.vehicle_1_transform)
            self.viewer.add_geom(vehicle_1)

        x_0, y_0, x_1, y_1, _, _, _, _ = self.state
        self.vehicle_0_transform.set_translation(x_0, y_0)
        self.vehicle_1_transform.set_translation(x_1, y_1)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
