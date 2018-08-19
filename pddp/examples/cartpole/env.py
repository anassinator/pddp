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
"""Cartpole environment."""

import torch
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from ...envs import GymEnv
from .model import CartpoleDynamicsModel
from ...models.utils.encoding import StateEncoding
from ...models.gaussian_variable import GaussianVariable
from ...models.utils.angular import augment_state, reduce_state


class CartpoleEnv(GymEnv):

    """Cartpole environment."""

    def __init__(self, model=None, dt=0.05, render=False):
        """Constructs a CartpoleEnv.

        Args:
            model (CartpoleDynamicsModel): Ground truth dynamics model.
            dt (float): Time difference (s). Only used if model isn't defined.
            render (bool): Whether to render the environment or not.
        """
        if model is None:
            model = CartpoleDynamicsModel(dt)

        self._model = model
        gym_env = _CartPoleEnv(model)
        super(CartpoleEnv, self).__init__(gym_env, render=render)

    def get_state(self):
        """Gets the current state of the environment.

        Returns:
            State distribution (GaussianVariable<augmented_state_size>).
        """
        state = augment_state(self._state, self._model.angular_indices,
                              self._model.non_angular_indices)
        return GaussianVariable(state, var=1e-6 * torch.ones_like(state))


class _CartPoleEnv(gym.Env):

    """Open AI gym cartpole environment.

    Based on the OpenAI gym CartPole-v0 environment, but with full-swing up
    support and a continuous action-space.
    """

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }

    def __init__(self, model):
        self.model = model

        high = np.array([np.finfo(np.float32).max])
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            2 * np.pi,
            np.finfo(np.float32).max,
        ])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        x = self.state.astype(np.float32)
        z = augment_state(
            torch.tensor(x), self.model.angular_indices,
            self.model.non_angular_indices)
        z_next = self.model(
            z,
            torch.tensor(action),
            0,
            encoding=StateEncoding.IGNORE_UNCERTAINTY)
        x_next = reduce_state(z_next, self.model.angular_indices,
                              self.model.non_angular_indices)

        self.state = x_next.detach().numpy()
        reward = 0.0
        done = False

        return self.state, reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[2] += np.pi  # Point the pole down.
        return self.state

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = 5.0
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = (-cartwidth / 2, cartwidth / 2, cartheight / 2,
                          -cartheight / 2)
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            l, r, t, b = (-polewidth / 2, polewidth / 2,
                          polelen - polewidth / 2, -polewidth / 2)
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x, _, theta, _ = self.state
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
