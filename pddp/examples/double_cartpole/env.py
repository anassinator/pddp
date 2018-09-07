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
"""Double cartpole environment."""

import time
import torch
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from ...envs import GymEnv
from .model import DoubleCartpoleDynamicsModel
from ...utils.encoding import StateEncoding
from ...utils.gaussian_variable import GaussianVariable
from ...utils.angular import augment_state, reduce_state


class DoubleCartpoleEnv(GymEnv):

    """Double cartpole environment.

    Note: This environment is preconstrained if the model is.
    """

    def __init__(self, model=None, dt=0.1, render=False):
        """Constructs a DoubleCartpoleEnv.

        Args:
            model (CartpoleDynamicsModel): Ground truth dynamics model.
            dt (float): Time difference (s). Only used if model isn't defined.
            render (bool): Whether to render the environment or not.
        """
        self.dt = dt
        if model is None:
            model = DoubleCartpoleDynamicsModel(dt)

        gym_env = _DoubleCartPoleEnv(model.eval())
        super(DoubleCartpoleEnv, self).__init__(gym_env, render=render)

    def apply(self, u):
        """Applies an action to the environment.

        Args:
            u (Tensor<action_size>): Action vector.
        """
        super(DoubleCartpoleEnv, self).apply(u)
        if self._render:
            time.sleep(self.dt)


class _DoubleCartPoleEnv(gym.Env):

    """Open AI gym double cartpole environment."""

    metadata = {
        "render.modes": ["human", "rgb_array"],
        "video.frames_per_second": 50,
    }

    def __init__(self, model):
        self.model = model.double()

        high = np.array([np.finfo(np.float32).max])
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            2 * np.pi,
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
        tensor_opts = {"dtype": torch.float64}

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
        self.state = np.array([0, 0, np.pi, 0, np.pi, 0])
        self.state += 1e-2 * np.random.random(self.state.shape)
        return self.state

    def render(self, mode="human"):
        screen_width = 1000
        screen_height = 600

        world_width = 5.0
        scale = screen_width / world_width
        carty = 200  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.6
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x, _, theta1, _, theta2, _ = self.state
        cartx = x * scale + screen_width / 2.0  # MIDDLE OF CART

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
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(0.8, 0.6, 0.4)
            self.pole1trans = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.pole1trans)
            pole1.add_attr(self.carttrans)
            self.viewer.add_geom(pole1)

            self.axle1 = rendering.make_circle(polewidth / 2)
            self.axle1.add_attr(self.pole1trans)
            self.axle1.add_attr(self.carttrans)
            self.axle1.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle1)

            l, r, t, b = (-polewidth / 2, polewidth / 2,
                          polelen - polewidth / 2, -polewidth / 2)
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(0.8, 0.6, 0.4)
            self.pole2trans = rendering.Transform(
                translation=(0, polelen - axleoffset))
            pole2.add_attr(self.pole2trans)
            pole2.add_attr(self.pole1trans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)

            self.axle2 = rendering.make_circle(polewidth / 2)
            self.axle2.add_attr(self.pole2trans)
            self.axle2.add_attr(self.pole1trans)
            self.axle2.add_attr(self.carttrans)
            self.axle2.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle2)

            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        self.carttrans.set_translation(cartx, carty)
        self.pole1trans.set_rotation(theta1)
        self.pole2trans.set_rotation(theta2 - theta1)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
