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
"""OpenAI Gym environments."""

import gym
import torch
import numpy as np
from .base import Env
from ..utils.gaussian_variable import GaussianVariable


class GymEnv(Env):
    """OpenAI Gym environment."""
    def __init__(self, gym_env, render=False):
        """Constructs a GymEnv.

        Args:
            gym_env (gym.Env): Gym environment.
            render (bool): Whether to render the environment or not.
        """
        super(GymEnv, self).__init__()

        self._env = gym_env
        self._render = render

        self._action_size = _size_from_space(gym_env.action_space)
        self._action_shape = _shape_from_space(gym_env.action_space)
        self._action_dtype = _dtype_from_space(gym_env.action_space)
        self._action_bounds = _bounds_from_space(gym_env.action_space)

        self._state_size = _size_from_space(gym_env.observation_space)

        # Reset to get initial state.
        self._state = torch.zeros(self._state_size, requires_grad=True)
        self.reset()

    @property
    def action_size(self):
        """Action size (int)."""
        return self._action_size

    @property
    def state_size(self):
        """State size (int)."""
        return self._state_size

    def apply(self, u):
        """Applies an action to the environment.

        Args:
            u (Tensor<action_size>): Action vector.
        """
        action = _action_from_u(u, self._action_shape, self._action_dtype, self._action_bounds)
        obs, _, _, _ = self._env.step(action)
        self._state = _state_from_observation(obs)

        if self._render:
            self._env.render()

    def get_state(self, var=1e-2):
        """Gets the current state of the environment.

        Args:
            var (Tensor<0>): Variance scaling.

        Returns:
            State distribution (GaussianVariable<state_size>).
        """
        return GaussianVariable(self._state, var=var * torch.ones_like(self._state))

    def reset(self):
        """Resets the environment."""
        obs = self._env.reset()
        self._state = _state_from_observation(obs)

        if self._render:
            self._env.render()

    def close(self):
        """Stops the current environment session."""
        self._env.close()


def _action_from_u(u, space_shape, space_type, space_bounds):
    """Converts a torch action vector to the Gym environment equivalent.

    Args:
        u (Tensor<action_size>): Action vector.
        space_shape (Size): Gym action space shape.
        space_type (dtype): Gym action space dtype.
        space_bounds (Tuple<Tensor, Tensor>): Gym action min and max bounds.

    Returns:
        Gym action.
    """
    action = u.view(space_shape)

    min_bounds, max_bounds = space_bounds
    if action.dim():
        for a, min_a, max_a in zip(action, min_bounds, max_bounds):
            a = a.clamp(min_a, max_a)
    else:
        action = action.clamp(min_bounds[0], max_bounds[0])

    action = action.detach().cpu().numpy()
    return action.astype(space_type)


def _state_from_observation(obs):
    """Converts an observation to a flattened state vector.

    Args:
        obs (ndarray, int, or float): Observation.

    Returns:
        State vector (Tensor<state_size>).
    """
    if isinstance(obs, np.ndarray):
        state = obs.reshape(-1)
    elif isinstance(obs, (int, float, bool)):
        state = np.array([obs])
    else:
        raise NotImplementedError("Unsupported observation type: {}".format(type(obs)))

    state = torch.tensor(state)
    state = state.to(torch.get_default_dtype())
    state.requires_grad_()
    return state


def _bounds_from_space(space):
    """Computes the bounds of a Gym space.

    Args:
        space (object): Gym space.

    Returns:
        Tuple of:
            Minimum bounds (Tensor)
            Maximum bounds (Tensor)
    """
    if isinstance(space, gym.spaces.Box):
        return torch.tensor(space.low), torch.tensor(space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return torch.tensor([0]), torch.tensor([space.n])
    elif isinstance(space, gym.spaces.MultiDiscrete):
        return torch.zeros(space.shape).long(), torch.tensor(space.nvec)
    elif isinstance(space, gym.spaces.MultiBinary):
        return torch.zeros(space.shape).long(), torch.ones(space.shape).long()

    raise NotImplementedError("Unsupported space type: {}".format(type(space)))


def _dtype_from_space(space):
    """Computes the type of a Gym space.

    Args:
        space (object): Gym space.

    Returns:
        Space type (np.dtype).
    """
    x = space.sample()
    if isinstance(x, np.ndarray):
        return x.dtype
    elif isinstance(x, (int, float, bool)):
        return type(x)

    raise NotImplementedError("Unsupported space type: {}".format(type(x)))


def _size_from_space(space):
    """Computes the size of a Gym space.

    Args:
        space (object): Gym space.

    Returns:
        Space size (int).
    """
    x = space.sample()
    if isinstance(x, np.ndarray):
        return x.size
    elif isinstance(x, (int, float, bool)):
        return 1

    raise NotImplementedError("Unsupported space type: {}".format(type(x)))


def _shape_from_space(space):
    """Computes the shape of a Gym space.

    Args:
        space (object): Gym space.

    Returns:
        Space shape (torch.Size).
    """
    x = space.sample()
    if isinstance(x, np.ndarray):
        return torch.Size(x.shape)
    elif isinstance(x, (int, float, bool)):
        return torch.Size([])

    raise NotImplementedError("Unsupported space type: {}".format(type(x)))
