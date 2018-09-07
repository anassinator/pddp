"""Unit tests for pddp.examples.*.env."""

import torch
import pytest

from pddp.examples import *

ENVS = [
    cartpole.CartpoleEnv,
    pendulum.PendulumEnv,
    rendezvous.RendezvousEnv,
    double_cartpole.DoubleCartpoleEnv,
]

MODELS = [
    cartpole.CartpoleDynamicsModel,
    pendulum.PendulumDynamicsModel,
    rendezvous.RendezvousDynamicsModel,
    double_cartpole.DoubleCartpoleDynamicsModel,
]


@pytest.mark.parametrize("env_class, model_class", zip(ENVS, MODELS))
def test_forward(env_class, model_class):
    with env_class() as env:
        u = torch.randn(env.action_size)
        env.apply(u)
        x = env.get_state()
        assert x.shape[0] == model_class.state_size
