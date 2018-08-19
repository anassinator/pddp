"""Unit tests for pddp.examples.*.env."""

import torch
import pytest

from pddp.examples import *
from pddp.models.utils.angular import infer_augmented_state_size

ENVS = [
    cartpole.CartpoleEnv,
    pendulum.PendulumEnv,
    rendezvous.RendezvousEnv,
]

MODELS = [
    cartpole.CartpoleDynamicsModel,
    pendulum.PendulumDynamicsModel,
    rendezvous.RendezvousDynamicsModel,
]


@pytest.mark.parametrize("env_class, model_class", zip(ENVS, MODELS))
def test_forward(env_class, model_class):
    with env_class() as env:
        model = model_class(0.1)
        u = torch.randn(env.action_size)
        env.apply(u)
        x = env.get_state()
        assert x.shape[0] == infer_augmented_state_size(
            model.angular_indices, model.non_angular_indices)
