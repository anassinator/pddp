"""Unit tests for pddp.models.bnn."""

import torch
import pytest

from pddp.models.bnn import *
from pddp.utils.autodiff import jacobian
from pddp.utils.encoding import StateEncoding
from pddp.utils.evaluation import eval_dynamics
from pddp.utils.gaussian_variable import GaussianVariable

N = 10
STATE_SIZE = 4
ACTION_SIZE = 2

STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_forward(encoding):
    model = bnn_dynamics_model_factory(STATE_SIZE, ACTION_SIZE, [10, 10])()

    X = [GaussianVariable.random(STATE_SIZE) for _ in range(N)]
    Z = torch.stack([x.encode(encoding) for x in X])
    U = torch.randn(N, ACTION_SIZE)

    Z_ = model(Z, U, 0, encoding)

    assert Z_.shape == Z.shape
    for z in Z_:
        for cost in z:
            torch.autograd.grad(cost, Z, retain_graph=True)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
@pytest.mark.parametrize("dropout_class", [BDropout, CDropout])
@pytest.mark.parametrize("use_predicted_std", [False, True])
def test_gradcheck(encoding, dropout_class, use_predicted_std):
    model = bnn_dynamics_model_factory(
        STATE_SIZE, ACTION_SIZE, [10, 10],
        dropout_layers=dropout_class)(n_particles=10).double()

    X = [GaussianVariable.random(STATE_SIZE) for _ in range(N)]
    Z = torch.stack([x.encode(encoding) for x in X]).double()
    U = torch.randn(N, ACTION_SIZE, requires_grad=True).double()

    z = Z[0]
    u = U[0]

    def deterministic_model(*args):
        # Needed for consistent gradients through finite differences.
        torch.random.manual_seed(4)
        return model(*args, use_predicted_std=use_predicted_std)

    # Force the model noise to be sampled once before the test.
    # Otherwise, the model will not be deterministic as the noise will only be
    # sampled on the first run.
    _ = deterministic_model(z, u, 0, encoding)

    assert torch.autograd.gradcheck(deterministic_model, (Z, U, 0, encoding))
    assert torch.autograd.gradcheck(deterministic_model, (z, u, 0, encoding))

    # Verify that batch jacobians are correct.
    z_next = deterministic_model(z, u, 0, encoding)
    F_z = jacobian(z_next, z)
    F_u = jacobian(z_next, u)
    z_next_, F_z_, F_u_ = eval_dynamics(deterministic_model, z, u, 0, encoding)

    assert z_next.allclose(z_next_)
    assert F_z.allclose(F_z_)
    assert F_u.allclose(F_u_)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_forward(benchmark, encoding):
    model = bnn_dynamics_model_factory(STATE_SIZE, ACTION_SIZE, [10, 10])()

    X = [GaussianVariable.random(STATE_SIZE) for _ in range(N)]
    Z = torch.stack([x.encode(encoding) for x in X])
    U = torch.randn(N, ACTION_SIZE)

    benchmark(model, Z, U, 0, encoding)
