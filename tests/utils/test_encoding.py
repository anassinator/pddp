"""Unit tests for pddp.utils.encoding."""

import torch
import pytest

from pddp.utils.encoding import *
from pddp.utils.gaussian_variable import GaussianVariable

STATE_ENCODINGS = [
    StateEncoding.FULL_COVARIANCE_MATRIX,
    StateEncoding.UPPER_TRIANGULAR_CHOLESKY,
    StateEncoding.VARIANCE_ONLY,
    StateEncoding.STANDARD_DEVIATION_ONLY,
    StateEncoding.IGNORE_UNCERTAINTY,
]


@pytest.mark.parametrize("encoding, size",
                         zip(STATE_ENCODINGS, [30, 20, 10, 10, 5]))
def test_infer_encoded_state_size(encoding, size):
    x = GaussianVariable.random(5)
    z = x.encode(encoding)
    assert infer_encoded_state_size(x, encoding) == size
    assert infer_encoded_state_size(x, encoding) == z.shape[0]

    X = torch.stack([torch.randn(5) for _ in range(3)])
    assert infer_encoded_state_size(X, encoding) == size
    assert infer_encoded_state_size(X, encoding) == z.shape[0]


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_infer_state_sze(encoding):
    N = 5
    x = GaussianVariable.random(N)
    z = x.encode(encoding)
    assert infer_state_size(z, encoding) == N

    Z = torch.stack([z for _ in range(20)])
    assert infer_state_size(Z, encoding) == N


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_gaussian_variable_encode_decode(encoding):
    N = 20
    x = GaussianVariable.random(N)

    z = x.encode(encoding)
    y = GaussianVariable.decode(z, encoding)

    assert (x.mean() == y.mean()).all()

    if encoding == StateEncoding.FULL_COVARIANCE_MATRIX:
        assert x.covar().isclose(y.covar(), 1e-3).all()
    elif encoding == StateEncoding.UPPER_TRIANGULAR_CHOLESKY:
        assert x.covar().isclose(y.covar(), 1e-3).all()
    elif encoding == StateEncoding.VARIANCE_ONLY:
        assert x.var().isclose(y.var(), 1e-3).all()
    elif encoding == StateEncoding.STANDARD_DEVIATION_ONLY:
        assert x.std().isclose(y.std(), 1e-3).all()
    elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
        assert y.covar().isclose(1e-6 * torch.eye(N), 1e-3).all()
    else:
        raise NotImplementedError

    # Verify differentiable.
    for i in range(z.shape[0]):
        torch.autograd.grad(
            z[i], x.mean(), allow_unused=True, retain_graph=True)
        torch.autograd.grad(
            z[i], x.covar(), allow_unused=True, retain_graph=True)

    for i in range(N):
        for cost in (y.mean(), y.covar().view(-1), y.var(), y.std()):
            torch.autograd.grad(
                cost[i], z, allow_unused=True, retain_graph=True)
            torch.autograd.grad(
                cost[i], x.mean(), allow_unused=True, retain_graph=True)
            torch.autograd.grad(
                cost[i], x.covar(), allow_unused=True, retain_graph=True)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_encode(encoding):
    n = 3
    m = 5

    X = [GaussianVariable.random(m) for _ in range(n)]
    M = torch.stack([x.mean() for x in X])
    C = torch.stack([x.covar() for x in X])
    V = torch.stack([x.var() for x in X])
    S = torch.stack([x.std() for x in X])

    encoded_state_size = infer_encoded_state_size(M, encoding)

    Zs = [
        (encode(M, C=C, encoding=encoding), C, True),
        (encode(M, V=V, encoding=encoding), V, False),
        (encode(M, S=S, encoding=encoding), S, False),
    ]
    for Z, wrt, lossless in Zs:
        assert Z.shape == (n, encoded_state_size)

        M_ = decode_mean(Z, encoding)
        C_ = decode_covar(Z, encoding)
        V_ = decode_var(Z, encoding)
        S_ = decode_std(Z, encoding)

        assert (M_ == M).all()

        if encoding == (StateEncoding.IGNORE_UNCERTAINTY):
            pass
        elif lossless and encoding in (StateEncoding.FULL_COVARIANCE_MATRIX,
                                       StateEncoding.UPPER_TRIANGULAR_CHOLESKY):
            assert C_.isclose(C, 1e-3).all()
            assert V_.isclose(V, 1e-3).all()
            assert S_.isclose(S, 1e-3).all()
        elif not lossless or encoding in (
                StateEncoding.VARIANCE_ONLY,
                StateEncoding.STANDARD_DEVIATION_ONLY):
            ix, iy = np.diag_indices(m)
            assert C_[..., ix, iy].isclose(C[..., ix, iy], 1e-3).all()
            assert V_.isclose(V, 1e-3).all()
            assert S_.isclose(S, 1e-3).all()
        else:
            raise NotImplementedError

        allow_unused = encoding == StateEncoding.IGNORE_UNCERTAINTY
        for z in Z:
            for cost in z:
                torch.autograd.grad(
                    cost, wrt, retain_graph=True, allow_unused=allow_unused)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_decode_mean_covar_var_std(encoding):
    N = 3
    M = 5

    X = [GaussianVariable.random(M) for _ in range(N)]
    original_mean = torch.stack([x.mean() for x in X])
    original_covar = torch.stack([x.covar() for x in X])
    Z = encode(original_mean, C=original_covar, encoding=encoding)

    expected_mean = torch.stack([x.mean() for x in X])
    expected_covar = torch.stack([x.covar() for x in X])
    expected_var = torch.stack([x.var() for x in X])
    expected_std = torch.stack([x.std() for x in X])

    mean = decode_mean(Z, encoding)
    covar = decode_covar(Z, encoding)
    covar_sqrt = decode_covar_sqrt(Z, encoding)
    var = decode_var(Z, encoding)
    std = decode_std(Z, encoding)

    assert mean.shape == (N, M)
    assert covar.shape == (N, M, M)
    assert var.shape == (N, M)
    assert std.shape == (N, M)

    assert covar_sqrt.transpose(-2, -1).matmul(covar_sqrt).allclose(covar, 1e-3)

    assert mean.isclose(expected_mean).all()

    if encoding in (StateEncoding.FULL_COVARIANCE_MATRIX,
                    StateEncoding.UPPER_TRIANGULAR_CHOLESKY):
        assert covar.isclose(expected_covar, 1e-3).all()
        assert var.isclose(expected_var, 1e-3).all()
        assert std.isclose(expected_std, 1e-3).all()
    elif encoding in (StateEncoding.VARIANCE_ONLY,
                      StateEncoding.STANDARD_DEVIATION_ONLY):
        for c, expected_c in zip(covar, expected_covar):
            assert c.diag().isclose(expected_c.diag(), 1e-3).all()
        assert var.isclose(expected_var, 1e-3).all()
        assert std.isclose(expected_std, 1e-3).all()
    elif encoding == StateEncoding.IGNORE_UNCERTAINTY:
        for c in covar:
            assert c.isclose(1e-6 * torch.eye(M), 1e-3).all()
        assert var.isclose(1e-6 * torch.ones_like(var), 1e-3).all()
        assert std.isclose(1e-3 * torch.ones_like(std), 1e-3).all()
    else:
        raise NotImplementedError

    # Verify differentiable.
    allow_unused = encoding == StateEncoding.IGNORE_UNCERTAINTY
    for i in range(N):
        costs = mean, var, std
        wrts = original_mean, original_covar, original_covar

        for j in range(M):
            for k in range(M):
                torch.autograd.grad(
                    covar[i, j, k],
                    original_covar,
                    allow_unused=allow_unused,
                    retain_graph=True)

            for cost, wrt in zip(costs, wrts):
                torch.autograd.grad(
                    cost[i, j],
                    wrt,
                    allow_unused=allow_unused,
                    retain_graph=True)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_encode_covar(benchmark, encoding):
    N = 3
    M = 5

    X = [GaussianVariable.random(M) for _ in range(N)]
    M = torch.stack([x.mean() for x in X])
    C = torch.stack([x.covar() for x in X])

    benchmark(encode, M, C=C, encoding=encoding)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_encode_var(benchmark, encoding):
    N = 3
    M = 5

    X = [GaussianVariable.random(M) for _ in range(N)]
    M = torch.stack([x.mean() for x in X])
    V = torch.stack([x.var() for x in X])

    benchmark(encode, M, V=V, encoding=encoding)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_encode_std(benchmark, encoding):
    N = 3
    M = 5

    X = [GaussianVariable.random(M) for _ in range(N)]
    M = torch.stack([x.mean() for x in X])
    S = torch.stack([x.std() for x in X])

    benchmark(encode, M, S=S, encoding=encoding)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_decode_mean(benchmark, encoding):
    N = 3
    M = 5

    X = [GaussianVariable.random(M) for _ in range(N)]
    M = torch.stack([x.mean() for x in X])
    C = torch.stack([x.covar() for x in X])
    Z = encode(M, C=C, encoding=encoding)

    benchmark(decode_mean, Z, encoding=encoding)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_decode_covar(benchmark, encoding):
    N = 3
    M = 5

    X = [GaussianVariable.random(M) for _ in range(N)]
    M = torch.stack([x.mean() for x in X])
    C = torch.stack([x.covar() for x in X])
    Z = encode(M, C=C, encoding=encoding)

    benchmark(decode_covar, Z, encoding=encoding)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_decode_var(benchmark, encoding):
    N = 3
    M = 5

    X = [GaussianVariable.random(M) for _ in range(N)]
    M = torch.stack([x.mean() for x in X])
    V = torch.stack([x.var() for x in X])
    Z = encode(M, V=V, encoding=encoding)

    benchmark(decode_var, Z, encoding=encoding)


@pytest.mark.parametrize("encoding", STATE_ENCODINGS)
def test_benchmark_decode_std(benchmark, encoding):
    N = 3
    M = 5

    X = [GaussianVariable.random(M) for _ in range(N)]
    M = torch.stack([x.mean() for x in X])
    S = torch.stack([x.std() for x in X])
    Z = encode(M, S=S, encoding=encoding)

    benchmark(decode_std, Z, encoding=encoding)
