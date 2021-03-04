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
"""Constraint utilities."""

from __future__ import print_function

import torch
import traceback
from .encoding import StateEncoding

BOXQP_RESULTS = {
    -1: 'Hessian is not positive definite',
    0: 'No descent direction found',
    1: 'Maximum main iterations exceeded',
    2: 'Maximum line-search iterations exceeded',
    3: 'No bounds, returning Newton point',
    4: 'Improvement smaller than tolerance',
    5: 'Gradient norm smaller than tolerance',
    6: 'All dimensions are clamped',
}


def constrain(u, min_bounds, max_bounds):
    """Constrains the action through a tanh() squash function.

    Args:
        u (Tensor<action_size>): Action vector.
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Constrained action vector (Tensor<action_size>).
    """
    diff = (max_bounds - min_bounds) / 2.0
    mean = (max_bounds + min_bounds) / 2.0
    return diff * u.tanh() + mean


def constrain_env(min_bounds, max_bounds):
    """Decorator that constrains the action space of an environment through a
    squash function.

    Args:
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Decorator that constrains an Env.
    """
    def decorator(cls):
        def apply_fn(self, u):
            """Applies an action to the environment.

            Args:
                u (Tensor<action_size>): Action vector.
            """
            u = constrain(u, min_bounds, max_bounds)
            return _apply_fn(self, u)

        # Monkey-patch the env.
        _apply_fn = cls.apply
        cls.apply = apply_fn

        return cls

    return decorator


def constrain_model(min_bounds, max_bounds):
    """Decorator that constrains the action space of a dynamics model through a
    squash function.

    Args:
        min_bounds (Tensor<action_size>): Minimum action bounds.
        max_bounds (Tensor<action_size>): Maximum action bounds.

    Returns:
        Decorator that constrains a DynamicsModel.
    """
    def decorator(cls):
        def init_fn(self, *args, **kwargs):
            """Constructs a DynamicsModel."""
            _init_fn(self, *args, **kwargs)
            self.max_bounds = torch.nn.Parameter(torch.tensor(max_bounds).expand(cls.action_size), requires_grad=False)
            self.min_bounds = torch.nn.Parameter(torch.tensor(max_bounds).expand(cls.action_size), requires_grad=False)

        def forward_fn(self, z, u, i, encoding=StateEncoding.DEFAULT, **kwargs):
            """Dynamics model function.

            Args:
                z (Tensor<..., encoded_state_size>): Encoded state distribution.
                u (Tensor<..., action_size>): Action vector(s).
                i (Tensor<...>): Time index.
                encoding (int): StateEncoding enum.

            Returns:
                Next encoded state distribution
                    (Tensor<..., encoded_state_size>).
            """
            u = constrain(u, min_bounds, max_bounds)
            return _forward_fn(self, z, u, i, encoding=encoding, **kwargs)

        def constrain_fn(self, u):
            """Constrains an action through a squash function.

            Args:
                u (Tensor<..., action_size>): Action vector(s).

            Returns:
                Constrained action vector(s) (Tensor<..., action_size>).
            """
            return constrain(u, min_bounds, max_bounds)

        # Monkey-patch the model.
        _init_fn = cls.__init__
        _forward_fn = cls.forward
        cls.__init__ = init_fn
        cls.forward = forward_fn
        cls.constrain = constrain_fn

        return cls

    return decorator


def clamp(u, min_bounds, max_bounds):
    return torch.min(torch.max(u, min_bounds), max_bounds)


@torch.no_grad()
def boxqp(x0,
          Q,
          c,
          lower,
          upper,
          max_iter=100,
          min_grad=1e-8,
          tol=1e-8,
          step_dec=0.6,
          min_step=1e-22,
          armijo=0.1,
          quiet=True):
    """
        BoxQP solver for constraned Quadratic programs of the form:
            Min 0.5*x.t()*Q*x + x.t()*c  s.t. lower<=x<=upper
        Re-implementation of the MATLAB solver by Yuval Tassa
    """
    # algorithm state variables
    D = Q.shape[0]
    result = 0
    old_f = 0
    gnorm = 0
    clamped = torch.zeros(D, dtype=torch.uint8)
    free = torch.ones(D, dtype=torch.uint8)
    Ufree = torch.zeros(D)

    # initial x
    x = clamp(x0, lower, upper)
    x[torch.isinf(x)] = 0.0

    # initial objective
    f = 0.5 * x.matmul(Q).matmul(x) + x.matmul(c)

    # solver loop
    for i in range(max_iter):
        # check if done
        if result != 0:
            break

        # check convergence
        if i > 0 and (old_f - f) < tol * abs(old_f):
            result = 4
            break
        old_f = f

        # get gradient
        g = Q.matmul(x) + c

        # get clamped dimensions
        old_clamped = clamped.clone()
        clamped[:] = 0
        clamped[(x == lower) * (g > 0)] = 1
        clamped[(x == upper) * (g < 0)] = 1
        free = (1 - clamped).bool()

        # check if all clamped
        if all(clamped):
            result = 6
            break

        # check if we need to factorize
        if i == 0:
            factorize = True
        else:
            factorize = any(old_clamped != clamped)

        if factorize:
            # get  free dimensions
            Qfree = Q[free][:, free]
            # factorize
            try:
                Ufree = Qfree.cholesky()
            except RuntimeError:
                if not quiet:
                    print('Qfree not positive definite:\n', Qfree)
                    traceback.print_exc()
                result = -1
                break

        # check gradient norm
        gnorm = g[free].norm()
        if gnorm < min_grad:
            result = 5
            break

        # get search direction
        g_clamped = Q.matmul(x * clamped.to(x.dtype)) + c
        search = torch.zeros_like(x)
        search[free] = -torch.cholesky_solve(g_clamped[free].unsqueeze(-1), Ufree).flatten() - x[free]

        # check if descent direction
        sdotg = (search * g).sum()
        if sdotg > 0 and not quiet:
            print("BoxQP didn't find a descent direction (Should not happen)")
            break

        # line search
        step = 1.0
        n_step = 0
        xc = clamp(x + step * search, lower, upper)
        fc = 0.5 * xc.matmul(Q).matmul(xc) + xc.matmul(c)
        while (fc - old_f) / (step * sdotg) < armijo:
            step *= step_dec
            n_step += 1
            xc = clamp(x + step * search, lower, upper)
            fc = 0.5 * xc.matmul(Q).matmul(xc) + xc.matmul(c)
            if step < min_step:
                result = 2
                break

        # accept change
        x = xc
        f = fc
    if not quiet:
        print(BOXQP_RESULTS[result], 'n_iter: {}'.format(i))
    return x, result, Ufree, free
