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
"""Iterative Linear Quadratic Regulator controller."""

import torch
import traceback
import warnings

with warnings.catch_warnings():
    # Ignore potential warning when in Jupyter environment.
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange

from enum import IntEnum

from .base import Controller
from ..utils.encoding import StateEncoding, decode_mean
from ..utils.constraint import clamp, boxqp, BOXQP_RESULTS
from ..utils.evaluation import (batch_eval_cost, batch_eval_dynamics, eval_cost,
                                eval_dynamics)


class iLQRState(IntEnum):

    """iLQR optimization step state."""

    # Undefined state.
    UNDEFINED = 0

    # Optimization step was accepted.
    ACCEPTED = 1

    # Optimization step was rejected due to insufficient descent.
    REJECTED = 2

    # Optimization step was rejected due to non-positive-definite Q_uu.
    NOT_PD = 3

    # Optimization step was rejected due to exceeding maximum regularization.
    MAX_REG = 4

    # Optimization step converged.
    CONVERGED = 5

    def should_retry(self):
        """Whether the state is expected to improve by retrying (bool)."""
        return self in (iLQRState.UNDEFINED, iLQRState.NOT_PD,
                        iLQRState.REJECTED)

    def is_terminal(self):
        """Whether the state is considered terminal (bool)."""
        return self in (iLQRState.CONVERGED, iLQRState.MAX_REG)


class iLQRController(Controller):

    """Iterative Linear Quadratic Regulator controller."""

    def __init__(self, env, model, cost, model_opts={}, cost_opts={}, **kwargs):
        """Constructs a iLQRController.

        Args:
            env (Env): Environment.
            model (DynamicsModel): Dynamics model.
            cost (Cost): Cost function.
            model_opts (dict): Additional key-word arguments to pass to
                `model()`.
            cost_opts (dict): Additional key-word arguments to pass to
                `cost()`.
        """
        super(iLQRController, self).__init__()

        self.env = env
        self.cost = cost
        self.model = model

        self._cost_opts = cost_opts
        self._model_opts = model_opts

        # Regularization terms.
        self._mu = 0.0
        self._mu_min = 1e-6
        self._delta_0 = 2.0
        self._delta = self._delta_0

        self._Z_nominal = None
        self._U_nominal = None
        self._K = None

    def _step(self,
              Z,
              U,
              F_z,
              F_u,
              L,
              L_z,
              L_u,
              L_zz,
              L_uz,
              L_uu,
              J_opt,
              encoding=StateEncoding.DEFAULT,
              max_reg=1e10,
              tol=5e-6,
              batch_rollout=True,
              alphas=10.0**torch.linspace(0, -3, 11),
              u_min=None,
              u_max=None,
              quiet=False):
        # Backward pass.
        try:
            k, K = backward(
                Z,
                F_z,
                F_u,
                L,
                L_z,
                L_u,
                L_zz,
                L_uz,
                L_uu,
                reg=self._mu,
                u_min=u_min,
                u_max=u_max,
                U=U,
                quiet=quiet)
        except RuntimeError:
            if not quiet:
                traceback.print_exc()
            if not self._increase_reg(max_reg):
                return iLQRState.MAX_REG, Z, U, J_opt
            return iLQRState.NOT_PD, Z, U, J_opt

        # Batch-backtracking line search.
        Z_new_b, U_new_b = _control_law(
            self.model,
            Z,
            U,
            k,
            K,
            alphas,
            encoding,
            self._model_opts,
            u_min=u_min,
            u_max=u_max)
        J_new_b = _trajectory_cost(self.cost, Z_new_b, U_new_b, encoding,
                                   self._cost_opts)
        amin = J_new_b.argmin()
        J_new = J_new_b[amin].detach()
        Z_new = Z_new_b[:, amin].detach()
        U_new = U_new_b[:, amin].detach()

        if J_new < J_opt:
            self._Z_nominal = Z_new
            self._U_nominal = U_new
            self._K = K
            self._decrease_reg()

            # Check if converged due to small change.
            if (J_opt - J_new).abs() / J_opt < tol:
                return iLQRState.CONVERGED, Z_new, U_new, J_new

            return iLQRState.ACCEPTED, Z_new, U_new, J_new

        if not self._increase_reg(max_reg):
            return iLQRState.MAX_REG, Z, U, J_opt

        return iLQRState.REJECTED, Z, U, J_opt

    def step(self,
             z0,
             U=None,
             i=0,
             encoding=StateEncoding.DEFAULT,
             max_reg=1e10,
             tol=5e-6,
             batch_rollout=True,
             alphas=10.0**torch.linspace(0, -3, 11),
             u_min=None,
             u_max=None,
             on_iteration=None,
             **kwargs):
        """Evaluates a single optimization step of iLQR."""
        if U is None:
            U = self._U_nominal

        Z, F_z, F_u, L, L_z, L_u, L_zz, L_uz, L_uu = forward(
            z0,
            U,
            self.model,
            self.cost,
            encoding,
            batch_rollout,
            self._model_opts,
            self._cost_opts,
            u_min=u_min,
            u_max=u_max)
        J_opt = L.sum()
        alphas = alphas.to(dtype=Z.dtype, device=Z.device)

        state = iLQRState.UNDEFINED
        while state.should_retry():
            state, Z, U, J_opt = self._step(
                Z, U, F_z, F_u, L, L_z, L_u, L_zz, L_uz, L_uu, J_opt, encoding,
                max_reg, tol, batch_rollout, alphas, u_min, u_max)
            if on_iteration:
                on_iteration(i, state, Z.detach(), U.detach(), J_opt.detach())

        return state

    def fit(self,
            U,
            encoding=StateEncoding.DEFAULT,
            n_iterations=50,
            tol=5e-6,
            max_reg=1e10,
            batch_rollout=True,
            quiet=False,
            on_iteration=None,
            u_min=None,
            u_max=None,
            **kwargs):
        """Determines the optimal path to minimize the cost.

        Args:
            U (Tensor<N, action_size>): Initial action path.
            encoding (int): StateEncoding enum.
            n_iterations (int): Maximum number of iterations until convergence.
            tol (Tensor<0>): Tolerance for early convergence.
            max_reg (Tensor<0>): Maximum regularization term.
            batch_rollout (bool): Whether to rollout in parallel or not.
            quiet (bool): Whether to print anything to screen or not.
            on_iteration (callable): Function with the following signature:
                Args:
                    iteration (int): Iteration index.
                    state (iLQRState): Latest iteration state.
                    Z (Tensor<N+1, encoded_state_size>): Iteration's encoded
                        state path.
                    U (Tensor<N, action_size>): Iteration's action path.
                    J_opt (Tensor<0>): Iteration's total cost to-go.

        Returns:
            Tuple of:
                Z (Tensor<N+1, encoded_state_size>): Optimal encoded state path.
                U (Tensor<N, action_size>): Optimal action path.
                state (iLQRState): Final optimization state.
        """
        self._U_nominal = U.detach()
        N, action_size = U.shape
        tensor_opts = {"dtype": U.dtype, "device": U.device}
        self._reset_reg()
        state = iLQRState.UNDEFINED

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 10.0**torch.linspace(0, -3, 11).to(**tensor_opts)
        alphas = 1.025**(-torch.arange(10.0)**2).to(**tensor_opts)

        # Get initial state distribution.
        z0 = self.env.get_state().encode(encoding).detach().to(**tensor_opts)

        with trange(n_iterations, desc="iLQR", disable=quiet) as pbar:

            def _on_iteration(iteration, state, Z, U, J_opt):
                pbar.set_postfix({
                    "state": state.name,
                    "loss": J_opt.cpu().numpy(),
                    "reg": self._mu,
                })
                if on_iteration:
                    on_iteration(iteration, state, Z, U, J_opt)

            for i in pbar:
                state = self.step(
                    z0,
                    None,
                    i,
                    encoding,
                    max_reg,
                    tol,
                    batch_rollout,
                    alphas,
                    u_min,
                    u_max,
                    on_iteration=_on_iteration)

                if state.is_terminal():
                    break

        return self._Z_nominal, self._U_nominal, state

    def forward(self,
                z,
                i,
                encoding=StateEncoding.DEFAULT,
                mpc=False,
                ignore_uncertainty=True,
                u_min=None,
                u_max=None,
                **kwargs):
        """Determines the optimal single-step control to minimize the cost.

        Note: You must `fit()` first.

        Args:
            z (Tensor<encoded_state_size>): Current encoded state distribution.
            i (int): Current time step.
            encoding (int): StateEncoding enum.

        Returns:
            Optimal action (Tensor<action_size>).
        """
        if not mpc:
            if self._U_nominal is None:
                raise RuntimeError(
                    "You need to either call fit or initialize _U_nominal")
            if self._Z_nominal is not None:
                if ignore_uncertainty:
                    x = decode_mean(z, encoding)
                    dx = x - decode_mean(self._Z_nominal[i], encoding)
                    D = x.shape[0]
                    du = self._K[i, :, :D].matmul(dx)
                else:
                    dz = z - self._Z_nominal[i]
                    du = self._K[i].matmul(dz)
                return self._U_nominal[i] + du
            else:
                return self._U_nominal[i]
        else:
            self._reset_reg()
            self.step(z, i=i, encoding=encoding, **kwargs)
            u = self._U_nominal[0]
            self._U_nominal = torch.cat(
                [self._U_nominal[1:], self._U_nominal[-1:]], 0)
            return u

    def _reset_reg(self):
        """Resets the regularization parameters."""
        self._mu = 0.0
        self._delta = self._delta_0

    def _decrease_reg(self):
        """Decreases the regularization parameters."""
        self._delta = min(1.0, self._delta) / self._delta_0
        self._mu *= self._delta
        if self._mu <= self._mu_min:
            self._mu = 0.0

    def _increase_reg(self, max_reg):
        """Increases the regularization parameters.

        Args:
            max_reg (float): Maximum regularization term.

        Returns:
            Whether the maximum regularization term has been exceeded (bool).
        """
        self._delta = max(1.0, self._delta) * self._delta_0
        self._mu = max(self._mu_min, self._mu * self._delta)
        if self._mu >= max_reg:
            warnings.warn("exceeded max regularization term")
            return False
        return True


def forward(z0,
            U,
            model,
            cost,
            encoding=StateEncoding.DEFAULT,
            batch_rollout=True,
            model_opts={},
            cost_opts={},
            u_min=None,
            u_max=None):
    """Evaluates the forward rollout.

    Args:
        z0 (Tensor<encoded_state_size>): Initial encoded state distribution.
        U (Tensor<N, action_size>): Initial action path.
        model (DynamicsModel): Dynamics model.
        cost (Cost): Cost function.
        encoding (int): StateEncoding enum.
        batch_rollout (bool): Whether to rollout in parallel or not.
        model_opts (dict): Additional key-word arguments to pass to `model()`.
        cost_opts (dict): Additional key-word arguments to pass to `cost()`.

    Returns:
        Tuple of
            Z (Tensor<N+1, encoded_state_size>): Encoded state path.
            F_z (Tensor<N, encoded_state_size, encoded_state_size>): Gradient
                of encoded state path w.r.t. previous encoded state.
            F_u (Tensor<N, encoded_state_size, action_size>): Gradient of
                encoded state path w.r.t. previous action.
            L (Tensor<N+1>): Cost path.
            L_z (Tensor<N+1, encoded_state_size>): Gradient of cost path
                w.r.t. encoded state path.
            L_u (Tensor<N, action_size>): Gradient of cost path w.r.t. action
                path.
            L_zz (Tensor<N+1, encoded_state_size, encoded_state_size>):
                Hessian of cost path w.r.t. encoded state.
            L_uz (Tensor<N+1, action_size, encoded_state_size>): Hessian of
                cost path w.r.t. action and encoded state.
            L_uu (Tensor<N+1, action_size, action_size>):
                Hessian of cost path w.r.t. action.
    """
    U.detach_()
    cost.eval()
    model.eval()

    eval_cost_fn = batch_eval_cost if batch_rollout else eval_cost
    eval_dynamics_fn = batch_eval_dynamics if batch_rollout else eval_dynamics

    N, action_size = U.shape
    encoded_state_size = z0.shape[-1]
    tensor_opts = {"dtype": z0.dtype, "device": z0.device}

    Z = torch.empty(N + 1, encoded_state_size, **tensor_opts)
    F_z = torch.empty(N, encoded_state_size, encoded_state_size, **tensor_opts)
    F_u = torch.empty(N, encoded_state_size, action_size, **tensor_opts)

    L = torch.empty(N + 1, **tensor_opts)
    L_z = torch.empty(N + 1, encoded_state_size, **tensor_opts)
    L_u = torch.empty(N, action_size, **tensor_opts)
    L_zz = torch.empty(N + 1, encoded_state_size, encoded_state_size,
                       **tensor_opts)
    L_uz = torch.empty(N, action_size, encoded_state_size, **tensor_opts)
    L_uu = torch.empty(N, action_size, action_size, **tensor_opts)

    Z[0] = z0
    for i in range(N):
        z = Z[i].detach().requires_grad_()
        u = U[i].detach().requires_grad_()
        if u_min is not None and u_max is not None:
            u = clamp(u, u_min, u_max)

        L[i], L_z[i], L_u[i], L_zz[i], L_uz[i], L_uu[i] = eval_cost_fn(
            cost, z, u, i, encoding=encoding, **cost_opts)

        Z[i + 1], F_z[i], F_u[i] = eval_dynamics_fn(
            model, z, u, i, encoding=encoding, **model_opts)

    # Terminal cost.
    z = Z[-1].detach().requires_grad_()
    L[-1], L_z[-1], _, L_zz[-1], _, _ = eval_cost_fn(
        cost, z, None, i, terminal=True, encoding=encoding, **cost_opts)

    Z.detach_()
    F_z.detach_()
    F_u.detach_()

    L.detach_()
    L_z.detach_()
    L_u.detach_()
    L_zz.detach_()
    L_uz.detach_()
    L_uu.detach_()

    return Z, F_z, F_u, L, L_z, L_u, L_zz, L_uz, L_uu


@torch.no_grad()
def Q(F_z, F_u, L_z, L_u, L_zz, L_uz, L_uu, V_z, V_zz):
    """Evaluates the first- and second-order derivatives of the Q-function.

    Args:
        F_z (Tensor<encoded_state_size, encoded_state_size>): Gradient of
            encoded state w.r.t. previous encoded state.
        F_u (Tensor<encoded_state_size, action_size>): Gradient of encoded
            state w.r.t. previous action.
        L_z (Tensor<encoded_state_size>): Gradient of cost path w.r.t.
            encoded state path.
        L_u (Tensor<action_size>): Gradient of cost w.r.t. action.
        L_zz (Tensor<encoded_state_size, encoded_state_size>): Hessian of cost
            w.r.t. encoded state.
        L_uz (Tensor<action_size, encoded_state_size>): Hessian of cost w.r.t.
            action and encoded state.
        L_uu (Tensor<action_size, action_size>): Hessian of cost w.r.t. action.
        V_z (Tensor<encoded_state_size>): Gradient of value function w.r.t.
            encoded state.
        V_zz (Tensor<encoded_state_size, encoded_state_size>): Hessian of value
            function w.r.t. encoded state.

    Returns:
        Tuple of first- and second-order derivatives of Q-function:
            Q_z (Tensor<encoded_state_size>)
            Q_u (Tensor<action_size>)
            Q_zz (Tensor<encoded_state_size, encoded_state_size>)
            Q_uz (Tensor<action_size, encoded_state_size>)
            Q_uu (Tensor<action_size, action_size>)
    """
    Q_z = L_z + F_z.t().matmul(V_z)
    Q_u = L_u + F_u.t().matmul(V_z)
    Q_zz = L_zz + F_z.t().mm(V_zz).mm(F_z)
    Q_zz = 0.5 * (Q_zz + Q_zz.t())  # To maintain symmetry.
    Q_uz = L_uz + F_u.t().mm(V_zz).mm(F_z)
    Q_uu = L_uu + F_u.t().mm(V_zz).mm(F_u)
    Q_uu = 0.5 * (Q_uu + Q_uu.t())  # To maintain symmetry.
    return Q_z, Q_u, Q_zz, Q_uz, Q_uu


@torch.no_grad()
def backward(Z,
             F_z,
             F_u,
             L,
             L_z,
             L_u,
             L_zz,
             L_uz,
             L_uu,
             reg=0.0,
             V_zz_reg=False,
             u_min=None,
             u_max=None,
             U=None,
             quiet=False):
    """Evaluates the backward pass.

    Args:
        Z (Tensor<N+1, encoded_state_size>): Encoded state path.
        F_z (Tensor<N, encoded_state_size, encoded_state_size>): Gradient of
            encoded state path w.r.t. previous encoded state.
        F_u (Tensor<N, encoded_state_size, action_size>): Gradient of encoded
            state path w.r.t. previous action.
        L (Tensor<N+1>): Cost path.
        L_z (Tensor<N+1, encoded_state_size>): Gradient of cost path w.r.t.
            encoded state path.
        L_u (Tensor<N, action_size>): Gradient of cost path w.r.t. action path.
        L_zz (Tensor<N+1, encoded_state_size, encoded_state_size>): Hessian of
            cost path w.r.t. encoded state.
        L_uz (Tensor<N+1, action_size, encoded_state_size>): Hessian of cost
            path w.r.t. action and encoded state.
        L_uu (Tensor<N+1, action_size, action_size>): Hessian of cost path
            w.r.t. action.
        reg (float): Regularization term to guarantee positive-definiteness.
        V_zz_reg (bool): Whether to regularize V_zz instead of Q_uu directly.

    Returns:
        Tuple of
            k (Tensor<N, action_size>): Feedforward gains.
            K (Tensor<N, action_size, encoded_state_size>): Feedback gains.

    Raises:
        RuntimeError: If Q_uu is not positive-definite.
    """
    V_z = L_z[-1]
    V_zz = L_zz[-1]

    N, action_size = L_u.shape
    encoded_state_size = Z.shape[1]

    tensor_opts = {"dtype": Z.dtype, "device": Z.device}
    k = torch.zeros(N, action_size, **tensor_opts)
    K = torch.zeros(N, action_size, encoded_state_size, **tensor_opts)

    if V_zz_reg:
        reg = reg * torch.eye(encoded_state_size, **tensor_opts)

        for i in range(N - 1, -1, -1):
            Q_z, Q_u, Q_zz, Q_uz, Q_uu = Q(F_z[i], F_u[i], L_z[i], L_u[i],
                                           L_zz[i], L_uz[i], L_uu[i], V_z,
                                           V_zz + reg)

            Q_uu_chol = Q_uu.potrf()
            Q_uz_u = torch.cat([Q_u.unsqueeze(1), Q_uz], dim=-1)
            kK = -Q_uz_u.potrs(Q_uu_chol)
            k[i] = kK[:, 0]
            K[i] = kK[:, 1:]

            V_z = Q_z + K[i].t().matmul(Q_u)
            V_z += K[i].t().mm(Q_uu).matmul(k[i])
            V_z += Q_uz.t().matmul(k[i])

            V_zz = Q_zz + K[i].t().mm(Q_uu).mm(K[i])
            V_zz += K[i].t().mm(Q_uz) + Q_uz.t().mm(K[i])
            V_zz = 0.5 * (V_zz + V_zz.t())  # To maintain symmetry.
    else:
        for i in range(N - 1, -1, -1):
            Q_z, Q_u, Q_zz, Q_uz, Q_uu = Q(F_z[i], F_u[i], L_z[i], L_u[i],
                                           L_zz[i], L_uz[i], L_uu[i], V_z, V_zz)

            e_uu, E_uu = Q_uu.eig(True)
            e_uu = e_uu[:, 0]
            e_uu[e_uu < 0] = 1e-12
            e_uu += reg
            if u_min is None or u_max is None:
                Q_uu_inv = (E_uu / e_uu).mm(E_uu.t())
                Q_uz_u = torch.cat([Q_u.unsqueeze(1), Q_uz], dim=-1)
                kK = -Q_uu_inv.mm(Q_uz_u)
                if torch.isnan(kK).any():
                    raise RuntimeError("non-positive definite matrix")

                k[i] = kK[:, 0]
                K[i] = kK[:, 1:]
            else:
                Q_uu = (E_uu * e_uu).mm(E_uu.t())
                # solve QP
                lower = u_min - U[i]
                upper = u_max - U[i]
                k_ip1 = k[i + 1] if i < N - 1 else k[-1]
                k[i], result, Q_uu_chol_free, free = boxqp(
                    k_ip1, Q_uu, Q_u, lower, upper, quiet=True)

                if result < 1:
                    raise RuntimeError("BoxQP failed: {}".format(
                        BOXQP_RESULTS[result]))

                # compute feedback matrix
                if any(free):
                    idxs = free.nonzero().flatten()
                    Q_uz_free = Q_uz[idxs, :]
                    K_free = -torch.potrs(Q_uz_free, Q_uu_chol_free)
                    K[i, idxs, :] = K_free

            V_z = Q_z + Q_uz.t().matmul(k[i])
            V_zz = Q_zz + Q_uz.t().mm(K[i])
            V_zz = 0.5 * (V_zz + V_zz.t())  # To maintain symmetry.

    return k, K


@torch.no_grad()
def _control_law(model,
                 Z,
                 U,
                 k,
                 K,
                 alpha,
                 encoding=StateEncoding.DEFAULT,
                 model_opts={},
                 u_min=None,
                 u_max=None):
    Z_new = torch.empty_like(Z)
    U_new = torch.empty_like(U)
    Z_new[0] = Z[0]
    model.eval()

    if alpha.numel() > 1:
        # Make alpha a column vector
        alpha = alpha.flatten().unsqueeze(-1)

        # Repeat Z_new and U_new (once per alpha)
        Z_new = Z_new.repeat(alpha.shape[0], 1, 1).transpose(0, 1)
        U_new = U_new.repeat(alpha.shape[0], 1, 1).transpose(0, 1)

    for i in range(U.shape[0]):
        z = Z[i]
        u = U[i]
        z_new = Z_new[i]

        # If alpha is a column vector, dz and du will be matrices
        dz = z_new - z
        du = alpha * k[i]
        if alpha.numel() > 1:
            du = du + dz.mm(K[i].t())
        else:
            du = du + K[i].matmul(dz)

        u_new = u + du
        if u_min is not None and u_max is not None:
            u_new = clamp(u_new, u_min, u_max)

        z_new_next = model(z_new, u_new, i, encoding, **model_opts)

        Z_new[i + 1] = z_new_next
        U_new[i] = u_new

    return Z_new, U_new


@torch.no_grad()
def _linear_control_law(Z, U, F_z, F_u, k, K, alpha, u_min=None, u_max=None):
    Z_new = torch.empty_like(Z)
    U_new = torch.empty_like(U)
    Z_new[0] = Z[0]

    if alpha.numel() > 1:
        # Make alpha a column vector
        alpha = alpha.flatten.unsqueeze(-1)

        # Repeat Z_new and U_new (once per alpha)
        Z_new = Z_new.repeat(alpha.shape[0], 1, 1).transpose(0, 1)
        U_new = U_new.repeat(alpha.shape[0], 1, 1).transpose(0, 1)

    for i in range(U.shape[0]):
        z = Z[i]
        u = U[i]
        z_new = Z_new[i]

        dz = z_new - z
        du = alpha * k[i]
        if alpha.numel() > 1:
            du = du + dz.mm(K[i].t())
            if u_min is not None and u_max is not None:
                du = clamp(du, u_min - u, u_max - u)
            dz_ = F_z[i].matmul(dz) + F_u[i].matmul(du)
        else:
            du = du + K[i].matmul(dz)
            if u_min is not None and u_max is not None:
                du = clamp(du, u_min - u, u_max - u)
            dz_ = dz.mm(F_z[i].t()) + du.mm(F_u[i].t())

        Z_new[i + 1] = Z[i + 1] + dz_
        U_new[i] = u + du

    return Z_new, U_new


@torch.no_grad()
def _trajectory_cost(cost, Z, U, encoding=StateEncoding.DEFAULT, cost_opts={}):
    cost.eval()
    N = torch.tensor(U.shape[0])
    I = torch.arange(N)
    batch_shape = None

    Z_run = Z[:-1]
    Z_end = Z[-1]
    if Z.dim() > 2:
        batch_shape = Z.shape[1:-1]
        numel = torch.tensor(batch_shape).prod()

        # We assume the first dimension is time
        Z_run = Z_run.reshape(-1, Z.shape[-1])
        Z_end = Z_end.reshape(-1, Z.shape[-1])
        U = U.reshape(-1, U.shape[-1])
        I = I.repeat(numel)
        N = N.repeat(numel)

    L = cost(Z_run, U, I, terminal=False, encoding=encoding, **cost_opts)
    l_f = cost(Z_end, None, N, terminal=True, encoding=encoding, **cost_opts)
    if batch_shape is not None:
        L = L.reshape(N[0], *batch_shape)
        l_f = l_f.reshape(*batch_shape)
        return L.sum(0) + l_f

    return L.sum() + l_f
