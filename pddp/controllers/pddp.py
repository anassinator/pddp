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
"""PDDP controller."""

import torch
import warnings

with warnings.catch_warnings():
    # Ignore potential warning when in Jupyter environment.
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange

from .base import Controller
from .utils.evaluation import eval_cost, eval_dynamics
from ..models.utils.encoding import StateEncoding, decode_var


class PDDPController(Controller):

    """PDDP controller."""

    def __init__(self,
                 env,
                 model,
                 cost,
                 model_opts={},
                 cost_opts={},
                 training_opts={}):
        """Constructs a PDDPController.

        Args:
            env (Env): Environment.
            model (DynamicsModel): Dynamics model.
            cost (Cost): Cost function.
            model_opts (dict): Additional key-word arguments to pass to
                `model()`.
            cost_opts (dict): Additional key-word arguments to pass to
                `cost()`.
            training_opts (dict): Additional key-word arguments to pass to
                `model.fit()`.
        """
        super(PDDPController, self).__init__()

        self._env = env
        self._cost = cost
        self._model = model

        self._cost_opts = cost_opts
        self._model_opts = model_opts
        self._training_opts = training_opts

        self._is_training = False

        # Regularization terms.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._delta_0 = 2.0
        self._delta = self._delta_0

    def train(self):
        """Switches to training mode."""
        self._is_training = True

    def eval(self):
        """Switches to evaluation mode."""
        self._is_training = False

    def fit(self,
            U,
            encoding=StateEncoding.DEFAULT,
            n_iterations=50,
            tol=torch.tensor(5e-6),
            max_reg=1e10,
            max_var=1.0,
            n_sample_trajectories=4,
            quiet=False,
            on_iteration=None,
            train_on_start=True,
            **kwargs):
        """Determines the optimal path to minimize the cost.

        Args:
            U (Tensor<N, action_size>): Initial action path.
            encoding (int): StateEncoding enum.
            n_iterations (int): Maximum number of iterations until convergence.
            tol (Tensor<0>): Tolerance for early convergence.
            max_reg (Tensor<0>): Maximum regularization term.
            max_var (Tensor<0>): Maximum variance allowed before resampling
                trajectories.
            n_sample_trajectories (int): Number of trajectories to sample from
                the environment at a time.
            quiet (bool): Whether to print anything to screen or not.
            on_iteration (callable): Function with the following signature:
                Args:
                    iteration (int): Iteration index.
                    Z (Tensor<N+1, encoded_state_size>): Iteration's encoded
                        state path.
                    U (Tensor<N, action_size>): Iteration's action path.
                    J_opt (Tensor<0>): Iteration's total cost to-go.
                    accepted (bool): Whether the iteration was accepted or not.
                    converged (bool): Whether the iteration converged or not.

        Returns:
            Tuple of:
                Z (Tensor<N+1, encoded_state_size>): Optimal encoded state path.
                U (Tensor<N, action_size>): Optimal action path.
        """
        U = U.detach()
        N, action_size = U.shape

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-torch.arange(10.0)**2)

        while True:
            # Reset regularization term.
            self._mu = 1.0
            self._delta = self._delta_0

            if self._is_training and train_on_start:
                # Sample trajectories and train model.
                Us = U.repeat(n_sample_trajectories, 1, 1)
                Us[1:] += torch.randn_like(Us[1:])
                X_, dX = _sample(self._env, Us, quiet)

                self._model.fit(X_, dX, quiet=quiet, **self._training_opts)

            # Get initial state distribution.
            z0 = self._env.get_state().encode(encoding).detach()
            encoded_state_size = z0.shape[-1]

            changed = True
            converged = False
            train_on_start = True
            pbar = trange(n_iterations, desc="PDDP", disable=quiet)
            for i in pbar:
                accepted = False

                # Forward rollout only if it needs to be recomputed.
                if changed:
                    Z, F_z, F_u, L, L_z, L_u, L_zz, L_uz, L_uu = forward(
                        z0, U, self._model, self._cost, encoding,
                        self._model_opts, self._cost_opts)
                    J_opt = L.sum()
                    changed = False

                # Backward pass.
                k, K = backward(
                    Z, F_z, F_u, L, L_z, L_u, L_zz, L_uz, L_uu, reg=self._mu)

                # Backtracking line search.
                for alpha in alphas:
                    Z_new, U_new = _control_law(self._model, Z, U, k, K, alpha,
                                                encoding, self._model_opts)
                    J_new = _trajectory_cost(self._cost, Z_new, U_new, encoding,
                                             self._cost_opts)

                    if J_new < J_opt:
                        # Check if converged due to small change.
                        if (J_opt - J_new).abs() / J_opt < tol:
                            converged = True

                        J_opt = J_new
                        Z = Z_new
                        U = U_new
                        changed = True

                        # Decrease regularization term.
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0

                        accepted = True
                        break

                pbar.set_postfix({
                    "loss": J_opt.detach().cpu().numpy(),
                    "reg": self._mu,
                    "accepted": accepted,
                })

                if on_iteration:
                    on_iteration(i, Z, U, J_opt, accepted, converged)

                if not accepted:
                    # Increase regularization term.
                    self._delta = max(1.0, self._delta) * self._delta_0
                    self._mu = max(self._mu_min, self._mu * self._delta)
                    if self._mu >= max_reg:
                        warnings.warn("exceeded max regularization term")
                        break

                if converged:
                    break

            if not self._is_training:
                break
            else:
                # Check if uncertainty is satisfactory.
                var = decode_var(Z, encoding=encoding)
                if (var < max_var).all():
                    break

        return Z, U


def forward(z0,
            U,
            model,
            cost,
            encoding=StateEncoding.DEFAULT,
            model_opts={},
            cost_opts={}):
    """Evaluates the forward rollout.

    Args:
        z0 (Tensor<encoded_state_size>): Initial encoded state distribution.
        U (Tensor<N, action_size>): Initial action path.
        model (DynamicsModel): Dynamics model.
        cost (Cost): Cost function.
        encoding (int): StateEncoding enum.
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
    N, action_size = U.shape
    encoded_state_size = z0.shape[-1]
    dtype = z0.dtype

    Z = torch.empty(N + 1, encoded_state_size, dtype=dtype)
    F_z = torch.empty(N, encoded_state_size, encoded_state_size, dtype=dtype)
    F_u = torch.empty(N, encoded_state_size, action_size, dtype=dtype)

    L = torch.empty(N + 1, dtype=dtype)
    L_z = torch.empty(N + 1, encoded_state_size, dtype=dtype)
    L_u = torch.empty(N, action_size, dtype=dtype)
    L_zz = torch.empty(
        N + 1, encoded_state_size, encoded_state_size, dtype=dtype)
    L_uz = torch.empty(N, action_size, encoded_state_size, dtype=dtype)
    L_uu = torch.empty(N, action_size, action_size, dtype=dtype)

    Z[0] = z0
    for i in range(N):
        z = Z[i].detach()
        u = U[i].detach()
        z.requires_grad_()
        u.requires_grad_()

        L[i], L_z[i], L_u[i], L_zz[i], L_uz[i], L_uu[i] = eval_cost(
            cost, z, u, i, encoding=encoding, **cost_opts)

        Z[i + 1], F_z[i], F_u[i] = eval_dynamics(
            model, z, u, i, encoding=encoding, **model_opts)

    # Terminal cost.
    z = Z[-1].detach()
    z.requires_grad_()
    L[-1], L_z[-1], _, L_zz[-1], _, _ = eval_cost(
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
    Q_uz = L_uz + F_u.t().mm(V_zz).mm(F_z)
    Q_uu = L_uu + F_u.t().mm(V_zz).mm(F_u)
    return Q_z, Q_u, Q_zz, Q_uz, Q_uu


def backward(Z, F_z, F_u, L, L_z, L_u, L_zz, L_uz, L_uu, reg=0.0):
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
        reg (float): Regularization term to guarantee V_zz
            positive-definiteness.

    Returns:
        Tuple of
            k (Tensor<N, action_size>): Feedforward gains.
            K (Tensor<N, action_size, encoded_state_size>): Feedback gains.
    """
    V_z = L_z[-1]
    V_zz = L_zz[-1]
    reg = reg * torch.eye(V_zz.shape[0], dtype=V_zz.dtype)

    N, action_size = L_u.shape
    encoded_state_size = Z.shape[1]

    k = torch.empty(N, action_size, dtype=Z.dtype)
    K = torch.empty(N, action_size, encoded_state_size, dtype=Z.dtype)

    for i in range(N - 1, -1, -1):
        Q_z, Q_u, Q_zz, Q_uz, Q_uu = Q(F_z[i], F_u[i], L_z[i], L_u[i], L_zz[i],
                                       L_uz[i], L_uu[i], V_z, V_zz + reg)

        try:
            k[i] = -Q_u.gesv(Q_uu)[0].view(-1)
            K[i] = -Q_uz.gesv(Q_uu)[0]
        except RuntimeError as e:
            # Fallback to pseudo-inverse.
            warnings.warn("singular matrix: falling back to pseudo-inverse")
            Q_uu_inv = Q_uu.pinverse()
            k[i] = -Q_uu_inv.matmul(Q_u)
            K[i] = -Q_uu_inv.matmul(Q_uz)

        V_z = Q_z + K[i].t().matmul(Q_u)
        V_z += K[i].t().matmul(Q_uu).matmul(k[i])
        V_z += Q_uz.t().matmul(k[i])

        V_zz = Q_zz + K[i].t().matmul(Q_uu).matmul(K[i])
        V_zz += K[i].t().matmul(Q_uz) + Q_uz.t().matmul(K[i])
        V_zz = 0.5 * (V_zz + V_zz.t())  # To maintain symmetry.

    return k, K


def _sample(env, Us, quiet=False):
    n_sample_trajectories, N, _ = Us.shape
    N_ = n_sample_trajectories * N

    X_ = torch.empty(N_, env.state_size + env.action_size)
    dX = torch.empty(N_, env.state_size)

    pbar = trange(Us.shape[0], desc="TRIALS", disable=quiet)
    for trajectory in pbar:
        U = Us[trajectory]
        base_i = N * trajectory
        for i, u in enumerate(U):
            u = u.detach()

            x = env.get_state().mean().detach()
            env.apply(u)
            x_next = env.get_state().mean().detach()

            j = base_i + i
            X_[j] = torch.cat([x, u], dim=-1)
            dX[j] = x_next - x

        env.reset()

    return X_, dX


def _control_law(model,
                 Z,
                 U,
                 k,
                 K,
                 alpha,
                 encoding=StateEncoding.DEFAULT,
                 model_opts={}):
    Z_new = torch.empty_like(Z)
    U_new = torch.empty_like(U)
    Z_new[0] = Z[0]

    for i in range(U.shape[0]):
        z = Z[i].detach()
        u = U[i].detach()
        z_new = Z_new[i].detach()

        dz = z_new - z
        du = alpha * (k[i] + K[i].matmul(dz))

        u_new = u + du
        z_new_next = model(z_new, u_new, i, encoding, **model_opts)

        Z_new[i + 1] = z_new_next.detach()
        U_new[i] = u_new.detach()

    return Z_new, U_new


def _control_law_fast(Z, U, F_z, F_u, k, K, alpha):
    Z_new = torch.empty_like(Z)
    U_new = torch.empty_like(U)
    Z_new[0] = Z[0]

    for i in range(U.shape[0]):
        z = Z[i].detach()
        u = U[i].detach()
        z_new = Z_new[i].detach()

        dz = z_new - z
        du = alpha * (k[i] + K[i].matmul(dz))
        dz_ = F_z[i].matmul(dz) + F_u[i].matmul(du)

        Z_new[i + 1] = (Z[i + 1] + dz_).detach()
        U_new[i] = (u + du).detach()

    return Z_new, U_new


def _trajectory_cost(cost, Z, U, encoding=StateEncoding.DEFAULT, cost_opts={}):
    N = U.shape[0]
    I = torch.arange(N)
    L = cost(Z[:-1], U, I, terminal=False, encoding=encoding, **cost_opts)
    l_f = cost(Z[-1], None, N, terminal=True, encoding=encoding, **cost_opts)
    return L.sum() + l_f
