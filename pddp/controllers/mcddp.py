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
"""MC DDP controller."""

import torch
import warnings
import numpy as np

with warnings.catch_warnings():
    # Ignore potential warning when in Jupyter environment.
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange

from torch.utils.data import TensorDataset

from .ilqr import (iLQRController, forward, backward, Q, _linear_control_law,
                   _control_law, _trajectory_cost)

from ..utils.encoding import StateEncoding, decode_var
from ..utils.evaluation import eval_cost, eval_dynamics


class MCDDPController(iLQRController):

    """Monte-Carlo DDP controller."""

    def __init__(self,
                 env,
                 model,
                 cost,
                 model_opts={},
                 cost_opts={},
                 training_opts={}):
        """Constructs a MCDDPController.

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
        super(PDDPController, self).__init__(env, model, cost, model_opts,
                                             cost_opts)
        self._training_opts = training_opts

    def fit(self,
            U,
            n_particles=100,
            n_iterations=50,
            tol=5e-6,
            max_reg=1e10,
            batch_rollout=False,
            quiet=False,
            on_iteration=None,
            on_trial=None,
            linearize_dynamics=False,
            max_var=0.2,
            max_J=0.0,
            max_trials=None,
            n_sample_trajectories=1,
            n_initial_sample_trajectories=2,
            sampling_noise=1e-2,
            train_on_start=True,
            concatenate_datasets=True,
            max_dataset_size=1000,
            start_from_bestU=True,
            resample_model=True,
            **kwargs):
        """Determines the optimal path to minimize the cost.

        Args:
            U (Tensor<N, action_size>): Initial action path.
            n_particles (int): Number of particles.
            n_iterations (int): Maximum number of iterations until convergence.
            tol (Tensor<0>): Tolerance for early convergence.
            max_reg (Tensor<0>): Maximum regularization term.
            batch_rollout (bool): Whether to rollout in parallel or not.
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
            on_trial (callable): Function with the following signature:
                Args:
                    trial (int): Trial index.
                    X (Tensor<N+1, state_size>): Trial's state path.
                    U (Tensor<N, action_size>): Trial's action path.
            linearize_dynamics (bool): Whether to linearize the dynamics when
                computing the control law.
            max_var (Tensor<0>): Maximum variance allowed before resampling
                trajectories.
            max_J (Tensor<0>): Maximum cost to converge.
                trajectories.
            max_trials (int): Maximum number of trials to run.
            n_sample_trajectories (int): Number of trajectories to sample from
                the environment at a time.
            n_initial_sample_trajectories (int): Number of initial trajectories
                to sample from the environment at a time.
            sampling_noise (float): Sampling noise to add to trajectories.
            train_on_start (bool): Whether to sample trajectories and train the
                model at the start or not.
            concatenate_datasets (bool): Whether to train on all data or just
                the latest.
            max_dataset_size (int): Maximum dataset size, None means limitless.
            start_from_bestU (bool): Whether to use the best overall trajectory
                as a seed.
            resample_model (bool): Whether to resample the model each episode.

        Returns:
            Tuple of:
                Z (Tensor<N+1, encoded_state_size>): Optimal encoded state path.
                U (Tensor<N, action_size>): Optimal action path.
        """
        U = U.detach()
        bestU = U
        bestJ = np.inf
        N, action_size = U.shape
        tensor_opts = {"dtype": U.dtype, "device": U.device}

        # Build initial dataset.
        dataset = None
        total_trials = 0
        if self.training and train_on_start:
            dataset, U, J = _train(
                self.env, self.model, self.cost, U,
                n_initial_sample_trajectories, None, concatenate_datasets,
                max_dataset_size, quiet, on_trial, total_trials, sampling_noise,
                self._training_opts, self._cost_opts)
            total_trials += n_initial_sample_trajectories
            bestU = U
            bestJ = J

        # Backtracking line search candidates 0 < alpha <= 1.
        alphas = 1.1**(-torch.arange(10.0)**2).to(**tensor_opts)

        while True:
            if resample_model and hasattr(self.model, "resample"):
                # Use different random numbers each episode.
                self.model.resample()

            # Reset regularization term.
            self._mu = 1.0
            self._delta = self._delta_0

            # Get initial particles.
            x0 = self.env.get_state().sample(n_particles).detach().to(
                **tensor_opts)

            changed = True
            converged = False

            with trange(n_iterations, desc="PDDP", disable=quiet) as pbar:
                for i in pbar:
                    accepted = False

                    # Forward rollout.
                    if changed:
                        X, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu = forward(
                            x0, U, self.model, self.cost, batch_rollout,
                            self._model_opts, self._cost_opts)
                        J_opt = L.sum()
                        changed = False

                    # Backward pass.
                    k, K = backward(
                        X,
                        F_x,
                        F_u,
                        L,
                        L_x,
                        L_u,
                        L_zz,
                        L_uz,
                        L_uu,
                        reg=self._mu)

                    # Backtracking line search.
                    for alpha in alphas:
                        if linearize_dynamics:
                            X_new, U_new = _linear_control_law(
                                X, U, F_x, F_u, k, K, alpha)
                        else:
                            X_new, U_new = _control_law(self.model, X, U, k, K,
                                                        alpha, self._model_opts)

                        J_new = _trajectory_cost(self.cost, X_new, U_new,
                                                 self._cost_opts)

                        if J_new < J_opt:
                            # Check if converged due to small change.
                            if (J_opt - J_new).abs() / J_opt <= tol:
                                converged = True
                            elif J_opt <= max_J:
                                converged = True

                            J_opt = J_new
                            X = X_new
                            U = U_new
                            changed = True

                            # Decrease regularization term.
                            self._delta = min(1.0, self._delta) / self._delta_0
                            self._mu *= self._delta
                            if self._mu <= self._mu_min:
                                self._mu = 0.0

                            accepted = True
                            break

                    info = {
                        "loss": J_opt.detach_().cpu().numpy(),
                        "reg": self._mu,
                        "accepted": accepted,
                    }
                    if self.training:
                        var = decode_var(X, encoding=encoding).max()
                        info["var"] = var.detach().numpy()
                    pbar.set_postfix(info)

                    if on_iteration:
                        on_iteration(i, Z.detach(), U.detach(), J_opt.detach(),
                                     accepted, converged)

                    if not accepted:
                        # Increase regularization term.
                        self._delta = max(1.0, self._delta) * self._delta_0
                        self._mu = max(self._mu_min, self._mu * self._delta)
                        if self._mu >= max_reg:
                            warnings.warn("exceeded max regularization term")
                            break

                    if converged:
                        break

            if not self.training:
                break

            if converged:
                # Check if uncertainty is satisfactory.
                var = decode_var(Z, encoding=encoding).max()
                if var < max_var:
                    break

            # Check if max number of trials was reached.
            if max_trials is not None and total_trials >= max_trials:
                break

            # Sample new data.
            dataset, U, J = _train(
                self.env, self.model, self.cost, U, n_sample_trajectories,
                dataset, concatenate_datasets, max_dataset_size, quiet,
                on_trial, total_trials, sampling_noise, self._training_opts,
                self._cost_opts)
            total_trials += n_sample_trajectories

            if J < bestJ:
                bestU = U
                bestJ = J

            if start_from_bestU:
                U = bestU

        return Z, U


def forward(x0,
            U,
            model,
            cost,
            batch_rollout=False,
            model_opts={},
            cost_opts={}):
    """Evaluates the forward rollout.

    Args:
        x0 (Tensor<n_particles, state_size>): Initial state distribution.
        U (Tensor<N, action_size>): Initial action path.
        model (DynamicsModel): Dynamics model.
        cost (Cost): Cost function.
        batch_rollout (bool): Whether to rollout in parallel or not.
        model_opts (dict): Additional key-word arguments to pass to `model()`.
        cost_opts (dict): Additional key-word arguments to pass to `cost()`.

    Returns:
        Tuple of
            X (Tensor<N+1, n_particles, state_size>): State particle path.
            F_x (Tensor<N, state_size, state_size>): Gradient of state path
                w.r.t. previous state.
            F_u (Tensor<N, state_size, action_size>): Gradient of state path
                w.r.t. previous action.
            L (Tensor<N+1>): Cost path.
            L_x (Tensor<N+1, state_size>): Gradient of cost path w.r.t. state
                path.
            L_u (Tensor<N, action_size>): Gradient of cost path w.r.t. action
                path.
            L_xx (Tensor<N+1, state_size, state_size>): Hessian of cost path
                w.r.t. state.
            L_ux (Tensor<N+1, action_size, state_size>): Hessian of cost path
                w.r.t. action and state.
            L_uu (Tensor<N+1, action_size, action_size>): Hessian of cost path
                w.r.t. action.
    """
    U.detach_()
    cost.eval()
    model.eval()

    eval_cost_fn = batch_eval_cost if batch_rollout else eval_cost
    eval_dynamics_fn = batch_eval_dynamics if batch_rollout else eval_dynamics

    N, action_size = U.shape
    n_particles, state_size = x0.shape
    tensor_opts = {"dtype": x0.dtype, "device": x0.device}

    X = torch.empty(N + 1, n_particles, state_size, **tensor_opts)
    F_x = torch.empty(N, state_size, state_size, **tensor_opts)
    F_u = torch.empty(N, state_size, action_size, **tensor_opts)

    L = torch.empty(N + 1, **tensor_opts)
    L_x = torch.empty(N + 1, state_size, **tensor_opts)
    L_u = torch.empty(N, action_size, **tensor_opts)
    L_xx = torch.empty(N + 1, state_size, state_size, **tensor_opts)
    L_ux = torch.empty(N, action_size, state_size, **tensor_opts)
    L_uu = torch.empty(N, action_size, action_size, **tensor_opts)

    X[0] = x0
    for i in range(N):
        x = X[i].detach().requires_grad_()
        u = U[i].detach().requires_grad_()

        L[i], L_x[i], L_u[i], L_xx[i], L_ux[i], L_uu[i] = eval_cost_fn(
            cost,
            x,
            u,
            i,
            encoding=StateEncoding.IGNORE_UNCERTAINTY,
            **cost_opts)

        X[i + 1], F_x[i], F_u[i] = eval_dynamics_fn(
            model,
            x,
            u,
            i,
            encoding=StateEncoding.IGNORE_UNCERTAINTY,
            **model_opts)

    # Terminal cost.
    x = X[-1].detach().requires_grad_()
    L[-1], L_x[-1], _, L_xx[-1], _, _ = eval_cost_fn(
        cost,
        x,
        None,
        i,
        terminal=True,
        encoding=StateEncoding.IGNORE_UNCERTAINTY,
        **cost_opts)

    X.detach_()
    F_x.detach_()
    F_u.detach_()

    L.detach_()
    L_x.detach_()
    L_u.detach_()
    L_xx.detach_()
    L_ux.detach_()
    L_uu.detach_()

    return X, F_x, F_u, L, L_x, L_u, L_xx, L_ux, L_uu


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
    Q_uz = L_uz + F_u.t().mm(V_zz).mm(F_z)
    Q_uu = L_uu + F_u.t().mm(V_zz).mm(F_u)
    return Q_z, Q_u, Q_zz, Q_uz, Q_uu


@torch.no_grad()
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

    N, action_size = L_u.shape
    encoded_state_size = Z.shape[1]

    tensor_opts = {"dtype": Z.dtype, "device": Z.device}
    k = torch.empty(N, action_size, **tensor_opts)
    K = torch.empty(N, action_size, encoded_state_size, **tensor_opts)

    reg = reg * torch.eye(V_zz.shape[0], **tensor_opts)

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


@torch.no_grad()
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
    model.eval()

    for i in range(U.shape[0]):
        z = Z[i]
        u = U[i]
        z_new = Z_new[i]

        dz = z_new - z
        du = alpha * (k[i] + K[i].matmul(dz))

        u_new = u + du
        z_new_next = model(z_new, u_new, i, encoding, **model_opts)

        Z_new[i + 1] = z_new_next
        U_new[i] = u_new

    return Z_new, U_new


@torch.no_grad()
def _linear_control_law(Z, U, F_z, F_u, k, K, alpha):
    Z_new = torch.empty_like(Z)
    U_new = torch.empty_like(U)
    Z_new[0] = Z[0]

    for i in range(U.shape[0]):
        z = Z[i]
        u = U[i]
        z_new = Z_new[i]

        dz = z_new - z
        du = alpha * (k[i] + K[i].matmul(dz))
        dz_ = F_z[i].matmul(dz) + F_u[i].matmul(du)

        Z_new[i + 1] = Z[i + 1] + dz_
        U_new[i] = u + du

    return Z_new, U_new


@torch.no_grad()
def _trajectory_cost(cost, Z, U, encoding=StateEncoding.DEFAULT, cost_opts={}):
    cost.eval()
    N = U.shape[0]
    I = torch.arange(N)
    L = cost(Z[:-1], U, I, terminal=False, encoding=encoding, **cost_opts)
    l_f = cost(Z[-1], None, N, terminal=True, encoding=encoding, **cost_opts)
    return L.sum() + l_f


@torch.no_grad()
def _train(env,
           model,
           cost,
           U,
           n_trajectories,
           dataset=None,
           concatenate_datasets=False,
           max_dataset_size=None,
           quiet=False,
           on_trial=None,
           n_trials=0,
           noise=1e-2,
           training_opts={},
           cost_opts={}):
    model.train()

    # Sample new trajectories.
    Us = U.repeat(n_trajectories, 1, 1)
    Us[1:] += noise * torch.randn_like(Us[1:])
    dataset_, sample_losses = _sample(env, Us, cost, quiet, on_trial, n_trials,
                                      cost_opts)

    # Update dataset.
    if concatenate_datasets:
        dataset = _concat_datasets(dataset, dataset_, max_dataset_size)
    else:
        dataset = dataset_

    # Train the model.
    model.fit(*dataset, quiet=quiet, **training_opts)

    # Pick best trajectory to continue.
    U = Us[sample_losses.argmax()].detach()
    J = sample_losses.max()
    return dataset, U, J


@torch.no_grad()
def _sample(env, Us, cost, quiet=False, on_trial=None, n_trials=0,
            cost_opts={}):
    n_sample_trajectories, N, _ = Us.shape
    N_ = n_sample_trajectories * N

    tensor_opts = {"dtype": Us.dtype, "device": Us.device}
    X = torch.empty(N_, env.state_size, **tensor_opts)
    U = torch.empty(N_, env.action_size, **tensor_opts)
    dX = torch.empty(N_, env.state_size, **tensor_opts)
    L = torch.zeros(n_sample_trajectories, **tensor_opts)

    with trange(Us.shape[0], desc="TRIALS", disable=quiet) as pbar:
        for trajectory in pbar:
            current_U = Us[trajectory]
            base_i = N * trajectory
            for i, u in enumerate(current_U):
                u = u.detach()

                x = env.get_state().mean().to(**tensor_opts)
                env.apply(u)
                x_next = env.get_state().mean().to(**tensor_opts)

                j = base_i + i
                X[j] = x
                U[j] = u
                dX[j] = x_next - x

                L[trajectory] += cost(
                    x,
                    u,
                    i,
                    encoding=StateEncoding.IGNORE_UNCERTAINTY,
                    **cost_opts).detach()

            L[trajectory] += cost(
                x_next,
                None,
                i,
                terminal=True,
                encoding=StateEncoding.IGNORE_UNCERTAINTY,
                **cost_opts).detach()

            if on_trial:
                on_trial(n_trials + trajectory, X[base_i:base_i + N].detach(),
                         U[base_i:base_i + N].detach())

            env.reset()

    X.detach_()
    U.detach_()
    dX.detach_()
    L.detach_()

    return (X, U, dX), L


@torch.no_grad()
def _concat_datasets(first, second, max_dataset_size=None):
    if first is None:
        return second
    elif second is None:
        return first

    X, U, dX = first
    X_, U_, dX_ = second

    X = torch.cat([X, X_])
    U = torch.cat([U, U_])
    dX = torch.cat([dX, dX_])

    if max_dataset_size is not None:
        X = X[-max_dataset_size:]
        U = U[-max_dataset_size:]
        dX = dX[-max_dataset_size:]

    return X, U, dX
