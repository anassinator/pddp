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
import numpy as np

with warnings.catch_warnings():
    # Ignore potential warning when in Jupyter environment.
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import trange

from torch.utils.data import TensorDataset

from .ilqr import (iLQRController, forward, backward, Q, _linear_control_law,
                   _control_law, _trajectory_cost)

from ..utils.evaluation import eval_cost, eval_dynamics
from ..utils.encoding import StateEncoding, decode_var, infer_encoded_state_size


class PDDPController(iLQRController):

    """PDDP controller."""

    def __init__(self,
                 env,
                 model,
                 cost,
                 model_opts={},
                 cost_opts={},
                 training_opts={},
                 **kwargs):
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
        super(PDDPController, self).__init__(env, model, cost, model_opts,
                                             cost_opts)
        self._training_opts = training_opts

    def fit(self,
            U,
            encoding=StateEncoding.DEFAULT,
            n_iterations=50,
            tol=5e-6,
            max_reg=1e10,
            batch_rollout=True,
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
            apply_feedback_control=False,
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
        encoded_state_size = infer_encoded_state_size(self.model.state_size,
                                                      encoding)
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
        alphas = 10.0**torch.linspace(0, -3, 11).to(**tensor_opts)

        while True:
            if resample_model and hasattr(self.model, "resample"):
                # Use different random numbers each episode.
                self.model.resample()

            # Reset regularization term.
            self._reset_reg()

            # Get initial state distribution.
            z0 = self.env.get_state().encode(encoding).detach().to(
                **tensor_opts)

            changed = True
            converged = False

            with trange(n_iterations, desc="PDDP", disable=quiet) as pbar:
                for i in pbar:
                    accepted = False

                    # Forward rollout.
                    if changed:
                        Z, F_z, F_u, L, L_z, L_u, L_zz, L_uz, L_uu = forward(
                            z0, U, self.model, self.cost, encoding,
                            batch_rollout, self._model_opts, self._cost_opts)
                        J_opt = L.sum()
                        changed = False

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
                            reg=self._mu)
                    except RuntimeError:
                        if self._increase_reg(max_reg):
                            continue
                        break

                    # Batch-backtracking line search.
                    Z_new_b, U_new_b = _control_law(self.model, Z, U, k, K,
                                                    alphas, encoding,
                                                    self._model_opts)
                    J_new_b = _trajectory_cost(self.cost, Z_new_b, U_new_b,
                                               encoding, self._cost_opts)
                    amin = J_new_b.argmin()
                    alpha = alphas[amin]
                    J_new = J_new_b[amin].detach()
                    Z_new = Z_new_b[:, amin].detach()
                    U_new = U_new_b[:, amin].detach()

                    if J_new < J_opt:
                        # Check if converged due to small change.
                        if (J_opt - J_new).abs() / J_opt < tol:
                            converged = True
                        J_opt = J_new
                        self._Z_nominal = Z = Z_new
                        self._U_nominal = U = U_new
                        self._K = K

                        changed = True
                        accepted = True

                        self._decrease_reg()

                    info = {
                        "loss": J_opt.detach_().cpu().numpy(),
                        "reg": self._mu,
                        "alpha": alpha.cpu().numpy(),
                        "accepted": accepted,
                    }
                    if self.training:
                        var = decode_var(Z, encoding=encoding).max()
                        info["var"] = var.detach().numpy()
                    pbar.set_postfix(info)

                    if on_iteration:
                        on_iteration(i, Z.detach(), U.detach(), J_opt.detach(),
                                     accepted, converged)

                    if not accepted and not self._increase_reg(max_reg):
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
            if apply_feedback_control:
                controls = U if K is None else (U, (K, Z))
            else:
                controls = U
            dataset, U, J = _train(
                self.env, self.model, self.cost, controls,
                n_sample_trajectories, dataset, concatenate_datasets,
                max_dataset_size, quiet, on_trial, total_trials, sampling_noise,
                self._training_opts, self._cost_opts, encoding)
            total_trials += n_sample_trajectories

            if J < bestJ:
                bestU = U
                bestJ = J

            if start_from_bestU:
                U = bestU

        return Z, U


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
           cost_opts={},
           encoding=StateEncoding.DEFAULT):
    model.train()
    if type(U) is tuple or type(U) is list:
        U, feedback_control = U
    else:
        feedback_control = None

    # Sample new trajectories.
    Us = U.repeat(n_trajectories, 1, 1)
    Us[1:] += noise * torch.randn_like(Us[1:])
    dataset_, sample_losses = _sample(env, Us, cost, quiet, on_trial, n_trials,
                                      cost_opts, feedback_control, encoding)

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
def _sample(env,
            Us,
            cost,
            quiet=False,
            on_trial=None,
            n_trials=0,
            cost_opts={},
            feedback_control=None,
            encoding=StateEncoding.DEFAULT):
    n_sample_trajectories, N, _ = Us.shape
    N_ = n_sample_trajectories * N

    tensor_opts = {"dtype": Us.dtype, "device": Us.device}
    X = torch.empty(N_, env.state_size, **tensor_opts)
    U = torch.empty(N_, env.action_size, **tensor_opts)
    dX = torch.empty(N_, env.state_size, **tensor_opts)
    L = torch.zeros(n_sample_trajectories, **tensor_opts)
    if feedback_control is not None:
        K, Z = feedback_control
    else:
        K, Z = None, None

    with trange(Us.shape[0], desc="TRIALS", disable=quiet) as pbar:
        for trajectory in pbar:
            current_U = Us[trajectory]
            base_i = N * trajectory
            for i, u in enumerate(current_U):
                u = u.detach()
                x = env.get_state().mean().to(**tensor_opts)
                if K is not None and Z is not None:
                    z = env.get_state().encode(encoding).to(**tensor_opts)
                    dz = z - Z[i]
                    u = u + K[i].matmul(dz)
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
