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

from torch.utils.data import TensorDataset

from .ilqr import (iLQRController, forward, backward, Q, _linear_control_law,
                   _control_law, _trajectory_cost)

from ..utils.encoding import StateEncoding, decode_var
from ..utils.evaluation import eval_cost, eval_dynamics


class PDDPController(iLQRController):

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
        super(PDDPController, self).__init__(env, model, cost, model_opts,
                                             cost_opts)
        self._training_opts = training_opts
        self._is_training = False

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
            quiet=False,
            on_iteration=None,
            linearize_dynamics=False,
            max_var=0.05,
            n_sample_trajectories=4,
            train_on_start=True,
            **kwargs):
        """Determines the optimal path to minimize the cost.

        Args:
            U (Tensor<N, action_size>): Initial action path.
            encoding (int): StateEncoding enum.
            n_iterations (int): Maximum number of iterations until convergence.
            tol (Tensor<0>): Tolerance for early convergence.
            max_reg (Tensor<0>): Maximum regularization term.
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
            linearize_dynamics (bool): Whether to linearize the dynamics when
                computing the control law.
            max_var (Tensor<0>): Maximum variance allowed before resampling
                trajectories.
            n_sample_trajectories (int): Number of trajectories to sample from
                the environment at a time.
            train_on_start (bool): Whether to sample trajectories and train the
                model at the start or not.

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
                dataset = _sample(self._env, Us, quiet)
                self._model.fit(dataset, quiet=quiet, **self._training_opts)

            # Get initial state distribution.
            z0 = self._env.get_state().encode(encoding).detach()
            encoded_state_size = z0.shape[-1]

            changed = True
            converged = False
            train_on_start = True

            with trange(n_iterations, desc="PDDP", disable=quiet) as pbar:
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

                    # Backtracking line search.
                    for alpha in alphas:
                        if linearize_dynamics:
                            Z_new, U_new = _linear_control_law(
                                Z, U, F_z, F_u, k, K, alpha)
                        else:
                            Z_new, U_new = _control_law(self._model, Z, U, k, K,
                                                        alpha, encoding,
                                                        self._model_opts)

                        J_new = _trajectory_cost(self._cost, Z_new, U_new,
                                                 encoding, self._cost_opts)

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
                var = decode_var(Z, encoding=encoding).max()
                if var < max_var:
                    break

        return Z, U


@torch.no_grad()
def _sample(env, Us, quiet=False):
    n_sample_trajectories, N, _ = Us.shape
    N_ = n_sample_trajectories * N

    tensor_opts = {"dtype": Us.dtype, "device": Us.device}
    X_ = torch.empty(N_, env.state_size + env.action_size, **tensor_opts)
    dX = torch.empty(N_, env.state_size, **tensor_opts)

    with trange(Us.shape[0], desc="TRIALS", disable=quiet) as pbar:
        for trajectory in pbar:
            U = Us[trajectory]
            base_i = N * trajectory
            for i, u in enumerate(U):
                u = u.detach()

                x = env.get_state().mean()
                env.apply(u)
                x_next = env.get_state().mean()

                j = base_i + i
                X_[j] = torch.cat([x, u], dim=-1)
                dX[j] = x_next - x

            env.reset()

    return TensorDataset(X_, dX)
