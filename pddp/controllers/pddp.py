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

from .base import Controller
from .ilqr import iLQRController, iLQRState, _trajectory_cost

from ..utils.encoding import StateEncoding, decode_mean, decode_var


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
            quiet=False,
            on_trial=None,
            max_trials=None,
            n_initial_sample_trajectories=2,
            sampling_noise=1.0,
            train_on_start=True,
            max_dataset_size=1000,
            resample_model=True,
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
            on_trial (callable): Function with the following signature:
                Args:
                    trial (int): Trial index.
                    X (Tensor<N+1, state_size>): Trial's state path.
                    U (Tensor<N, action_size>): Trial's action path.
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
                state (iLQRState): Final optimization state.
        """
        U = U.detach()
        total_trials = 0
        state = iLQRState.UNDEFINED

        # Build initial dataset.
        dataset = None
        if train_on_start:
            with trange(
                    n_initial_sample_trajectories, desc="TRIALS",
                    disable=quiet) as pbar:
                for i in pbar:
                    self.env.reset()
                    if i == 0:
                        Ui = U
                    else:
                        Ui = sampling_noise * torch.rand_like(U)
                        if u_min is not None and u_max is not None:
                            Ui = (u_max - u_min) * Ui + u_min

                    new_data, Ji = _apply_controller(
                        self.env,
                        self.cost,
                        Ui,
                        U.shape[0],
                        encoding,
                        False,
                        quiet,
                        self._cost_opts,
                        u_min=u_min,
                        u_max=u_max)
                    dataset = _concat_datasets(dataset, new_data,
                                               max_dataset_size)

                    if callable(on_trial):
                        on_trial(total_trials, new_data[0], new_data[1])
                    total_trials += 1

            # train initial model
            self.model.train()
            self.model.fit(*dataset, quiet=quiet, **self._training_opts)

        while True:
            # reset state for iLQR
            self.env.reset()

            # set model in evaluation mode
            self.model.eval()

            if resample_model and hasattr(self.model, "resample"):
                # Use different random numbers each episode.
                self.model.resample()

            # open loop control optimization
            Z, U, state = super(PDDPController, self).fit(
                U,
                encoding=encoding,
                quiet=quiet,
                u_min=u_min,
                u_max=u_max,
                **kwargs)

            if not self.training:
                break

            # apply mpc controller
            H = 2 * U.shape[0]
            new_data, J = _apply_controller(
                self.env,
                self.cost,
                self,
                H,
                encoding,
                True,
                quiet,
                self._cost_opts,
                u_min=u_min,
                u_max=u_max,
                **kwargs)
            if callable(on_trial):
                on_trial(total_trials, new_data[0], new_data[1])

            # Re-train model
            dataset = _concat_datasets(dataset, new_data, max_dataset_size)
            self.model.train()
            self.model.fit(*dataset, quiet=quiet, **self._training_opts)

            # Check if max number of trials was reached.
            total_trials += 1
            if max_trials is not None and total_trials >= max_trials:
                break

        return Z, U, state


def _apply_controller(env,
                      cost,
                      controller,
                      H,
                      encoding,
                      mpc=False,
                      quiet=False,
                      cost_opts={},
                      **kwargs):
    Z = []
    U = []

    if isinstance(controller, torch.Tensor):
        # if we got open loop controls, create a function
        # that can be called similar to the feedback controller
        open_loop_U = controller
        controller = lambda z, i, *args, **kwargs: open_loop_U[i]
    with trange(H, desc="MPC", disable=quiet) as pbar:
        for i in pbar:
            z = env.get_state().encode(encoding)
            Z.append(z)
            u = controller(z, i, encoding, mpc, **kwargs)
            U.append(u)
            env.apply(u)

    # append last state (after applying last action)
    z = env.get_state().encode(encoding)
    Z.append(z)
    # stack trajectory data into a tensor
    Z = torch.stack(Z)
    U = torch.stack(U)

    J = _trajectory_cost(cost, Z, U, encoding, cost_opts)
    X = decode_mean(Z, encoding=encoding)
    dX = X[1:] - X[:-1]
    X = X[:-1]
    return (X.detach(), U.detach(), dX.detach()), J.detach()


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
