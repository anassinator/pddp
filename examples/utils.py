"""Experiment utilities."""
import torch
import warnings
import matplotlib
import matplotlib.pyplot as plt

# Ignore matplotlib deprecation warnings.
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


def plot_pause(interval):
    # From https://stackoverflow.com/questions/45729092/
    # Used to avoid focusing the plots.
    backend = plt.rcParams["backend"]
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


def rollout(model, z0, U, encoding, **kwargs):
    Z = torch.empty(U.shape[0] + 1, z0.shape[-1])
    Z[0] = z0.detach()
    for i in range(U.shape[0]):
        Z[i + 1] = model(Z[i], U[i], i, encoding=encoding, **kwargs).detach()
    return Z.detach()
