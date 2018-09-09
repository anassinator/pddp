"""Experiment utilities."""
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