import matplotlib.pyplot as plt

from .artists import kdeplot_op, kde2plot_op


def kdeplot(data, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True)
    kdeplot_op(ax, data)
    return ax


def kde2plot(x, y, grid=200, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True)
    kde2plot_op(ax, x, y, grid, **kwargs)
    return ax
