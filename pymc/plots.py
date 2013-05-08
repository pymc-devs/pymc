import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde

__all__ = ['traceplot', 'kdeplot', 'kde2plot']


def traceplot(trace, vars=None):
    if vars is None:
        vars = trace.samples.keys()

    n = len(vars)
    f, ax = plt.subplots(n, 2, squeeze=False)

    for i, v in enumerate(vars):
        d = np.squeeze(trace[v])

        kdeplot_op(ax[i, 0], d)
        ax[i, 0].set_title(str(v))
        ax[i, 1].plot(d, alpha=.35)

        ax[i, 0].set_ylabel("frequency")
        ax[i, 1].set_ylabel("sample value")

    return f


def kdeplot_op(ax, data):
    data = np.atleast_2d(data.T).T
    for i in range(data.shape[1]):
        d = data[:, i]
        density = kde.gaussian_kde(d)
        l = np.min(d)
        u = np.max(d)
        x = np.linspace(0, 1, 100) * (u - l) + l

        ax.plot(x, density(x))


def kde2plot_op(ax, x, y, grid=200):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    grid = grid * 1j
    X, Y = np.mgrid[xmin:xmax:grid, ymin:ymax:grid]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = kde.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax])


def kdeplot(data):
    f, ax = plt.subplots(1, 1, squeeze=True)
    kdeplot_op(ax, data)
    return f


def kde2plot(x, y, grid=200):
    f, ax = plt.subplots(1, 1, squeeze=True)
    kde2plot_op(ax, x, y, grid)
    return f
