import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import gaussian, convolve


def identity_transform(x):
    """f(x) = x"""
    return x


def get_default_varnames(trace, include_transformed):
    """Helper to extract default varnames from a trace."""
    if include_transformed:
        return [name for name in trace.varnames]
    else:
        return [name for name in trace.varnames if not name.endswith('_')]


def get_axis(ax, default_rows, default_columns, **default_kwargs):
    """Verifies the provided axis is of the correct shape, and creates one if needed.

    Args:
        ax: matplotlib axis or None
        default_rows: int, expected rows in axis
        default_columns: int, expected columns in axis
        **default_kwargs: keyword arguments to pass to plt.subplot

    Returns:
        axis, or raises an error
    """

    default_shape = (default_rows, default_columns)
    if ax is None:
        _, ax = plt.subplots(*default_shape, **default_kwargs)
    elif ax.shape != default_shape:
        raise ValueError('Subplots with shape %r required' % (default_shape,))
    return ax


def make_2d(a):
    """Ravel the dimensions after the first."""
    a = np.atleast_2d(a.T).T
    # flatten out dimensions beyond the first
    n = a.shape[0]
    newshape = np.product(a.shape[1:]).astype(int)
    a = a.reshape((n, newshape), order='F')
    return a


def fast_kde(x):
    """
    A fft-based Gaussian kernel density estimate (KDE) for computing
    the KDE on a regular grid.
    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------

    x : Numpy array or list

    Returns
    -------

    grid: A gridded 1D KDE of the input points (x).
    xmin: minimum value of x
    xmax: maximum value of x

    """
    x = x[~np.isnan(x)]
    x = x[~np.isinf(x)]
    n = len(x)
    nx = 200

    # add small jitter in case input values are the same
    x += np.random.uniform(-1E-12, 1E-12, size=n)
    xmin, xmax = np.min(x), np.max(x)

    # compute histogram
    bins = np.linspace(xmin, xmax, nx)
    xyi = np.digitize(x, bins)
    dx = (xmax - xmin) / (nx - 1)
    grid = np.histogram(x, bins=nx)[0]

    # Scaling factor for bandwidth
    scotts_factor = n ** (-0.2)
    # Determine the bandwidth using Scott's rule
    std_x = np.std(xyi)
    kern_nx = int(np.round(scotts_factor * 2 * np.pi * std_x))

    # Evaluate the gaussian function on the kernel grid
    kernel = np.reshape(gaussian(kern_nx, scotts_factor * std_x), kern_nx)

    # Compute the KDE
    # use symmetric padding to correct for data boundaries in the kde
    npad = np.min((nx, 2 * kern_nx))

    grid = np.concatenate([grid[npad: 0: -1], grid, grid[nx: nx - npad: -1]])
    grid = convolve(grid, kernel, mode='same')[npad: npad + nx]

    norm_factor = n * dx * (2 * np.pi * std_x ** 2 * scotts_factor ** 2) ** 0.5

    grid = grid / norm_factor

    return grid, xmin, xmax
