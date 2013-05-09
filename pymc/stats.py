"""Utility functions for PyMC"""

import numpy as np

__all__ = ['hpd', 'quantiles', 'batchsd']

def statfunc(f):
    """
    Decorator for statistical utility function to automatically
    extract the trace array from whatever object is passed.
    """

    def wrapped_f(pymc_obj, *args, **kwargs):

        try:
            # MultiTrace
            traces = pymc_obj.traces

            try:
                vars = kwargs.pop('vars')
            except KeyError:
                vars = traces[0].varnames

            return [{v:f(trace[v], *args, **kwargs) for v in vars} for trace in traces]

        except AttributeError:
            pass

        try:
            # NpTrace
            try:
                vars = kwargs.pop('vars')
            except KeyError:
                vars = pymc_obj.varnames

            return {v:f(pymc_obj[v], *args, **kwargs) for v in vars}
        except AttributeError:
            pass

        # If others fail, assume that raw data is passed
        return f(pymc_obj, *args, **kwargs)

    wrapped_f.__doc__ = f.__doc__
    wrapped_f.__name__ = f.__name__

    return wrapped_f

@statfunc
def autocorr(x, lag=1):
    """Sample autocorrelation at specified lag.
    The autocorrelation is the correlation of x_i with x_{i+lag}.
    """

    S = autocov(x, lag)
    return S[0,1]/np.sqrt(np.prod(np.diag(S)))

@statfunc
def autocov(x, lag=1):
    """
    Sample autocovariance at specified lag.
    The autocovariance is a 2x2 matrix with the variances of
    x[:-lag] and x[lag:] in the diagonal and the autocovariance
    on the off-diagonal.
    """

    if not lag: return 1
    if lag<0:
        raise ValueError("Autocovariance lag must be a positive integer")
    return np.cov(x[:-lag], x[lag:], bias=1)

def make_indices(dimensions):
    # Generates complete set of indices for given dimensions

    level = len(dimensions)

    if level==1: return range(dimensions[0])

    indices = [[]]

    while level:

        _indices = []

        for j in range(dimensions[level-1]):

            _indices += [[j]+i for i in indices]

        indices = _indices

        level -= 1

    try:
        return [tuple(i) for i in indices]
    except TypeError:
        return indices

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of
    a given width

    Assumes that x is sorted numpy array.
    """

    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor( cred_mass*n ))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width)==0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

@statfunc
def hpd(x, alpha=0.05):
    """Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      alpha : float
          Desired probability of type I error (defaults to 0.05)

    """

    # Make a copy of trace
    x = x.copy()

    # For multivariate node
    if x.ndim>1:

        # Transpose first, then sort
        tx = np.transpose(x, range(x.ndim)[1:]+[0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1]+(2,))

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = np.sort(tx[index])

            # Append to list
            intervals[index] = calc_min_interval(sx, alpha)

        # Transpose back before returning
        return np.array(intervals)

    else:
        # Sort univariate node
        sx = np.sort(x)

        return np.array(calc_min_interval(sx, alpha))

@statfunc
def batchsd(x, batches=5):
    """
    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      batches : integer
          Number of batchas
    """

    if x.ndim > 1:

        dims = np.shape(x)
        #ttrace = np.transpose(np.reshape(trace, (dims[0], sum(dims[1:]))))
        trace = np.transpose([t.ravel() for t in x])

        return np.reshape([batchsd(t, batches) for t in trace], dims[1:])

    else:
        if batches == 1: return np.std(x)/np.sqrt(len(x))

        try:
            batched_traces = np.resize(x, (batches, len(x)/batches))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(x) % batches
            batched_traces = np.resize(x[:-resid], (batches, len(x)/batches))

        means = np.mean(batched_traces, 1)

        return np.std(means)/np.sqrt(batches)

@statfunc
def quantiles(x, qlist=(2.5, 25, 50, 75, 97.5)):
    """Returns a dictionary of requested quantiles from array

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      qlist : tuple or list
          A list of desired quantiles (defaults to (2.5, 25, 50, 75, 97.5))

    """

    # Make a copy of trace
    x = x.copy()

    # For multivariate node
    if x.ndim>1:
        # Transpose first, then sort, then transpose back
        sx = np.sort(x.T).T
    else:
        # Sort univariate node
        sx = np.sort(x)

    try:
        # Generate specified quantiles
        quants = [sx[int(len(sx)*q/100.0)] for q in qlist]

        return dict(zip(qlist, quants))

    except IndexError:
        print("Too few elements for quantile calculation")