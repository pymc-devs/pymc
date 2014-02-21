"""Convergence diagnostics and model validation"""

import numpy as np
from .stats import autocorr, autocov, statfunc
from copy import copy

__all__ = ['geweke', 'gelman_rubin', 'trace_to_dataframe']


@statfunc
def geweke(x, first=.1, last=.5, intervals=20):
    """Return z-scores for convergence diagnostics.

    Compare the mean of the first % of series with the mean of the last % of
    series. x is divided into a number of segments for which this difference is
    computed. If the series is converged, this score should oscillate between
    -1 and 1.

    Parameters
    ----------
    x : array-like
      The trace of some stochastic parameter.
    first : float
      The fraction of series at the beginning of the trace.
    last : float
      The fraction of series at the end to be compared with the section
      at the beginning.
    intervals : int
      The number of segments.

    Returns
    -------
    scores : list [[]]
      Return a list of [i, score], where i is the starting index for each
      interval and score the Geweke score on the interval.

    Notes
    -----

    The Geweke score on some series x is computed by:

      .. math:: \frac{E[x_s] - E[x_e]}{\sqrt{V[x_s] + V[x_e]}}

    where :math:`E` stands for the mean, :math:`V` the variance,
    :math:`x_s` a section at the start of the series and
    :math:`x_e` a section at the end of the series.

    References
    ----------
    Geweke (1992)
    """

    if np.rank(x) > 1:
        return [geweke(y, first, last, intervals) for y in np.transpose(x)]

    # Filter out invalid intervals
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first,
             last))

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(x) - 1

    # Calculate starting indices
    sindices = np.arange(0, end // 2, step=int((end / 2) / (intervals - 1)))

    # Loop over start indices
    for start in sindices:

        # Calculate slices
        first_slice = x[start: start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]

        z = (first_slice.mean() - last_slice.mean())
        z /= np.sqrt(first_slice.std() ** 2 + last_slice.std() ** 2)

        zscores.append([start, z])

    if intervals is None:
        return np.array(zscores[0])
    else:
        return np.array(zscores)


def gelman_rubin(trace):
    """ Returns estimate of R for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    trace
      A trace object containing parallel traces (minimum 2)
      of one or more stochastic parameters.

    Returns
    -------
    Rhat : dict
      Returns dictionary of the potential scale reduction factors, :math:`\hat{R}`

    Notes
    -----

    The diagnostic is computed by:

      .. math:: \hat{R} = \frac{\hat{V}}{W}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is
    the posterior variance estimate for the pooled traces. This is the
    potential scale reduction factor, which converges to unity when each
    of the traces is a sample from the target posterior. Values greater
    than one indicate that one or more chains have not yet converged.

    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)"""

    if trace.nchains < 2:
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    def calc_rhat(x):

        try:
            # When the variable is multidimensional, this assignment will fail, triggering
            # a ValueError that will handle the multidimensional case
            m, n = x.shape

            # Calculate between-chain variance
            B = n * np.var(np.mean(x, axis=1), ddof=1)

            # Calculate within-chain variance
            W = np.mean(np.var(x, axis=1, ddof=1))

            # Estimate of marginal posterior variance
            Vhat = W*(n - 1)/n + B/n

            return np.sqrt(Vhat/W)

        except ValueError:

            # Tricky transpose here, shifting the last dimension to the first
            rotated_indices = np.roll(np.arange(x.ndim), 1)
            # Now iterate over the dimension of the variable
            return np.squeeze([calc_rhat(xi) for xi in x.transpose(rotated_indices)])

    Rhat = {}
    for var in trace.varnames:

        # Get all traces for var
        x = np.array(trace.get_values(var))

        try:
            Rhat[var] = calc_rhat(x)
        except ValueError:
            Rhat[var] = [calc_rhat(y.transpose()) for y in x.transpose()]

    return Rhat


def trace_to_dataframe(trace):
    """Convert a PyMC trace consisting of 1-D variables to a pandas DataFrame
    """
    import pandas as pd
    return pd.DataFrame(
        {varname: np.squeeze(trace.get_values(varname, combine=True))
         for varname in trace.varnames})
