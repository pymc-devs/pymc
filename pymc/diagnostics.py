"""Convergence diagnostics and model validation"""

import numpy as np
from stats import autocorr, autocov, statfunc
from copy import copy

__all__ = ['geweke', 'gelman_rubin']


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
    sindices = np.arange(0, end / 2, step=int((end / 2) / (intervals - 1)))

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


def gelman_rubin(mtrace, burn=0):
    """ Returns estimate of R for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    mtrace : MultiTrace
      A MultiTrace object containing parallel traces (minimum 2)
      of one or more stochastic parameters.
    burn : int
      Burn-in interval (defaults to zero)

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

    m = len(mtrace.traces)
    if m < 2:
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    varnames = mtrace.traces[0].varnames
    n = len(mtrace.traces[0][varnames[0]]) - burn

    Rhat = {}
    for var in varnames:

        # Get all traces for var
        x = np.array([mtrace.traces[i][var][burn:] for i in range(m)])

        # Calculate between-chain variance
        B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

        # Calculate within-chain variances
        W = np.sum(
            [(x[i] - xbar) ** 2 for i,
             xbar in enumerate(np.mean(x,
                                       1))]) / (m * (n - 1))

        # (over) estimate of variance
        s2 = W * (n - 1) / n + B_over_n

        # Pooled posterior variance estimate
        V = s2 + B_over_n / m

        # Calculate PSRF
        Rhat[var] = V / W

    return Rhat
