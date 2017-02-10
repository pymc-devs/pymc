"""Convergence diagnostics and model validation"""

import numpy as np
from .stats import statfunc

__all__ = ['geweke', 'gelman_rubin', 'effective_n']


@statfunc
def geweke(x, first=.1, last=.5, intervals=20):
    R"""Return z-scores for convergence diagnostics.

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

    if np.ndim(x) > 1:
        return [geweke(y, first, last, intervals) for y in np.transpose(x)]

    # Filter out invalid intervals
    for interval in (first, last):
        if interval <= 0 or interval >= 1:
            raise ValueError(
                "Invalid intervals for Geweke convergence analysis",
                (first,
                 last))
    if first + last >= 1:
        raise ValueError(
            "Invalid intervals for Geweke convergence analysis",
            (first,
             last))

    # Initialize list of z-scores
    zscores = []

    # Last index value
    end = len(x) - 1

    # Start intervals going up to the <last>% of the chain
    last_start_idx = (1 - last) * end

    # Calculate starting indices
    start_indices = np.arange(0, int(last_start_idx), step=int(
        (last_start_idx) / (intervals - 1)))

    # Loop over start indices
    for start in start_indices:
        # Calculate slices
        first_slice = x[start: start + int(first * (end - start))]
        last_slice = x[int(end - last * (end - start)):]

        z = first_slice.mean() - last_slice.mean()
        z /= np.sqrt(first_slice.var() + last_slice.var())

        zscores.append([start, z])

    if intervals is None:
        return np.array(zscores[0])
    else:
        return np.array(zscores)


def gelman_rubin(mtrace):
    R"""Returns estimate of R for a set of traces.

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

    Returns
    -------
    Rhat : dict
      Returns dictionary of the potential scale reduction
      factors, :math:`\hat{R}`

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

    if mtrace.nchains < 2:
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains '
            'of the same length.')

    Rhat = {}
    for var in mtrace.varnames:
        x = np.array(mtrace.get_values(var, combine=False))
        num_samples = x.shape[1]

        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        Rhat[var] = np.sqrt(Vhat / W)

    return Rhat


def effective_n(mtrace):
    R"""Returns estimate of the effective sample size of a set of traces.

    Parameters
    ----------
    mtrace : MultiTrace
        A MultiTrace object containing parallel traces (minimum 2)
        of one or more stochastic parameters.

    Returns
    -------
    n_eff : float
        Return the effective sample size, :math:`\hat{n}_{eff}`

    Notes
    -----
    The diagnostic is computed by:

    .. math:: \hat{n}_{eff} = \frac{mn}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}

    where :math:`\hat{\rho}_t` is the estimated autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` is negative.

    References
    ----------
    Gelman et al. (2014)"""

    if mtrace.nchains < 2:
        raise ValueError(
            'Calculation of effective sample size requires multiple chains '
            'of the same length.')

    def get_vhat(x):
        # number of chains is last dim (-1)
        # chain samples are second to last dim (-2)
        num_samples = x.shape[-2]

        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=-2), axis=-1, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=-2, ddof=1), axis=-1)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        return Vhat

    def get_neff(x, Vhat):
        num_chains = x.shape[-1]
        num_samples = x.shape[-2]

        negative_autocorr = False
        t = 1

        rho = np.ones(num_samples)
        # Iterate until the sum of consecutive estimates of autocorrelation is
        # negative
        while not negative_autocorr and (t < num_samples):

            variogram = np.mean((x[t:, :] - x[:-t, :])**2)
            rho[t] = 1. - variogram / (2. * Vhat)

            negative_autocorr = sum(rho[t - 1:t + 1]) < 0

            t += 1

        if t % 2:
            t -= 1

        return min(num_chains * num_samples,
                   int(num_chains * num_samples / (1. + 2 * rho[1:t-1].sum())))

    n_eff = {}
    for var in mtrace.varnames:
        x = np.array(mtrace.get_values(var, combine=False))

        # make sure to handle scalars correctly - add extra dim if needed
        if len(x.shape) == 2:
            is_scalar = True
            x = np.atleast_3d(mtrace.get_values(var, combine=False))
        else:
            is_scalar = False

        # now we are going to transpose all dims - makes the loop below
        # easier by moving the axes of the variable to the front instead
        # of the chain and sample axes
        x = x.transpose()

        Vhat = get_vhat(x)

        # get an array the same shape as the var
        _n_eff = np.zeros(x.shape[:-2])

        # iterate over tuples of indices of the shape of var
        for tup in np.ndindex(*list(x.shape[:-2])):
            _n_eff[tup] = get_neff(x[tup], Vhat[tup])

        # we could be using np.squeeze here, but we don't want to squeeze
        # out dummy dimensions that a user inputs
        if is_scalar:
            n_eff[var] = _n_eff[0]
        else:
            # make sure to transpose the dims back
            n_eff[var] = np.transpose(_n_eff)

    return n_eff
