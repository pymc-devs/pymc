"""Convergence diagnostics and model validation"""

import numpy as np
from .stats import statfunc, autocov
from .util import get_default_varnames
from .backends.base import MultiTrace

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


def gelman_rubin(mtrace, varnames=None, include_transformed=False):
    R"""Returns estimate of R for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    mtrace : MultiTrace or trace object
      A MultiTrace object containing parallel traces (minimum 2)
      of one or more stochastic parameters.
    varnames : list
      Names of variables to include in the rhat report
    include_transformed : bool
      Flag for reporting automatically transformed variables in addition
      to original variables (defaults to False).

    Returns
    -------
    Rhat : dict of floats (MultiTrace) or float (trace object)
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

    def rscore(x, num_samples):
        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        return np.sqrt(Vhat / W)

    if not isinstance(mtrace, MultiTrace):
        # Return rscore for passed arrays
        return rscore(np.array(mtrace), mtrace.shape[1])

    if mtrace.nchains < 2:
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains '
            'of the same length.')

    if varnames is None:
        varnames = get_default_varnames(mtrace.varnames, include_transformed=include_transformed)

    Rhat = {}

    for var in varnames:
        x = np.array(mtrace.get_values(var, combine=False))
        num_samples = x.shape[1]
        Rhat[var] = rscore(x, num_samples)

    return Rhat


def effective_n(mtrace, varnames=None, include_transformed=False):
    R"""Returns estimate of the effective sample size of a set of traces.

    Parameters
    ----------
    mtrace : MultiTrace or trace object
      A MultiTrace object containing parallel traces (minimum 2)
      of one or more stochastic parameters.
    varnames : list
      Names of variables to include in the effective_n report
    include_transformed : bool
      Flag for reporting automatically transformed variables in addition
      to original variables (defaults to False).

    Returns
    -------
    n_eff : dictionary of floats (MultiTrace) or float (trace object)
        Return the effective sample size, :math:`\hat{n}_{eff}`

    Notes
    -----
    The diagnostic is computed by:

    .. math:: \hat{n}_{eff} = \frac{mn}{1 + 2 \sum_{t=1}^T \hat{\rho}_t}

    where :math:`\hat{\rho}_t` is the estimated autocorrelation at lag t, and T
    is the first odd positive integer for which the sum
    :math:`\hat{\rho}_{T+1} + \hat{\rho}_{T+1}` is negative.

    The current implementation is similar to Stan, which uses Geyer's initial
    monotone sequence criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    Gelman et al. BDA (2014)"""

    def get_neff(x):
        """Compute the effective sample size for a 2D array
        """
        trace_value = x.T
        nchain, n_samples = trace_value.shape

        acov = np.asarray([autocov(trace_value[chain]) for chain in range(nchain)])

        chain_mean = trace_value.mean(axis=1)
        chain_var = acov[:, 0] * n_samples / (n_samples - 1.)
        acov_t = acov[:, 1] * n_samples / (n_samples - 1.)
        mean_var = np.mean(chain_var)
        var_plus = mean_var * (n_samples - 1.) / n_samples
        var_plus += np.var(chain_mean, ddof=1)

        rho_hat_t = np.zeros(n_samples)
        rho_hat_even = 1.
        rho_hat_t[0] = rho_hat_even
        rho_hat_odd = 1. - (mean_var - np.mean(acov_t)) / var_plus
        rho_hat_t[1] = rho_hat_odd
        # Geyer's initial positive sequence
        max_t = 1
        t = 1
        while t < (n_samples - 2) and (rho_hat_even + rho_hat_odd) >= 0.:
            rho_hat_even = 1. - (mean_var - np.mean(acov[:, t + 1])) / var_plus
            rho_hat_odd = 1. - (mean_var - np.mean(acov[:, t + 2])) / var_plus
            if (rho_hat_even + rho_hat_odd) >= 0:
                rho_hat_t[t + 1] = rho_hat_even
                rho_hat_t[t + 2] = rho_hat_odd
            max_t = t + 2
            t += 2

        # Geyer's initial monotone sequence
        t = 3
        while t <= max_t - 2:
            if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
                rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.
                rho_hat_t[t + 2] = rho_hat_t[t + 1]
            t += 2
        ess = nchain * n_samples
        ess = ess / (-1. + 2. * np.sum(rho_hat_t))
        return ess

    def generate_neff(trace_values):
        x = np.array(trace_values)
        shape = x.shape

        # Make sure to handle scalars correctly, adding extra dimensions if
        # needed. We could use np.squeeze here, but we don't want to squeeze
        # out dummy dimensions that a user inputs.
        if len(shape) == 2:
            x = np.atleast_3d(trace_values)

        # Transpose all dimensions, which makes the loop below
        # easier by moving the axes of the variable to the front instead
        # of the chain and sample axes.
        x = x.transpose()

        # Get an array the same shape as the var
        _n_eff = np.zeros(x.shape[:-2])

        # Iterate over tuples of indices of the shape of var
        for tup in np.ndindex(*list(x.shape[:-2])):
            _n_eff[tup] = get_neff(x[tup])

        if len(shape) == 2:
            return _n_eff[0]

        return np.transpose(_n_eff)

    if not isinstance(mtrace, MultiTrace):
        # Return neff for non-multitrace array
        return generate_neff(mtrace)

    if mtrace.nchains < 2:
        raise ValueError(
            'Calculation of effective sample size requires multiple chains '
            'of the same length.')

    if varnames is None:
        varnames = get_default_varnames(mtrace.varnames, include_transformed=include_transformed)

    n_eff = {}

    for var in varnames:
        n_eff[var] = generate_neff(mtrace.get_values(var, combine=False))

    return n_eff
