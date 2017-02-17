"""Statistical utility functions for PyMC"""

import numpy as np
import pandas as pd
import itertools
import sys
import warnings
from collections import namedtuple
from .model import modelcontext

from scipy.misc import logsumexp
from scipy.stats.distributions import pareto

from .backends import tracetab as ttab

__all__ = ['autocorr', 'autocov', 'dic', 'bpic', 'waic', 'loo', 'hpd', 'quantiles',
           'mc_error', 'summary', 'df_summary', 'compare']


def statfunc(f):
    """
    Decorator for statistical utility function to automatically
    extract the trace array from whatever object is passed.
    """

    def wrapped_f(pymc3_obj, *args, **kwargs):
        try:
            vars = kwargs.pop('vars',  pymc3_obj.varnames)
            chains = kwargs.pop('chains', pymc3_obj.chains)
        except AttributeError:
            # If fails, assume that raw data was passed.
            return f(pymc3_obj, *args, **kwargs)

        burn = kwargs.pop('burn', 0)
        thin = kwargs.pop('thin', 1)
        combine = kwargs.pop('combine', False)
        # Remove outer level chain keys if only one chain)
        squeeze = kwargs.pop('squeeze', True)

        results = {chain: {} for chain in chains}
        for var in vars:
            samples = pymc3_obj.get_values(var, chains=chains, burn=burn,
                                           thin=thin, combine=combine,
                                           squeeze=False)
            for chain, data in zip(chains, samples):
                results[chain][var] = f(np.squeeze(data), *args, **kwargs)

        if squeeze and (len(chains) == 1 or combine):
            results = results[chains[0]]
        return results

    wrapped_f.__doc__ = f.__doc__
    wrapped_f.__name__ = f.__name__

    return wrapped_f


@statfunc
def autocorr(x, lag=1):
    """Sample autocorrelation at specified lag.
    The autocorrelation is the correlation of x_i with x_{i+lag}.
    """

    S = autocov(x, lag)
    return S[0, 1] / np.sqrt(np.prod(np.diag(S)))


@statfunc
def autocov(x, lag=1):
    """
    Sample autocovariance at specified lag.
    The autocovariance is a 2x2 matrix with the variances of
    x[:-lag] and x[lag:] in the diagonal and the autocovariance
    on the off-diagonal.
    """
    x = np.asarray(x)

    if not lag:
        return 1
    if lag < 0:
        raise ValueError("Autocovariance lag must be a positive integer")
    return np.cov(x[:-lag], x[lag:], bias=1)


def dic(trace, model=None):
    """
    Calculate the deviance information criterion of the samples in trace from model
    Read more theory here - in a paper by some of the leading authorities on Model Selection - dx.doi.org/10.1111/1467-9868.00353
    """
    model = modelcontext(model)

    mean_deviance = -2 * np.mean([model.logp(pt) for pt in trace])

    free_rv_means = {rv.name: trace[rv.name].mean(
        axis=0) for rv in model.free_RVs}
    deviance_at_mean = -2 * model.logp(free_rv_means)

    return 2 * mean_deviance - deviance_at_mean


def log_post_trace(trace, model):
    '''
    Calculate the elementwise log-posterior for the sampled trace.
    '''
    return np.vstack([obs.logp_elemwise(pt) for obs in model.observed_RVs] for pt in trace)


def waic(trace, model=None, pointwise=False):
    """
    Calculate the widely available information criterion, its standard error
    and the effective number of parameters of the samples in trace from model.
    Read more theory here - in a paper by some of the leading authorities on
    Model Selection - dx.doi.org/10.1111/1467-9868.00353


    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    pointwise: bool
        if True the pointwise predictive accuracy will be returned.
        Default False


    Returns
    -------
    namedtuple with the following elements:
    waic: widely available information criterion
    waic_se: standard error of waic
    p_waic: effective number parameters
    waic_i: and array of the pointwise predictive accuracy, only if pointwise True
    """
    model = modelcontext(model)

    log_py = log_post_trace(trace, model)

    lppd_i = logsumexp(log_py, axis=0, b=1.0 / log_py.shape[0])

    vars_lpd = np.var(log_py, axis=0)
    if np.any(vars_lpd > 0.4):
        warnings.warn("""For one or more samples the posterior variance of the
        log predictive densities exceeds 0.4. This could be indication of
        WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
        """)
    waic_i = - 2 * (lppd_i - vars_lpd)

    waic_se = np.sqrt(len(waic_i) * np.var(waic_i))

    waic = np.sum(waic_i)

    p_waic = np.sum(vars_lpd)

    if pointwise:
        WAIC_r = namedtuple('WAIC_r', 'WAIC, WAIC_se, p_WAIC, WAIC_i')
        return WAIC_r(waic, waic_se, p_waic, waic_i)
    else:
        WAIC_r = namedtuple('WAIC_r', 'WAIC, WAIC_se, p_WAIC')
        return WAIC_r(waic, waic_se, p_waic)


def loo(trace, model=None, pointwise=False):
    """
    Calculates leave-one-out (LOO) cross-validation for out of sample predictive
    model fit, following Vehtari et al. (2015). Cross-validation is computed using
    Pareto-smoothed importance sampling (PSIS).


    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    pointwise: bool
        if True the pointwise predictive accuracy will be returned.
        Default False


    Returns
    -------
    namedtuple with the following elements:
    loo: approximated Leave-one-out cross-validation
    loo_se: standard error of loo
    p_loo: effective number of parameters
    loo_i: and array of the pointwise predictive accuracy, only if pointwise True
    """

    model = modelcontext(model)

    log_py = log_post_trace(trace, model)

    # Importance ratios
    r = np.exp(-log_py)
    r_sorted = np.sort(r, axis=0)

    # Extract largest 20% of importance ratios and fit generalized Pareto to each
    # (returns tuple with shape, location, scale)
    q80 = int(len(log_py) * 0.8)
    pareto_fit = np.apply_along_axis(
        lambda x: pareto.fit(x, floc=0), 0, r_sorted[q80:])

    if np.any(pareto_fit[0] > 0.7):
        warnings.warn("""Estimated shape parameter of Pareto distribution is
        greater than 0.7 for one or more samples.
        You should consider using a more robust model, this is
        because importance sampling is less likely to work well if the marginal
        posterior and LOO posterior are very different. This is more likely to
        happen with a non-robust model and highly influential observations.""")

    elif np.any(pareto_fit[0] > 0.5):
        warnings.warn("""Estimated shape parameter of Pareto distribution is
        greater than 0.5 for one or more samples. This may indicate
        that the variance of the Pareto smoothed importance sampling estimate
        is very large.""")

    # Calculate expected values of the order statistics of the fitted Pareto
    S = len(r_sorted)
    M = S - q80
    z = (np.arange(M) + 0.5) / M
    expvals = map(lambda x: pareto.ppf(z, x[0], scale=x[2]), pareto_fit.T)

    # Replace importance ratios with order statistics of fitted Pareto
    r_sorted[q80:] = np.vstack(expvals).T
    # Unsort ratios (within columns) before using them as weights
    r_new = np.array([r[np.argsort(i)]
                      for r, i in zip(r_sorted.T, np.argsort(r.T, axis=1))]).T

    # Truncate weights to guarantee finite variance
    w = np.minimum(r_new, r_new.mean(axis=0) * S**0.75)

    loo_lppd_i = - 2. * logsumexp(log_py, axis=0, b=w / np.sum(w, axis=0))

    loo_lppd_se = np.sqrt(len(loo_lppd_i) * np.var(loo_lppd_i))

    loo_lppd = np.sum(loo_lppd_i)

    lppd = np.sum(logsumexp(log_py, axis=0, b=1. / log_py.shape[0]))

    p_loo = lppd + (0.5 * loo_lppd)

    if pointwise:
        LOO_r = namedtuple('LOO_r', 'LOO, LOO_se, p_LOO, LOO_i')
        return LOO_r(loo_lppd, loo_lppd_se, p_loo, loo_lppd_i)
    else:
        LOO_r = namedtuple('LOO_r', 'LOO, LOO_se, p_LOO')
        return LOO_r(loo_lppd, loo_lppd_se, p_loo)


def bpic(trace, model=None):
    """
    Calculates Bayesian predictive information criterion n of the samples in trace from model
    Read more theory here - in a paper by some of the leading authorities on Model Selection - dx.doi.org/10.1111/1467-9868.00353
    """
    model = modelcontext(model)

    mean_deviance = -2 * np.mean([model.logp(pt) for pt in trace])

    free_rv_means = {rv.name: trace[rv.name].mean(
        axis=0) for rv in model.free_RVs}
    deviance_at_mean = -2 * model.logp(free_rv_means)

    return 3 * mean_deviance - 2 * deviance_at_mean


def compare(traces, models, ic='WAIC'):
    """
    Compare models based on the widely available information criterion (WAIC)
    or leave-one-out (LOO) cross-validation.
    Read more theory here - in a paper by some of the leading authorities on
    Model Selection - dx.doi.org/10.1111/1467-9868.00353

    Parameters
    ----------
    traces : list of PyMC3 traces
    models : list of PyMC3 models
        in the same order as traces.
    ic : string
        Information Criterion (WAIC or LOO) used to compare models.
        Default WAIC.

    Returns
    -------
    A DataFrame, ordered from lowest to highest IC. The index reflects
    the order in which the models are passed to this function. The columns are:
    IC : Information Criteria (WAIC or LOO).
        Smaller IC indicates higher out-of-sample predictive fit ("better" model). 
        Default WAIC. 
    pIC : Estimated effective number of parameters.
    dIC : Relative difference between each IC (WAIC or LOO)
    and the lowest IC (WAIC or LOO).
        It's always 0 for the top-ranked model.
    weight: Akaike weights for each model. 
        This can be loosely interpreted as the probability of each model
        (among the compared model) given the data. Be careful that these
        weights are based on point estimates of the IC (uncertainty is ignored).
    SE : Standard error of the IC estimate.
        For a "large enough" sample size this is an estimate of the uncertainty
        in the computation of the IC.
    dSE : Standard error of the difference in IC between each model and
    the top-ranked model.
        It's always 0 for the top-ranked model.
    warning : A value of 1 indicates that the computation of the IC may not be
    reliable see http://arxiv.org/abs/1507.04544 for details.
    """
    if ic == 'WAIC':
        ic_func = waic
        df_comp = pd.DataFrame(index=np.arange(len(models)),
                               columns=['WAIC', 'pWAIC', 'dWAIC', 'weight',
                               'SE', 'dSE', 'warning'])
    elif ic == 'LOO':
        ic_func = loo
        df_comp = pd.DataFrame(index=np.arange(len(models)),
                               columns=['LOO', 'pLOO', 'dLOO', 'weight',
                               'SE', 'dSE', 'warning'])
    else:
        raise NotImplementedError(
            'The information criterion {} is not supported.'.format(ic))

    warns = np.zeros(len(models))

    c = 0
    def add_warns(*args):
        warns[c] = 1

    with warnings.catch_warnings():
        warnings.showwarning = add_warns
        warnings.filterwarnings('always')

        ics = []
        for c, (t, m) in enumerate(zip(traces, models)):
            ics.append((c, ic_func(t, m, pointwise=True)))

    ics.sort(key=lambda x: x[1][0])

    min_ic = ics[0][1][0]
    Z = np.sum([np.exp(-0.5 * (x[1][0] - min_ic)) for x in ics])

    for idx, res in ics:
        diff = ics[0][1][3] - res[3]
        d_ic = np.sum(diff)
        d_se = len(diff) ** 0.5 * np.var(diff)
        weight = np.exp(-0.5 * (res[0] - min_ic)) / Z
        df_comp.at[idx] = (res[0], res[2], abs(d_ic), weight, res[1],
                           d_se, warns[idx])

    return df_comp.sort_values(by=ic)


def make_indices(dimensions):
    # Generates complete set of indices for given dimensions

    level = len(dimensions)

    if level == 1:
        return list(range(dimensions[0]))

    indices = [[]]

    while level:

        _indices = []

        for j in range(dimensions[level - 1]):

            _indices += [[j] + i for i in indices]

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
    cred_mass = 1.0 - alpha

    interval_idx_inc = int(np.floor(cred_mass * n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return hdi_min, hdi_max


@statfunc
def hpd(x, alpha=0.05, transform=lambda x: x):
    """Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      alpha : float
          Desired probability of type I error (defaults to 0.05)
      transform : callable
          Function to transform data (defaults to identity)

    """

    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:] + [0])
        dims = np.shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, dims[:-1] + (2,))

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
def mc_error(x, batches=5):
    """
    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      batches : integer
          Number of batches
    """

    if x.ndim > 1:

        dims = np.shape(x)
        #ttrace = np.transpose(np.reshape(trace, (dims[0], sum(dims[1:]))))
        trace = np.transpose([t.ravel() for t in x])

        return np.reshape([mc_error(t, batches) for t in trace], dims[1:])

    else:
        if batches == 1:
            return np.std(x) / np.sqrt(len(x))

        try:
            batched_traces = np.resize(x, (batches, int(len(x) / batches)))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(x) % batches
            new_shape = (batches, (len(x) - resid) / batches)
            batched_traces = np.resize(x[:-resid], new_shape)

        means = np.mean(batched_traces, 1)

        return np.std(means) / np.sqrt(batches)


@statfunc
def quantiles(x, qlist=(2.5, 25, 50, 75, 97.5), transform=lambda x: x):
    """Returns a dictionary of requested quantiles from array

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      qlist : tuple or list
          A list of desired quantiles (defaults to (2.5, 25, 50, 75, 97.5))
      transform : callable
          Function to transform data (defaults to identity)
    """

    # Make a copy of trace
    x = transform(x.copy())

    # For multivariate node
    if x.ndim > 1:
        # Transpose first, then sort, then transpose back
        sx = np.sort(x.T).T
    else:
        # Sort univariate node
        sx = np.sort(x)

    try:
        # Generate specified quantiles
        quants = [sx[int(len(sx) * q / 100.0)] for q in qlist]

        return dict(zip(qlist, quants))

    except IndexError:
        _log.warning("Too few elements for quantile calculation")


def df_summary(trace, varnames=None, stat_funcs=None, extend=False, include_transformed=False,
               alpha=0.05, batches=None):
    R"""Create a data frame with summary statistics.

    Parameters
    ----------
    trace : MultiTrace instance
    varnames : list
        Names of variables to include in summary
    stat_funcs : None or list
        A list of functions used to calculate statistics. By default,
        the mean, standard deviation, simulation standard error, and
        highest posterior density intervals are included.

        The functions will be given one argument, the samples for a
        variable as a 2 dimensional array, where the first axis
        corresponds to sampling iterations and the second axis
        represents the flattened variable (e.g., x__0, x__1,...). Each
        function should return either
        1) A `pandas.Series` instance containing the result of
           calculating the statistic along the first axis. The name
           attribute will be taken as the name of the statistic.
        2) A `pandas.DataFrame` where each column contains the
           result of calculating the statistic along the first axis.
           The column names will be taken as the names of the
           statistics.
    extend : boolean
        If True, use the statistics returned by `stat_funcs` in
        addition to, rather than in place of, the default statistics.
        This is only meaningful when `stat_funcs` is not None.
    include_transformed : bool
        Flag for reporting automatically transformed variables in addition
        to original variables (defaults to False).
    alpha : float
        The alpha level for generating posterior intervals. Defaults
        to 0.05. This is only meaningful when `stat_funcs` is None.
    batches : None or int
        Batch size for calculating standard deviation for non-independent
        samples. Defaults to the smaller of 100 or the number of samples.
        This is only meaningful when `stat_funcs` is None.


    See also
    --------
    summary : Generate a pretty-printed summary of a trace.


    Returns
    -------
    `pandas.DataFrame` with summary statistics for each variable


    Examples
    --------
    >>> import pymc3 as pm
    >>> trace.mu.shape
    (1000, 2)
    >>> pm.df_summary(trace, ['mu'])
               mean        sd  mc_error     hpd_5    hpd_95
    mu__0  0.106897  0.066473  0.001818 -0.020612  0.231626
    mu__1 -0.046597  0.067513  0.002048 -0.174753  0.081924

    Other statistics can be calculated by passing a list of functions.

    >>> import pandas as pd
    >>> def trace_sd(x):
    ...     return pd.Series(np.std(x, 0), name='sd')
    ...
    >>> def trace_quantiles(x):
    ...     return pd.DataFrame(pm.quantiles(x, [5, 50, 95]))
    ...
    >>> pm.df_summary(trace, ['mu'], stat_funcs=[trace_sd, trace_quantiles])
                 sd         5        50        95
    mu__0  0.066473  0.000312  0.105039  0.214242
    mu__1  0.067513 -0.159097 -0.045637  0.062912
    """
    if varnames is None:
        if include_transformed:
            varnames = [name for name in trace.varnames]
        else:
            varnames = [name for name in trace.varnames if not name.endswith('_')]

    if batches is None:
        batches = min([100, len(trace)])

    funcs = [lambda x: pd.Series(np.mean(x, 0), name='mean'),
             lambda x: pd.Series(np.std(x, 0), name='sd'),
             lambda x: pd.Series(mc_error(x, batches), name='mc_error'),
             lambda x: _hpd_df(x, alpha)]

    if stat_funcs is not None and extend:
        stat_funcs = funcs + stat_funcs
    elif stat_funcs is None:
        stat_funcs = funcs

    var_dfs = []
    for var in varnames:
        vals = trace.get_values(var, combine=True)
        flat_vals = vals.reshape(vals.shape[0], -1)
        var_df = pd.concat([f(flat_vals) for f in stat_funcs], axis=1)
        var_df.index = ttab.create_flat_names(var, vals.shape[1:])
        var_dfs.append(var_df)
    return pd.concat(var_dfs, axis=0)


def _hpd_df(x, alpha):
    cnames = ['hpd_{0:g}'.format(100 * alpha / 2),
              'hpd_{0:g}'.format(100 * (1 - alpha / 2))]
    return pd.DataFrame(hpd(x, alpha), columns=cnames)


def summary(trace, varnames=None, alpha=0.05, start=0, batches=None, roundto=3,
            include_transformed=False, to_file=None):
    R"""
    Generate a pretty-printed summary of the node.

    Parameters
    ----------
    trace : Trace object
      Trace containing MCMC sample
    varnames : list of strings
      List of variables to summarize. Defaults to None, which results
      in all variables summarized.
    alpha : float
      The alpha level for generating posterior intervals. Defaults to
      0.05.
    start : int
      The starting index from which to summarize (each) chain. Defaults
      to zero.
    batches : None or int
        Batch size for calculating standard deviation for non-independent
        samples. Defaults to the smaller of 100 or the number of samples.
        This is only meaningful when `stat_funcs` is None.
    roundto : int
      The number of digits to round posterior statistics.
    include_transformed : bool
      Flag for summarizing automatically transformed variables in addition to
      original variables (defaults to False).
    to_file : None or string
      File to write results to. If not given, print to stdout.

    """
    if varnames is None:
        if include_transformed:
            varnames = [name for name in trace.varnames]
        else:
            varnames = [name for name in trace.varnames if not name.endswith('_')]

    if batches is None:
        batches = min([100, len(trace)])

    stat_summ = _StatSummary(roundto, batches, alpha)
    pq_summ = _PosteriorQuantileSummary(roundto, alpha)

    if to_file is None:
        fh = sys.stdout
    else:
        fh = open(to_file, mode='w')

    for var in varnames:
        # Extract sampled values
        sample = trace.get_values(var, burn=start, combine=True)

        fh.write('\n%s:\n\n' % var)

        fh.write(stat_summ.output(sample))
        fh.write(pq_summ.output(sample))

    if fh is not sys.stdout:
        fh.close()


class _Summary(object):
    """Base class for summary output"""

    def __init__(self, roundto):
        self.roundto = roundto
        self.header_lines = None
        self.leader = '  '
        self.spaces = None
        self.width = None

    def output(self, sample):
        return '\n'.join(list(self._get_lines(sample))) + '\n\n'

    def _get_lines(self, sample):
        for line in self.header_lines:
            yield self.leader + line
        summary_lines = self._calculate_values(sample)
        for line in self._create_value_output(summary_lines):
            yield self.leader + line

    def _create_value_output(self, lines):
        for values in lines:
            try:
                self._format_values(values)
                yield self.value_line.format(pad=self.spaces, **values).strip()
            except AttributeError:
                # This is a key for the leading indices, not a normal row.
                # `values` will be an empty tuple unless it is 2d or above.
                if values:
                    leading_idxs = [str(v) for v in values]
                    numpy_idx = '[{}, :]'.format(', '.join(leading_idxs))
                    yield self._create_idx_row(numpy_idx)
                else:
                    yield ''

    def _calculate_values(self, sample):
        raise NotImplementedError

    def _format_values(self, summary_values):
        for key, val in summary_values.items():
            summary_values[key] = '{:.{ndec}f}'.format(
                float(val), ndec=self.roundto)

    def _create_idx_row(self, value):
        return '{:.^{}}'.format(value, self.width)


class _StatSummary(_Summary):

    def __init__(self, roundto, batches, alpha):
        super(_StatSummary, self).__init__(roundto)
        spaces = 17
        hpd_name = '{0:g}% HPD interval'.format(100 * (1 - alpha))
        value_line = '{mean:<{pad}}{sd:<{pad}}{mce:<{pad}}{hpd:<{pad}}'
        header = value_line.format(mean='Mean', sd='SD', mce='MC Error',
                                   hpd=hpd_name, pad=spaces).strip()
        self.width = len(header)
        hline = '-' * self.width

        self.header_lines = [header, hline]
        self.spaces = spaces
        self.value_line = value_line
        self.batches = batches
        self.alpha = alpha

    def _calculate_values(self, sample):
        return _calculate_stats(sample, self.batches, self.alpha)

    def _format_values(self, summary_values):
        roundto = self.roundto
        for key, val in summary_values.items():
            if key == 'hpd':
                summary_values[key] = '[{:.{ndec}f}, {:.{ndec}f}]'.format(
                    *val, ndec=roundto)
            else:
                summary_values[key] = '{:.{ndec}f}'.format(
                    float(val), ndec=roundto)


class _PosteriorQuantileSummary(_Summary):

    def __init__(self, roundto, alpha):
        super(_PosteriorQuantileSummary, self).__init__(roundto)
        spaces = 15
        title = 'Posterior quantiles:'
        value_line = '{lo:<{pad}}{q25:<{pad}}{q50:<{pad}}{q75:<{pad}}{hi:<{pad}}'
        lo, hi = 100 * alpha / 2, 100 * (1. - alpha / 2)
        qlist = (lo, 25, 50, 75, hi)
        header = value_line.format(lo=lo, q25=25, q50=50, q75=75, hi=hi,
                                   pad=spaces).strip()
        self.width = len(header)
        hline = '|{thin}|{thick}|{thick}|{thin}|'.format(
            thin='-' * (spaces - 1), thick='=' * (spaces - 1))

        self.header_lines = [title, header, hline]
        self.spaces = spaces
        self.lo, self.hi = lo, hi
        self.qlist = qlist
        self.value_line = value_line

    def _calculate_values(self, sample):
        return _calculate_posterior_quantiles(sample, self.qlist)


def _calculate_stats(sample, batches, alpha):
    means = sample.mean(0)
    sds = sample.std(0)
    mces = mc_error(sample, batches)
    intervals = hpd(sample, alpha)
    for key, idxs in _groupby_leading_idxs(sample.shape[1:]):
        yield key
        for idx in idxs:
            mean, sd, mce = [stat[idx] for stat in (means, sds, mces)]
            interval = intervals[idx].squeeze().tolist()
            yield {'mean': mean, 'sd': sd, 'mce': mce, 'hpd': interval}


def _calculate_posterior_quantiles(sample, qlist):
    var_quantiles = quantiles(sample, qlist=qlist)
    # Replace ends of qlist with 'lo' and 'hi'
    qends = {qlist[0]: 'lo', qlist[-1]: 'hi'}
    qkeys = {q: qends[q] if q in qends else 'q{}'.format(q) for q in qlist}
    for key, idxs in _groupby_leading_idxs(sample.shape[1:]):
        yield key
        for idx in idxs:
            yield {qkeys[q]: var_quantiles[q][idx] for q in qlist}


def _groupby_leading_idxs(shape):
    """Group the indices for `shape` by the leading indices of `shape`.

    All dimensions except for the rightmost dimension are used to create
    groups.

    A 3d shape will be grouped by the indices for the two leading
    dimensions.

        >>> for key, idxs in _groupby_leading_idxs((3, 2, 2)):
        ...     print('key: {}'.format(key))
        ...     print(list(idxs))
        key: (0, 0)
        [(0, 0, 0), (0, 0, 1)]
        key: (0, 1)
        [(0, 1, 0), (0, 1, 1)]
        key: (1, 0)
        [(1, 0, 0), (1, 0, 1)]
        key: (1, 1)
        [(1, 1, 0), (1, 1, 1)]
        key: (2, 0)
        [(2, 0, 0), (2, 0, 1)]
        key: (2, 1)
        [(2, 1, 0), (2, 1, 1)]

    A 1d shape will only have one group.

        >>> for key, idxs in _groupby_leading_idxs((2,)):
        ...     print('key: {}'.format(key))
        ...     print(list(idxs))
        key: ()
        [(0,), (1,)]
    """
    idxs = itertools.product(*[range(s) for s in shape])
    return itertools.groupby(idxs, lambda x: x[:-1])
