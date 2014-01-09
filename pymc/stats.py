"""Utility functions for PyMC"""

import numpy as np
import warnings


__all__ = ['autocorr', 'autocov', 'hpd', 'quantiles', 'mc_error', 'summary']

def statfunc(f):
    """
    Decorator for statistical utility function to automatically
    extract the trace array from whatever object is passed.
    """

    def wrapped_f(pymc_obj, *args, **kwargs):
        burn = kwargs.pop('burn', 0)
        thin = kwargs.pop('thin', 1)
        combine = kwargs.pop('combine', False)

        try:
            var_names = kwargs.pop('vars',  pymc_obj.var_names)
            chains = kwargs.pop('chains', pymc_obj.active_chains)
        except AttributeError:
            # If fails, assume that raw data is passed
            return f(pymc_obj, *args, **kwargs)

        results = {chain: {} for chain in chains}
        for var_name in var_names:
            samples = pymc_obj.get_values(var_name, chains=chains, burn=burn,
                                          thin=thin, combine=combine,
                                          squeeze=False)
            for chain, data in zip(chains, samples):
                results[chain][var_name] = f(np.squeeze(data), *args, **kwargs)

        if len(chains) == 1 or combine:
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
    return S[0, 1]/np.sqrt(np.prod(np.diag(S)))

@statfunc
def autocov(x, lag=1):
    """
    Sample autocovariance at specified lag.
    The autocovariance is a 2x2 matrix with the variances of
    x[:-lag] and x[lag:] in the diagonal and the autocovariance
    on the off-diagonal.
    """
    x = np.asarray(x)

    if not lag: return 1
    if lag < 0:
        raise ValueError("Autocovariance lag must be a positive integer")
    return np.cov(x[:-lag], x[lag:], bias=1)

def make_indices(dimensions):
    # Generates complete set of indices for given dimensions

    level = len(dimensions)

    if level == 1: return list(range(dimensions[0]))

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

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
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
    if x.ndim > 1:

        # Transpose first, then sort
        tx = np.transpose(x, list(range(x.ndim))[1:]+[0])
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
def mc_error(x, batches=5):
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

        return np.reshape([mc_error(t, batches) for t in trace], dims[1:])

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
    if x.ndim > 1:
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


def summary(trace, var_names=None, alpha=0.05, start=0, batches=100,
            roundto=3):
    """
    Generate a pretty-printed summary of the node.

    :Parameters:
    trace : Trace object
      Trace containing MCMC samples

    var_names : list of strings
      List of variables to summarize. Defaults to None, which results
      in all variables summarized.

    alpha : float
      The alpha level for generating posterior intervals. Defaults to
      0.05.

    start : int
      The starting index from which to summarize (each) chain. Defaults
      to zero.

    batches : int
      Batch size for calculating standard deviation for non-independent
      samples. Defaults to 100.

    roundto : int
      The number of digits to round posterior statistics.

    """
    if var_names is None:
        var_names = trace.var_names

    stat_summ = _StatSummary(roundto, batches, alpha)
    pq_summ = _PosteriorQuantileSummary(roundto, alpha)

    for var_name in var_names:
        # Extract sampled values
        sample = trace.get_values(var_name, burn=start, combine=True)
        if sample.ndim == 1:
            sample = sample[:, None]
        elif sample.ndim > 2:
            ## trace dimensions greater than 2 (variable greater than 1)
            warnings.warn('Skipping {} (above 1 dimension)'.format(var_name))
            continue

        print('\n%s:' % var_name)
        print(' ')

        stat_summ.print_output(sample)
        pq_summ.print_output(sample)


class _Summary(object):
    """Base class for summary output"""
    def __init__(self, roundto):
        self.roundto = roundto
        self.header_lines = None
        self.leader = '  '
        self.spaces = None

    def print_output(self, sample):
        print('\n'.join(list(self._get_lines(sample))) + '\n')

    def _get_lines(self, sample):
        for line in self.header_lines:
            yield self.leader + line
        summary_lines = self._calculate_values(sample)
        for line in self._create_value_output(summary_lines):
            yield self.leader + line

    def _create_value_output(self, lines):
        for values in lines:
            self._format_values(values)
            yield self.value_line.format(pad=self.spaces, **values).strip()

    def _calculate_values(self, sample):
        raise NotImplementedError

    def _format_values(self, summary_values):
        for key, val in summary_values.items():
            summary_values[key] = '{:.{ndec}f}'.format(
                float(val), ndec=self.roundto)


class _StatSummary(_Summary):
    def __init__(self, roundto, batches, alpha):
        super(_StatSummary, self).__init__(roundto)
        spaces = 17
        hpd_name = '{}% HPD interval'.format(int(100 * (1 - alpha)))
        value_line = '{mean:<{pad}}{sd:<{pad}}{mce:<{pad}}{hpd:<{pad}}'
        header = value_line.format(mean='Mean', sd='SD', mce='MC Error',
                                  hpd=hpd_name, pad=spaces).strip()
        hline = '-' * len(header)

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
    for index in range(sample.shape[1]):
        mean, sd, mce = [stat[index] for stat in (means, sds, mces)]
        interval = intervals[index].squeeze().tolist()
        yield {'mean': mean, 'sd': sd, 'mce': mce, 'hpd': interval}


def _calculate_posterior_quantiles(sample, qlist):
    var_quantiles = quantiles(sample, qlist=qlist)
    ## Replace ends of qlist with 'lo' and 'hi'
    qends = {qlist[0]: 'lo', qlist[-1]: 'hi'}
    qkeys = {q: qends[q] if q in qends else 'q{}'.format(q) for q in qlist}
    for index in range(sample.shape[1]):
        yield {qkeys[q]: var_quantiles[q][index] for q in qlist}
