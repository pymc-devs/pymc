import numpy as np
from .core import *
from .stats import *
import copy
import types
import warnings

__all__ = ['NpTrace', 'MultiTrace', 'summary']

class NpTrace(object):
    """
    encapsulates the recording of a process chain
    """
    def __init__(self, vars):
        vars = list(vars)
        model = vars[0].model
        self.f = model.fastfn(vars)
        self.vars = vars
        self.varnames = list(map(str, vars))
        self.samples = dict((v, ListArray()) for v in self.varnames)

    def record(self, point):
        """
        Records the position of a chain at a certain point in time.
        """
        for var, value in zip(self.varnames, self.f(point)):
            self.samples[var].append(value)
        return self

    def __getitem__(self, index_value):
        """
        Return copy NpTrace with sliced sample values if a slice is passed,
        or the array of samples if a varname is passed.
        """

        if isinstance(index_value, slice):

            sliced_trace = NpTrace(self.vars)
            sliced_trace.samples = dict((name, vals[index_value]) for (name, vals) in self.samples.items())

            return sliced_trace

        else:
            try:
                return self.point(index_value)
            except ValueError:
                pass
            except TypeError:
                pass

            return self.samples[str(index_value)].value

    def __len__(self):
        return len(self.samples[self.varnames[0]])

    def point(self, index):
        return dict((k, v.value[index]) for (k, v) in self.samples.items())


class ListArray(object):
    def __init__(self, *args):
        self.vals = list(args)

    @property
    def value(self):
        if len(self.vals) > 1:
            self.vals = [np.concatenate(self.vals, axis=0)]

        return self.vals[0]

    def __getitem__(self, idx): 
        return ListArray(self.value[idx])


    def append(self, v):
        self.vals.append(v[np.newaxis])

    def __len__(self):
        if self.vals:
            return self.value.shape[0]
        else:
            return 0


class MultiTrace(object):
    def __init__(self, traces, vars=None):
        try:
            self.traces = list(traces)
        except TypeError:
            if vars is None:
                raise ValueError("vars can't be None if trace count specified")
            self.traces = [NpTrace(vars) for _ in range(traces)]

    def __getitem__(self, index_value):

        item_list = [h[index_value] for h in self.traces]

        if isinstance(index_value, slice):
            return MultiTrace(item_list)
        return item_list

    @property
    def varnames(self):
        return self.traces[0].varnames

    def point(self, index):
        return [h.point(index) for h in self.traces]

    def combined(self):
        # Returns a trace consisting of concatenated MultiTrace elements
        h = NpTrace(self.traces[0].vars)
        for k in self.traces[0].samples:
            h.samples[k].vals = [s[k] for s in self.traces]
        return h


def summary(trace, vars=None, alpha=0.05, start=0, batches=100, roundto=3):
    """
    Generate a pretty-printed summary of the node.

    :Parameters:
    trace : Trace object
      Trace containing MCMC sample

    vars : list of strings
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
    if vars is None:
        vars = trace.varnames
    if isinstance(trace, MultiTrace):
        trace = trace.combined()

    stat_summ = _StatSummary(roundto, batches, alpha)
    pq_summ = _PosteriorQuantileSummary(roundto, alpha)

    for var in vars:
        # Extract sampled values
        sample = trace[var][start:]
        if sample.ndim == 1:
            sample = sample[:, None]
        elif sample.ndim > 2:
            ## trace dimensions greater than 2 (variable greater than 1)
            warnings.warn('Skipping {} (above 1 dimension)'.format(var))
            continue

        print('\n%s:' % var)
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
