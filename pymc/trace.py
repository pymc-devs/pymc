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
        self.f = compilef(vars)
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

    def copy(self):
        return copy.deepcopy(self)

    def __getitem__(self, index_value):
        """
        Return copy NpTrace with sliced sample values if a slice is passed,
        or the array of samples if a varname is passed.
        """

        if isinstance(index_value, slice):

            sliced_trace = self.copy()
            for v in sliced_trace.varnames:
                sliced_trace.samples[v].vals = [sliced_trace.samples[v].value[index_value]]

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
    def __init__(self):
        self.vals = []

    @property
    def value(self):
        if len(self.vals) > 1:
            self.vals = [np.concatenate(self.vals, axis=0)]

        return self.vals[0]

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

        # Initialize buffer
        buffer = []

        # Print basic stats
        buffer += ['Mean             SD               MC Error        {0}% HPD interval'.format((int(100*(1-alpha))))]
        buffer += ['-'*len(buffer[-1])]

        means = sample.mean(0)
        sds = sample.std(0)
        mces = mc_error(sample, batches)
        intervals = hpd(sample, alpha)

        indices = list(range(sample.shape[1]))
        for index in indices:
            # Extract statistics and convert to string
            m = str(round(means[index], roundto))
            sd = str(round(sds[index], roundto))
            mce = str(round(mces[index], roundto))
            interval = str(intervals[index].squeeze().round(roundto))
            # Build up string buffer of values
            valstr = m
            valstr += ' '*(17-len(m)) + sd
            valstr += ' '*(17-len(sd)) + mce
            valstr += ' '*(len(buffer[-1]) - len(valstr) - len(interval)) + interval

            buffer += [valstr]

        buffer += ['']*2

        # Print quantiles
        buffer += ['Posterior quantiles:','']
        lo, hi = 100*alpha/2, 100*(1.-alpha/2)
        buffer += ['{0}             25              50              75             {1}'.format(lo, hi)]
        buffer += [' |---------------|===============|===============|---------------|']
        qlist = (lo, 25, 50, 75, hi)
        var_quantiles = quantiles(sample, qlist=qlist)
        for index in indices:
            quantile_str = ''
            for i, q in enumerate(qlist):
                qstr = str(round(var_quantiles[q][index], roundto))
                quantile_str += qstr + ' '*(17-i-len(qstr))
            buffer += [quantile_str.strip()]
        buffer += ['']

        print('\t' + '\n\t'.join(buffer))
