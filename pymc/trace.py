import numpy as np
from .core import *
from .stats import *
import copy
import types

__all__ = ['NpTrace', 'MultiTrace', 'summary']

class NpTrace(object):
    """
    encapsulates the recording of a process chain
    """
    def __init__(self, vars):
        self.f = compilef(vars)
        self.vars = vars
        self.varnames =map(str, vars)
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

        if type(index_value) is types.SliceType:

            sliced_trace = self.copy()
            for v in sliced_trace.varnames:
                sliced_trace.samples[v].vals = [s[index_value] for s in self.samples[v].vals]

            return sliced_trace

        else:
            try :
                return self.point(index_value)
            except ValueError:
                pass
            except TypeError:
                pass

            return self.samples[str(index_value)].value


    def point(self, index):
        return dict((k, v.value[index]) for (k,v) in self.samples.iteritems())


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

    for var in vars:

        print('\n%s:' % var)
        print(' ')

        # Extract sampled values
        sample = trace[var][start:]

        shape = sample.shape
        if len(shape)>1:
            size = shape[1]
        else:
            size = 1

        # Initialize buffer
        buffer = []

        # Print basic stats
        buffer += ['Mean             SD               MC Error        {0}% HPD interval'.format((int(100*(1-alpha))))]
        buffer += ['-'*len(buffer[-1])]

        indices = range(size)
        if len(indices)==1:
            indices = [None]

        for index in indices:
            # Extract statistics and convert to string
            m = str(round(sample.mean(0)[index], roundto))
            sd = str(round(sample.std(0)[index], roundto))
            mce = str(round(mc_error(sample, batches)[index], roundto))
            interval = str(hpd(sample, alpha)[index].squeeze().round(roundto))

            # Build up string buffer of values
            valstr = m
            valstr += ' '*(17-len(m)) + sd
            valstr += ' '*(17-len(sd)) + mce
            valstr += ' '*(len(buffer[-1]) - len(valstr) - len(interval)) + interval

            buffer += [valstr]

        buffer += ['']*2

        # Print quantiles
        buffer += ['Posterior quantiles:','']
        lo,hi = 100*alpha/2, 100*(1.-alpha/2)
        buffer += ['{0}             25              50              75             {1}'.format(lo, hi)]
        buffer += [' |---------------|===============|===============|---------------|']

        for index in indices:
            quantile_str = ''
            for i,q in enumerate((lo, 25, 50, 75, hi)):
                qstr = str(round(quantiles(sample, qlist=(lo, 25, 50, 75, hi))[q][index], roundto))
                quantile_str += qstr + ' '*(17-i-len(qstr))
            buffer += [quantile_str.strip()]

        buffer += ['']

        print('\t' + '\n\t'.join(buffer))


class ListArray(object):
    def __init__(self):
        self.vals = []

    @property
    def value(self):
        if len(self.vals) > 1:
            self.vals = [np.concatenate(self.vals, axis =0)]
        return self.vals[0]

    def append(self, v):
        self.vals.append(v[np.newaxis])


class MultiTrace(object):
    def __init__(self, traces, vars = None):
        try :
            self.traces = list(traces)
        except TypeError:
            if vars is None:
                raise ValueError("vars can't be None if trace count specified")
            self.traces = [NpTrace(vars) for _ in xrange(traces)]

    def __getitem__(self, key):
        return [h[key] for h in self.traces]
    def point(self, index):
        return [h.point(index) for h in self.traces]

    def combined(self):
        h = NpTrace(self.traces[0].vars)
        for k in self.traces[0].samples:
            h.samples[k].vals = [s[k] for s in self.traces]
        return h