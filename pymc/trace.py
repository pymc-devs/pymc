'''
Created on Mar 15, 2011

@author: jsalvatier
'''
import numpy as np
from core import *

__all__ = ['NpTrace', 'MultiTrace']

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
        records the position of a chain at a certain point in time
        """
        for var, value in zip(self.varnames, self.f(point)):
            self.samples[var].append(value)
        return self

    def __getitem__(self, key):
        try :
            return self.point(key)
        except ValueError:
            pass
        except TypeError:
            pass
        return self.samples[str(key)].value

    def point(self, index):
        return dict((k, v.value[index]) for (k,v) in self.samples.iteritems())

def stats(trace, alpha=0.05, start=0, batches=100, chain=None, quantiles=(2.5, 25, 50, 75, 97.5)):
    """
    Generate posterior statistics from trace.

    :Parameters:
    name : string
      The name of the tallyable object.

    alpha : float
      The alpha level for generating posterior intervals. Defaults to
      0.05.

    start : int
      The starting index from which to summarize (each) chain. Defaults
      to zero.

    batches : int
      Batch size for calculating standard deviation for non-independent
      samples. Defaults to 100.

    chain : int
      The index for which chain to summarize. Defaults to None (all
      chains).

    quantiles : tuple or list
      The desired quantiles to be calculated. Defaults to (2.5, 25, 50, 75, 97.5).
    """

    def var_stats(trace):


            x = trace[start:]

            n = len(x)
            if not n:
                print('Cannot generate statistics for zero-length trace.')
                return

            stats =  {
                'n': n,
                'standard deviation': x.std(0),
                'mean': x.mean(0),
                '%s%s HPD interval' % (int(100*(1-alpha)),'%'): hpd(x, alpha),
                'mc error': batchsd(x, batches),
                'quantiles': calc_quantiles(x, qlist=quantiles)
            }

            return stats




    try:
        # For entire Trace object
        stats = {}
        for v in trace.varnames:
            try:
                stats[v] = var_stats(trace[v])
            except:
                print('Could not generate output statistics for ' + v)
        return stats

    except AttributeError:
        # For trace of single variable
        return var_stats(trace)


def summary(trace, alpha=0.05, start=0, batches=100, roundto=3):
    """
    Generate a pretty-printed summary of the node.

    :Parameters:
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

    def var_summary(statdict):

        size = np.size(statdict['mean'])

        # Initialize buffer
        buffer = []

        # Index to interval label
        iindex = [key.split()[-1] for key in statdict.keys()].index('interval')
        interval = statdict.keys()[iindex]

        # Print basic stats
        buffer += ['Mean             SD               MC Error        %s' % interval]
        buffer += ['-'*len(buffer[-1])]

        indices = range(size)
        if len(indices)==1:
            indices = [None]

        for index in indices:
            # Extract statistics and convert to string
            m = str(round(statdict['mean'][index], roundto))
            sd = str(round(statdict['standard deviation'][index], roundto))
            mce = str(round(statdict['mc error'][index], roundto))
            hpd = str(statdict[interval][index].squeeze().round(roundto))

            # Build up string buffer of values
            valstr = m
            valstr += ' '*(17-len(m)) + sd
            valstr += ' '*(17-len(sd)) + mce
            valstr += ' '*(len(buffer[-1]) - len(valstr) - len(hpd)) + hpd

            buffer += [valstr]

        buffer += ['']*2

        # Print quantiles
        buffer += ['Posterior quantiles:','']

        buffer += ['2.5             25              50              75             97.5']
        buffer += [' |---------------|===============|===============|---------------|']

        for index in indices:
            quantile_str = ''
            for i,q in enumerate((2.5, 25, 50, 75, 97.5)):
                qstr = str(round(statdict['quantiles'][q][index], roundto))
                quantile_str += qstr + ' '*(17-i-len(qstr))
            buffer += [quantile_str.strip()]

        buffer += ['']

        print('\t' + '\n\t'.join(buffer))


    # Calculate statistics
    s = stats(trace, alpha=alpha, start=start, batches=batches)

    try:
        for var in trace.varnames:

            try:
                vstats = s[var]

                print('\n%s:' % var)
                print(' ')

                var_summary(vstats)
            except KeyError:

                pass

    except AttributeError:

        var_summary(s)


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
        print('Too few elements for interval calculation')
        return [None,None]

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

def hpd(x, alpha):
    """Calculate highest posterior density (HPD) of array for given alpha. The HPD is the
    minimum width Bayesian credible interval (BCI).

    :Arguments:
      x : Numpy array
          An array containing MCMC samples
      alpha : float
          Desired probability of type I error

    """

    # Make a copy of trace
    x = x.copy()

    # For multivariate node
    if x.ndim>1:

        # Transpose first, then sort
        tx = tr(x, range(x.ndim)[1:]+[0])
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


def batchsd(trace, batches=5):
    """
    Calculates the simulation standard error, accounting for non-independent
    samples. The trace is divided into batches, and the standard deviation of
    the batch means is calculated.
    """

    if len(np.shape(trace)) > 1:

        dims = np.shape(trace)
        #ttrace = np.transpose(np.reshape(trace, (dims[0], sum(dims[1:]))))
        ttrace = np.transpose([t.ravel() for t in trace])

        return np.reshape([batchsd(t, batches) for t in ttrace], dims[1:])

    else:
        if batches == 1: return np.std(trace)/np.sqrt(len(trace))

        try:
            batched_traces = np.resize(trace, (batches, len(trace)/batches))
        except ValueError:
            # If batches do not divide evenly, trim excess samples
            resid = len(trace) % batches
            batched_traces = np.resize(trace[:-resid], (batches, len(trace)/batches))

        means = np.mean(batched_traces, 1)

        return np.std(means)/np.sqrt(batches)

def calc_quantiles(x, qlist=(2.5, 25, 50, 75, 97.5)):
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