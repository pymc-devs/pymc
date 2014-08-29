"""Utility functions for PyMC"""


# License: Scipy compatible
# Author: David Huard, 2006

import numpy as np
import sys
import select
import os
import time
from copy import copy
from .PyMCObjects import Variable
from . import flib
import pdb
from numpy.linalg.linalg import LinAlgError
from numpy.linalg import cholesky, eigh
from .Node import logp_of_set, logp_gradient_of_set
import types
from .datatypes import *

from . import six
from .six import print_
from functools import reduce
reduce = six.moves.reduce

from numpy import (sqrt, ndarray, asmatrix, array, prod,
                   asarray, atleast_1d, iterable, linspace, diff,
                   around, log10, zeros, arange, digitize, apply_along_axis,
                   concatenate, bincount, sort, hsplit, argsort, inf, shape,
                   ndim, swapaxes, ravel, diag, cov, transpose as tr)

__all__ = ['append', 'check_list', 'autocorr', 'calc_min_interval',
           'check_type', 'ar1',
           'ar1_gen', 'draw_random', 'histogram', 'hpd', 'invcdf',
           'make_indices', 'normcdf', 'quantiles', 'rec_getattr',
           'rec_setattr', 'round_array', 'trace_generator', 'msqrt', 'safe_len',
           'log_difference', 'find_generations', 'crawl_dataless', 'logit',
           'invlogit', 'stukel_logit', 'stukel_invlogit', 'symmetrize', 'value']

symmetrize = flib.symmetrize


def value(a):
    """
    Returns a.value if a is a Variable, or just a otherwise.
    """
    if isinstance(a, Variable):
        return a.value
    else:
        return a


# =====================================================================
# = Please don't use numpy.vectorize with these! It will leak memory. =
# =====================================================================
def logit(theta):
    return flib.logit(ravel(theta)).reshape(shape(theta))


def invlogit(ltheta):
    return flib.invlogit(ravel(ltheta)).reshape(shape(ltheta))


def stukel_invlogit(ltheta, a1, a2):
    return flib.stukel_invlogit(ravel(ltheta), a1, a2).reshape(shape(ltheta))


def stukel_logit(theta, a1, a2):
    return flib.stukel_logit(ravel(theta), a1, a2).reshape(shape(theta))


def check_list(thing, label):
    if thing is not None:
        if thing.__class__ is not list:
            return [thing]
        return thing

# TODO: Look into using numpy.core.numerictypes to do this part.


# TODO : Wrap the nd histogramming fortran function.

def safe_len(val):
    if np.isscalar(val):
        return 1
    else:
        return np.prod(np.shape(val))


def round_array(array_in):
    """
    arr_out = round_array(array_in)

    Rounds an array and recasts it to int. Also works on scalars.
    """
    if isinstance(array_in, ndarray):
        return np.round(array_in).astype(int)
    else:
        return int(np.round(array_in))

try:
    from .flib import dchdc_wrap

    def msqrt(C):
        """
        U=incomplete_chol(C)

        Computes a Cholesky factorization of C. Works for matrices that are
        positive-semidefinite as well as positive-definite, though in these
        cases the Cholesky factorization isn't unique.

        U will be upper triangular.

        This is the dchdc version. It's faster for full-rank matrices,
        but it has to compute the entire matrix.

        """
        chol = C.copy()
        piv, N = dchdc_wrap(a=chol)
        if N < 0:
            raise ValueError(
                "Matrix does not appear to be positive semidefinite")
        return asmatrix(chol[:N, argsort(piv)])

except:
    def msqrt(cov):
        """
        sig = msqrt(cov)

        Return a matrix square root of a covariance matrix. Tries Cholesky
        factorization first, and factorizes by diagonalization if that fails.
        """
        # Try Cholesky factorization
        try:
            sig = asmatrix(cholesky(cov))

        # If there's a small eigenvalue, diagonalize
        except LinAlgError:
            val, vec = eigh(cov)
            sig = np.zeros(vec.shape)
            for i in range(len(val)):
                if val[i] < 0.:
                    val[i] = 0.
                sig[:, i] = vec[:, i] * sqrt(val[i])
        return np.asmatrix(sig).T


def histogram(a, bins=10, range=None, normed=False,
              weights=None, axis=None, strategy=None):
    """histogram(a, bins=10, range=None, normed=False, weights=None, axis=None)
                                                                   -> H, dict

    Return the distribution of sample.

    :Stochastics:
      `a` : Array sample.
      `bins` : Number of bins, or an array of bin edges, in which case the
              range is not used. If 'Scott' or 'Freeman' is passed, then
              the named method is used to find the optimal number of bins.
      `range` : Lower and upper bin edges, default: [min, max].
      `normed` :Boolean, if False, return the number of samples in each bin,
              if True, return the density.
      `weights` : Sample weights. The weights are normed only if normed is
              True. Should weights.sum() not equal len(a), the total bin count
              will not be equal to the number of samples.
      `axis` : Specifies the dimension along which the histogram is computed.
              Defaults to None, which aggregates the entire sample array.
      `strategy` : Histogramming method (binsize, searchsorted or digitize).

    :Return:
      `H` : The number of samples in each bin.
        If normed is True, H is a frequency distribution.
      dict{ 'edges':      The bin edges, including the rightmost edge.
        'upper':      Upper outliers.
        'lower':      Lower outliers.
        'bincenters': Center of bins.
        'strategy': the histogramming method employed.}

    :Examples:
      >>> x = random.rand(100,10)
      >>> H, D = histogram(x, bins=10, range=[0,1], normed=True)
      >>> H2, D = histogram(x, bins=10, range=[0,1], normed=True, axis=0)

    :SeeAlso: histogramnd
    """
    weighted = weights is not None

    a = asarray(a)
    if axis is None:
        a = atleast_1d(a.ravel())
        if weighted:
            weights = atleast_1d(weights.ravel())
        axis = 0

    # Define the range
    if range is None:
        mn, mx = a.min(), a.max()
        if mn == mx:
            mn = mn - .5
            mx = mx + .5
        range = [mn, mx]

    # Find the optimal number of bins.
    if bins is None or isinstance(bins, str):
        bins = _optimize_binning(a, range, bins)

    # Compute the bin edges if they are not given explicitely.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    # Hence, we shift the last bin by a tiny amount.
    if not iterable(bins):
        dr = diff(range) / bins * 1e-10
        edges = linspace(range[0], range[1] + dr, bins + 1, endpoint=True)
    else:
        edges = asarray(bins, float)

    dedges = diff(edges)
    bincenters = edges[:-1] + dedges / 2.

    # Number of bins
    nbin = len(edges) - 1

    # Measure of bin precision.
    decimal = int(-log10(dedges.min()) + 10)

    # Choose the fastest histogramming method
    even = (len(set(around(dedges, decimal))) == 1)
    if strategy is None:
        if even:
            strategy = 'binsize'
        else:
            if nbin > 30:  # approximative threshold
                strategy = 'searchsort'
            else:
                strategy = 'digitize'
    else:
        if strategy not in ['binsize', 'digitize', 'searchsort']:
            raise ValueError('Unknown histogramming strategy.', strategy)
        if strategy == 'binsize' and not even:
            raise ValueError(
                'This binsize strategy cannot be used for uneven bins.')

    # Stochastics for the fixed_binsize functions.
    start = float(edges[0])
    binwidth = float(dedges[0])

    # Looping to reduce memory usage
    block = 66600
    slices = [slice(None)] * a.ndim
    for i in arange(0, len(a), block):
        slices[axis] = slice(i, i + block)
        at = a[slices]
        if weighted:
            at = concatenate((at, weights[slices]), axis)
            if strategy == 'binsize':
                count = apply_along_axis(_splitinmiddle, axis, at,
                                         flib.weighted_fixed_binsize, start, binwidth, nbin)
            elif strategy == 'searchsort':
                count = apply_along_axis(_splitinmiddle, axis, at,
                                         _histogram_searchsort_weighted, edges)
            elif strategy == 'digitize':
                count = apply_along_axis(_splitinmiddle, axis, at,
                                         _histogram_digitize, edges, normed)
        else:
            if strategy == 'binsize':
                count = apply_along_axis(
                    flib.fixed_binsize,
                    axis,
                    at,
                    start,
                    binwidth,
                    nbin)
            elif strategy == 'searchsort':
                count = apply_along_axis(
                    _histogram_searchsort,
                    axis,
                    at,
                    edges)
            elif strategy == 'digitize':
                count = apply_along_axis(
                    _histogram_digitize, axis, at, None, edges,
                    normed)

        if i == 0:
            total = count
        else:
            total += count

    # Outlier count
    upper = total.take(array([-1]), axis)
    lower = total.take(array([0]), axis)

    # Non-outlier count
    core = a.ndim * [slice(None)]
    core[axis] = slice(1, -1)
    hist = total[core]

    if normed:
        normalize = lambda x: atleast_1d(x / (x * dedges).sum())
        hist = apply_along_axis(normalize, axis, hist)

    return hist, {'edges': edges, 'lower': lower, 'upper': upper,
                  'bincenters': bincenters, 'strategy': strategy}


def _histogram_fixed_binsize(a, start, width, n):
    """histogram_even(a, start, width, n) -> histogram

    Return an histogram where the first bin counts the number of lower
    outliers and the last bin the number of upper outliers. Works only with
    fixed width bins.

    :Stochastics:
      a : array
        Array of samples.
      start : float
        Left-most bin edge.
      width : float
        Width of the bins. All bins are considered to have the same width.
      n : int
        Number of bins.

    :Return:
      H : array
        Array containing the number of elements in each bin. H[0] is the number
        of samples smaller than start and H[-1] the number of samples
        greater than start + n*width.
    """

    return flib.fixed_binsize(a, start, width, n)


def _histogram_binsize_weighted(a, w, start, width, n):
    """histogram_even_weighted(a, start, width, n) -> histogram

    Return an histogram where the first bin counts the number of lower
    outliers and the last bin the number of upper outliers. Works only with
    fixed width bins.

    :Stochastics:
      a : array
        Array of samples.
      w : array
        Weights of samples.
      start : float
        Left-most bin edge.
      width : float
        Width of the bins. All bins are considered to have the same width.
      n : int
        Number of bins.

    :Return:
      H : array
        Array containing the number of elements in each bin. H[0] is the number
        of samples smaller than start and H[-1] the number of samples
        greater than start + n*width.
    """
    return flib.weighted_fixed_binsize(a, w, start, width, n)


def _histogram_searchsort(a, bins):
    n = sort(a).searchsorted(bins)
    n = concatenate([n, [len(a)]])
    count = concatenate([[n[0]], n[1:] - n[:-1]])
    return count


def _histogram_searchsort_weighted(a, w, bins):
    i = sort(a).searchsorted(bins)
    sw = w[argsort(a)]
    i = concatenate([i, [len(a)]])
    n = concatenate([[0], sw.cumsum()])[i]
    count = concatenate([[n[0]], n[1:] - n[:-1]])
    return count


def _splitinmiddle(x, function, *args, **kwds):
    x1, x2 = hsplit(x, 2)
    return function(x1, x2, *args, **kwds)


def _histogram_digitize(a, w, edges, normed):
    """Internal routine to compute the 1d weighted histogram for uneven bins.
    a: sample
    w: weights
    edges: bin edges
    weighted: Means that the weights are appended to array a.
    Return the bin count or frequency if normed.
    """
    weighted = w is not None
    nbin = edges.shape[0] + 1
    if weighted:
        count = zeros(nbin, dtype=w.dtype)
        if normed:
            count = zeros(nbin, dtype=float)
            w = w / w.mean()
    else:
        count = zeros(nbin, int)

    binindex = digitize(a, edges)

    # Count the number of identical indices.
    flatcount = bincount(binindex, w)

    # Place the count in the histogram array.
    count[:len(flatcount)] = flatcount

    return count


def _optimize_binning(x, range, method='Freedman'):
    """Find the optimal number of bins.
    Available methods : Freedman, Scott
    """
    N = x.shape[0]
    if method.lower() == 'freedman':
        s = sort(x)
        IQR = s[int(N * .75)] - s[int(
            N * .25)]  # Interquantile range (75% -25%)
        width = 2 * IQR * N ** (-1. / 3)

    elif method.lower() == 'scott':
        width = 3.49 * x.std() * N ** (-1. / 3)
    else:
        raise ValueError('Method must be Scott or Freedman', method)
    return int(diff(range) / width)


def normcdf(x, log=False):
    """Normal cumulative density function."""
    y = np.atleast_1d(x).copy()
    flib.normcdf(y)
    if log:
        return np.where(y > 0, np.log(y), -np.inf)
    return y


def lognormcdf(x, mu, tau):
    """Log-normal cumulative density function"""
    x = np.atleast_1d(x)
    return np.array(
        [0.5 * (1 - flib.derf(-(np.sqrt(tau / 2)) * (np.log(y) - mu))) for y in x])


def invcdf(x):
    """Inverse of normal cumulative density function."""
    x = np.atleast_1d(x)
    return np.array([flib.ppnd16(y, 1) for y in x])


def ar1_gen(rho, mu, sigma, size=1):
    """Create an autoregressive series of order one AR(1) generator.

    .. math::
        X_t = \mu_t + \rho (X_{t-1}-\mu_{t-1} + \epsilon_t

    If mu is a sequence and size > len(mu), the algorithm loops through
    mu.

    :Stochastics:
        rho : scalar in [0,1]
        mu : scalar or sequence
        sigma : scalar > 0
        size : integer
    """
    mu = np.asarray(mu, float)
    mu = np.resize(mu, size)
    r = mu.copy()
    r += np.random.randn(size) * sigma
    r[0] = np.random.randn(1) * sigma / np.sqrt(1 - rho ** 2)
    i = 0
    while True:
        yield r[i]
        i += 1
        if i == size:
            break
        r[i] += rho * (r[i - 1] - mu[i - 1])


def ar1(rho, mu, sigma, size=1):
    """Return an autoregressive series of order one AR(1).

    .. math::
        X_t = \mu_t + \rho (X_{t-1}-\mu_{t-1} + \epsilon_t

    If mu is a sequence and size > len(mu), the algorithm loops through
    mu.

    :Stochastics:
        rho : scalar in [0,1]
        mu : scalar or sequence
        sigma : scalar > 0
        size : integer
    """
    return np.array([x for x in ar1_gen(rho, mu, sigma, size)])


def autocorr(x, lag=1):
    """Sample autocorrelation at specified lag.
    The autocorrelation is the correlation of x_i with x_{i+lag}.
    """

    if not lag:
        return 1
    if lag < 0:
        return
    # x = np.squeeze(asarray(x))
    #     mu = x.mean()
    #     v = x.var()
    #     return ((x[:-lag]-mu)*(x[lag:]-mu)).sum()/v/(len(x) - lag)
    S = autocov(x, lag)
    return S[0, 1] / sqrt(prod(diag(S)))


def autocov(x, lag=1):
    """
    Sample autocovariance at specified lag.
    The autocovariance is a 2x2 matrix with the variances of
    x[:-lag] and x[lag:] in the diagonal and the autocovariance
    on the off-diagonal.
    """

    if not lag:
        return 1
    if lag < 0:
        return
    return cov(x[:-lag], x[lag:], bias=1)


def trace_generator(trace, start=0, stop=None, step=1):
    """Return a generator returning values from the object's trace.

    Ex:
    T = trace_generator(theta.trace)
    T.next()
    for t in T:...
    """
    i = start
    stop = stop or np.inf
    size = min(trace.length(), stop)
    while i < size:
        index = slice(i, i + 1)
        yield trace.gettrace(slicing=index)[0]
        i += step


def draw_random(obj, **kwds):
    """Draw random variates from obj.random method.

    If the object has parents whose value must be updated, use
    parent_name=trace_generator_function.

    Ex:
    R = draw_random(theta, beta=pymc.utils.trace_generator(beta.trace))
    R.next()
    """
    while True:
        for k, v in six.iteritems(kwds):
            obj.parents[k] = v.next()
        yield obj.random()


def rec_getattr(obj, attr):
    """Get object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_getattr(a, 'b.c')
    4
    """
    return reduce(getattr, attr.split('.'), obj)


def rec_setattr(obj, attr, value):
    """Set object's attribute. May use dot notation.

    >>> class C(object): pass
    >>> a = C()
    >>> a.b = C()
    >>> a.b.c = 4
    >>> rec_setattr(a, 'b.c', 2)
    >>> a.b.c
    2
    """
    attrs = attr.split('.')
    setattr(reduce(getattr, attrs[:-1], obj), attrs[-1], value)


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
    if x.ndim > 1:

        # Transpose first, then sort
        tx = tr(x, list(range(x.ndim)[1:]) + [0])
        dims = shape(tx)

        # Container list for intervals
        intervals = np.resize(0.0, (2,) + dims[:-1])

        for index in make_indices(dims[:-1]):

            try:
                index = tuple(index)
            except TypeError:
                pass

            # Sort trace
            sx = sort(tx[index])

            # Append to list
            intervals[0][index], intervals[1][index] = calc_min_interval(sx, alpha)
        
        # Transpose back before returning
        return array(intervals)

    else:
        # Sort univariate node
        sx = sort(x)

        return array(calc_min_interval(sx, alpha))


def make_indices(dimensions):
    # Generates complete set of indices for given dimensions

    level = len(dimensions)

    if level == 1:
        return range(dimensions[0])

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
        print_('Too few elements for interval calculation')
        return [None, None]

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx + interval_idx_inc]
    return [hdi_min, hdi_max]


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
        sx = sort(x.T).T
    else:
        # Sort univariate node
        sx = sort(x)

    try:
        # Generate specified quantiles
        quants = [sx[int(len(sx) * q / 100.0)] for q in qlist]

        return dict(zip(qlist, quants))

    except IndexError:
        print_("Too few elements for quantile calculation")


def coda_output(pymc_object, name=None, chain=-1):
    """Generate output files that are compatible with CODA

    :Arguments:
      pymc_object : Model or Node
          A PyMC object containing MCMC output.

    """

    print_()
    print_("Generating CODA output")
    print_('=' * 50)

    if name is None:
        name = pymc_object.__name__

    # Open trace file
    trace_file = open(name + '_coda.out', 'w')

    # Open index file
    index_file = open(name + '_coda.ind', 'w')

    variables = [pymc_object]
    if hasattr(pymc_object, 'stochastics'):
        variables = pymc_object.stochastics

    # Initialize index
    index = 1

    # Loop over all parameters
    for v in variables:

        vname = v.__name__
        print_("Processing", vname)
        try:
            index = _process_trace(
                trace_file,
                index_file,
                v.trace(chain=chain),
                vname,
                index)
        except TypeError:
            pass

    # Close files
    trace_file.close()
    index_file.close()

# Lazy shortcut
coda = coda_output


def _process_trace(trace_file, index_file, trace, name, index):
    """Support function for coda_output(); writes output to files"""

    if ndim(trace) > 1:
        trace = swapaxes(trace, 0, 1)
        for i, seq in enumerate(trace):
            _name = '%s_%s' % (name, i)
            index = _process_trace(trace_file, index_file, seq, _name, index)
    else:
        index_buffer = '%s\t%s\t' % (name, index)
        for i, val in enumerate(trace):
            trace_file.write('%s\t%s\r\n' % (i + 1, val))
            index += 1
        index_file.write('%s%s\r\n' % (index_buffer, index - 1))

    return index


def log_difference(lx, ly):
    """Returns log(exp(lx) - exp(ly)) without leaving log space."""

    # Negative log of double-precision infinity
    li = -709.78271289338397
    diff = ly - lx
    # Make sure log-difference can succeed
    if np.any(diff >= 0):
        raise ValueError(
            'Cannot compute log(x-y), because y>=x for some elements.')
    # Otherwise evaluate log-difference
    return lx + np.log(1. - np.exp(diff))


def getInput():
    """Read the input buffer without blocking the system."""
    input = ''

    if sys.platform == 'win32':
        import msvcrt
        if msvcrt.kbhit():  # Check for a keyboard hit.
            input += msvcrt.getch()
            print_(input)
        else:
            time.sleep(.1)

    else:  # Other platforms
        # Posix will work with sys.stdin or sys.stdin.fileno()
        # Mac needs the file descriptor.
        # This solution does not work for windows since select
        # expects a socket, and I have no idea how to create a
        # socket from standard input.
        sock = sys.stdin.fileno()

        # select(rlist, wlist, xlist, timeout)
        while len(select.select([sock], [], [], 0.1)[0]) > 0:
            input += os.read(sock, 4096)

    return input


def crawl_dataless(sofar, gens):
    """
    Crawls out from v to find the biggest dataless submodel containing v.
    TODO: Let MCMC start the crawl from its last generation. It doesn't
    matter that there won't be one contiguous group.
    """
    new_gen = set([])
    all_ext_parents = set()
    for s in gens[-1]:
        all_ext_parents.update(s.extended_parents)

    for p in all_ext_parents:
        if p._random is not None and not p.observed:
            if len(p.extended_children - sofar) == 0:
                new_gen.add(p)
    if len(new_gen) == 0:
        return sofar, gens
    else:
        sofar |= new_gen
        gens.append(new_gen)
        return crawl_dataless(sofar, gens)


def find_generations(container, with_data=False):
    """
    A generation is the set of stochastic variables that only has parents in
    previous generations.
    """

    generations = []

    # Find root generation
    generations.append(set())
    all_children = set()
    if with_data:
        stochastics_to_iterate = container.stochastics | container.observed_stochastics
    else:
        stochastics_to_iterate = container.stochastics
    for s in stochastics_to_iterate:
        all_children.update(s.extended_children & stochastics_to_iterate)
    generations[0] = stochastics_to_iterate - all_children

    # Find subsequent _generations
    children_remaining = True
    gen_num = 0
    while children_remaining:
        gen_num += 1

        # Find children of last generation
        generations.append(set())
        for s in generations[gen_num - 1]:
            generations[gen_num].update(
                s.extended_children & stochastics_to_iterate)

        # Take away stochastics that have parents in the current generation.
        thisgen_children = set()
        for s in generations[gen_num]:
            thisgen_children.update(
                s.extended_children & stochastics_to_iterate)
        generations[gen_num] -= thisgen_children

        # Stop when no subsequent _generations remain
        if len(thisgen_children) == 0:
            children_remaining = False
    return generations


def append(nodelist, node, label=None, sep='_'):
    """
    Append function to automate the naming of list elements in Containers.

    :Arguments:
        - `nodelist` : List containing nodes for Container.
        - `node` : Node to be added to list.
        - `label` : Label to be appended to list (If not passed,
        defaults to element number).
        - `sep` : Separator character for label (defaults to underscore).

    :Return:
        - `nodelist` : Passed list with node added.

    """

    nname = node.__name__

    # Determine label
    label = label or len(nodelist)

    # Look for separator at the end of name
    ind = nname.rfind(sep)

    # If there is no separator, we will remove last character and
    # replace with label.
    node.__name__ = nname[:ind] + sep + str(label)

    nodelist.append(node)

    return nodelist


# deterministic related utilities
def find_element(names, modules, error_on_fail):
    element = None
    found = False

    if isinstance(names, str):
        names = [names]

    if isinstance(modules, dict) or isinstance(modules, types.ModuleType):
        modules = [modules]

    for module in modules:

        if isinstance(module, types.ModuleType):
            module = copy(module.__dict__)
        elif isinstance(module, dict):
            module = copy(module)
        else:
            raise AttributeError

        for name in names:
            try:
                function = module[name]
                found = True
            except KeyError:
                pass

    if not found and error_on_fail:
        raise NameError(
            "no function or variable " + str(
                names) + " in " + str(
                modules))

    return function
