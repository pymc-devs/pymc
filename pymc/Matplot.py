"""
Plotting module using matplotlib.
"""

from __future__ import division

# Import matplotlib functions
try:
    import matplotlib.gridspec as gridspec
except ImportError:
    gridspec = None
from . import Variable
import os
from pylab import hist, plot as pyplot, xlabel, ylabel, xlim, ylim, savefig, acorr, mlab
from pylab import figure, subplot, subplots_adjust, gca, scatter, axvline, yticks, xticks
from pylab import setp, contourf, cm, title, colorbar, fill, text
from pylab import errorbar

# Import numpy functions
from numpy import arange, ravel, rank, swapaxes, concatenate, asarray, ndim
from numpy import mean, std, sort, prod, floor, shape, size, transpose
from numpy import min as nmin, max as nmax, abs, log2, log, sqrt, isnan
from numpy import append, ones, dtype, indices, array, unique, zeros
from .utils import quantiles as calc_quantiles, hpd as calc_hpd
try:
    from scipy import special
except ImportError:
    special = None
from scipy.stats import kurtosis

from . import six
from .six import print_

__all__ = ['func_quantiles', 'func_envelopes', 'func_sd_envelope',
           'centered_envelope', 'get_index_list', 'plot', 'histogram', 'trace',
           'geweke_plot', 'gof_plot', 'pair_posterior', 'summary_plot']


def get_index_list(shape, j):
    """
    index_list = get_index_list(shape, j)

    :Arguments:
        shape: a tuple
        j: an integer

    Assumes index j is from a ravelled version of an array
    with specified shape, returns the corresponding
    non-ravelled index tuple as a list.
    """

    r = range(len(shape))
    index_list = (r)

    for i in r:
        if i < len(shape):
            prodshape = prod(shape[i + 1:])
        else:
            prodshape = 0
        index_list[i] = int(floor(j / prodshape))
        if index_list[i] > shape[i]:
            raise IndexError('Requested index too large')
        j %= prodshape

    return index_list


def func_quantiles(node, qlist=(.025, .25, .5, .75, .975)):
    """
    Returns an array whose ith row is the q[i]th quantile of the
    function.

    :Arguments:
        func_stacks: The samples of the function. func_stacks[i,:]
            gives sample i.
        qlist: A list or array of the quantiles you would like.

    :SeeAlso: func_envelopes, func_hist, weightplot
    """

    # For very large objects, this will be rather long.
    # Too get the length of the table, use obj.trace.length()

    if isinstance(node, Variable):
        func_stacks = node.trace()
    else:
        func_stacks = node

    if any(qlist < 0.) or any(qlist > 1.):
        raise TypeError('The elements of qlist must be between 0 and 1')

    func_stacks = func_stacks.copy()

    N_samp = shape(func_stacks)[0]
    func_len = tuple(shape(func_stacks)[1:])

    func_stacks.sort(axis=0)

    quants = zeros((len(qlist), func_len), dtype=float)
    alphas = 1. - abs(array(qlist) - .5) / .5

    for i in range(len(qlist)):
        quants[i, ] = func_stacks[int(qlist[i] * N_samp), ]

    return quants, alphas


def func_envelopes(node, CI=(.25, .5, .95)):
    """
    func_envelopes(node, CI = (.25, .5, .95))

    Returns a list of centered_envelope objects for func_stacks,
    each one corresponding to an element of CI, and one
    corresponding to mass 0 (the median).

    :Arguments:
        func_stacks: The samples of the function. func_stacks[i,:]
            gives sample i.
        CI: A list or array containing the probability masses
            the envelopes should enclose.

    :Note: The return list of envelopes is sorted from high to low
        enclosing probability masses, so they should be plotted in
        order.

    :SeeAlso: centered_envelope, func_quantiles, func_hist, weightplot
    """

    if isinstance(node, Variable):
        func_stacks = asarray(node.trace())
    else:
        func_stacks = node

    func_stacks = func_stacks.copy()
    func_stacks.sort(axis=0)

    envelopes = []
    qsort = sort(CI)

    for i in range(len(qsort)):
        envelopes.append(
            centered_envelope(func_stacks,
                              qsort[len(qsort) - i - 1]))
    envelopes.append(centered_envelope(func_stacks, 0.))

    return envelopes

# FIXME: Not sure of the best way to bring these two into PlotFactory...


class func_sd_envelope(object):

    """
    F = func_sd_envelope(func_stacks)
    F.display(axes,xlab=None,ylab=None,name=None)

    This object plots the mean and +/- 1 sd error bars for
    the one or two-dimensional function whose trace
    """

    def __init__(self, node, format='pdf', plotpath='', suffix=''):

        if isinstance(node, Variable):
            func_stacks = node.trace()
        else:
            func_stacks = node
        self.name = node.__name__
        self._format = format
        self._plotpath = plotpath
        self.suffix = suffix

        self.mean = mean(func_stacks, axis=0)
        self.std = std(func_stacks, axis=0)

        self.lo = self.mean - self.std
        self.hi = self.mean + self.std

        self.ndim = len(shape(func_stacks)) - 1

    def display(self, axes, xlab=None, ylab=None, name=None, new=True):
        if name:
            name_str = name
        else:
            name_str = ''

        if self.ndim == 1:
            if new:
                figure()
            pyplot(axes, self.lo, 'k-.', label=name_str + ' mean-sd')
            pyplot(axes, self.hi, 'k-.', label=name_str + 'mean+sd')
            pyplot(axes, self.mean, 'k-', label=name_str + 'mean')
            if name:
                title(name)

        elif self.ndim == 2:
            if new:
                figure(figsize=(14, 4))
            subplot(1, 3, 1)
            contourf(axes[0], axes[1], self.lo, cmap=cm.bone)
            title(name_str + ' mean-sd')
            if xlab:
                xlabel(xlab)
            if ylab:
                ylabel(ylab)
            colorbar()

            subplot(1, 3, 2)
            contourf(axes[0], axes[1], self.mean, cmap=cm.bone)
            title(name_str + ' mean')
            if xlab:
                xlabel(xlab)
            if ylab:
                ylabel(ylab)
            colorbar()

            subplot(1, 3, 3)
            contourf(axes[0], axes[1], self.hi, cmap=cm.bone)
            title(name_str + ' mean+sd')
            if xlab:
                xlabel(xlab)
            if ylab:
                ylabel(ylab)
            colorbar()
        else:
            raise ValueError(
                'Only 1- and 2- dimensional functions can be displayed')
        savefig(
            "%s%s%s.%s" % (
                self._plotpath,
                self.name,
                self.suffix,
                self._format))


class centered_envelope(object):

    """
    E = centered_envelope(sorted_func_stack, mass)

    An object corresponding to the centered CI envelope
    of a function enclosing a particular probability mass.

    :Arguments:
        sorted_func_stack: The samples of the function, sorted.
            if func_stacks[i,:] gives sample i, then
            sorted_func_stack is sort(func_stacks,0).

        mass: The probability mass enclosed by the CI envelope.

    :SeeAlso: func_envelopes
    """
    def __init__(self, sorted_func_stack, mass):
        if mass < 0 or mass > 1:
            raise ValueError('mass must be between 0 and 1')
        N_samp = shape(sorted_func_stack)[0]
        self.mass = mass
        self.ndim = len(sorted_func_stack.shape) - 1

        if self.mass == 0:
            self.value = sorted_func_stack[int(N_samp * .5), ]
        else:
            quandiff = .5 * (1. - self.mass)
            self.lo = sorted_func_stack[int(N_samp * quandiff), ]
            self.hi = sorted_func_stack[int(N_samp * (1. - quandiff)), ]

    def display(self, xaxis, alpha, new=True):
        """
        E.display(xaxis, alpha = .8)

        :Arguments: xaxis, alpha

        Plots the CI region on the current figure, with respect to
        xaxis, at opacity alpha.

        :Note: The fill color of the envelope will be self.mass
            on the grayscale.
        """
        if new:
            figure()
        if self.ndim == 1:
            if self.mass > 0.:
                x = concatenate((xaxis, xaxis[::-1]))
                y = concatenate((self.lo, self.hi[::-1]))
                fill(
                    x,
                    y,
                    facecolor='%f' % self.mass,
                    alpha=alpha,
                    label=(
                        'centered CI ' + str(
                            self.mass)))
            else:
                pyplot(
                    xaxis,
                    self.value,
                    'k-',
                    alpha=alpha,
                    label=(
                        'median'))
        else:
            if self.mass > 0.:
                subplot(1, 2, 1)
                contourf(xaxis[0], xaxis[1], self.lo, cmap=cm.bone)
                colorbar()
                subplot(1, 2, 2)
                contourf(xaxis[0], xaxis[1], self.hi, cmap=cm.bone)
                colorbar()
            else:
                contourf(xaxis[0], xaxis[1], self.value, cmap=cm.bone)
                colorbar()


def plotwrapper(f):
    """
    This decorator allows for PyMC arguments of various types to be passed to
    the plotting functions. It identifies the type of object and locates its
    trace(s), then passes the data to the wrapped plotting function.

    """

    def wrapper(pymc_obj, *args, **kwargs):
        start = 0
        if 'start' in kwargs:
            start = kwargs.pop('start')

        # Figure out what type of object it is
        try:
            # First try Model type
            for variable in pymc_obj._variables_to_tally:
                # Plot object
                if variable._plot is not False:
                    data = pymc_obj.trace(variable.__name__)[start:]
                    if size(data[-1]) >= 10 and variable._plot != True:
                        continue
                    elif variable.dtype is dtype('object'):
                        continue
                    name = variable.__name__
                    if args:
                        name = '%s_%s' % (args[0], variable.__name__)
                    f(data, name, *args, **kwargs)
            return
        except AttributeError:
            pass

        try:
            # Then try Trace type
            data = pymc_obj()[:]
            name = pymc_obj.name
            f(data, name, *args, **kwargs)
            return
        except (AttributeError, TypeError):
            pass

        try:
            # Then try Node type
            if pymc_obj._plot is not False:
                data = pymc_obj.trace()[start:]  # This is deprecated. DH
                name = pymc_obj.__name__
                f(data, name, *args, **kwargs)
            return
        except AttributeError:
            pass

        if isinstance(pymc_obj, dict):
            # Then try dictionary
            for i in pymc_obj:
                data = pymc_obj[i][start:]
                if args:
                    i = '%s_%s' % (args[0], i)
                elif 'name' in kwargs:
                    i = '%s_%s' % (kwargs.pop('name'), i)
                f(data, i, *args, **kwargs)
            return

        # If others fail, assume that raw data is passed
        f(pymc_obj, *args, **kwargs)

    wrapper.__doc__ = f.__doc__
    wrapper.__name__ = f.__name__
    return wrapper


@plotwrapper
def plot(
    data, name, format='png', suffix='', path='./', common_scale=True, datarange=(None, None),
        new=True, last=True, rows=1, num=1, fontmap=None, verbose=1):
    """
    Generates summary plots for nodes of a given PyMC object.

    :Arguments:
        data: PyMC object, trace or array
            A trace from an MCMC sample or a PyMC object with one or more traces.

        name: string
            The name of the object.

        format (optional): string
            Graphic output format (defaults to png).

        suffix (optional): string
            Filename suffix.

        path (optional): string
            Specifies location for saving plots (defaults to local directory).

        common_scale (optional): bool
            Specifies whether plots of multivariate nodes should be on the same scale
            (defaults to True).

    """
    if fontmap is None:
        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

    # If there is only one data array, go ahead and plot it ...
    if rank(data) == 1:

        if verbose > 0:
            print_('Plotting', name)

        # If new plot, generate new frame
        if new:

            figure(figsize=(10, 6))

        # Call trace
        trace(
            data,
            name,
            datarange=datarange,
            rows=rows * 2,
            columns=2,
            num=num + 3 * (num - 1),
            last=last,
            fontmap=fontmap)
        # Call autocorrelation
        autocorrelation(
            data,
            name,
            rows=rows * 2,
            columns=2,
            num=num + 3 * (
                num - 1) + 2,
            last=last,
            fontmap=fontmap)
        # Call histogram
        histogram(
            data,
            name,
            datarange=datarange,
            rows=rows,
            columns=2,
            num=num * 2,
            last=last,
            fontmap=fontmap)

        if last:
            if not os.path.exists(path):
                os.mkdir(path)
            if not path.endswith('/'):
                path += '/'
            savefig("%s%s%s.%s" % (path, name, suffix, format))

    else:
        # ... otherwise plot recursively
        tdata = swapaxes(data, 0, 1)

        datarange = (None, None)
        # Determine common range for plots
        if common_scale:
            datarange = (nmin(tdata), nmax(tdata))

        # How many rows?
        _rows = min(4, len(tdata))

        for i in range(len(tdata)):

            # New plot or adding to existing?
            _new = not i % _rows
            # Current subplot number
            _num = i % _rows + 1
            # Final subplot of current figure?
            _last = (_num == _rows) or (i == len(tdata) - 1)

            plot(
                tdata[i],
                name + '_' + str(i),
                format=format,
                path=path,
                common_scale=common_scale,
                datarange=datarange,
                suffix=suffix,
                new=_new,
                last=_last,
                rows=_rows,
                num=_num)

_sturges = lambda n: log2(n) + 1

_doanes = lambda data, n: 1 + log(n) + log(
    1 + kurtosis(data) * (n / 6.) ** 0.5)

_sqrt_choice = lambda n: sqrt(n)


@plotwrapper
def histogram(
    data, name, bins='sturges', datarange=(None, None), format='png', suffix='', path='./', rows=1,
        columns=1, num=1, last=True, fontmap = None, verbose=1):
    """
    Generates histogram from an array of data.

    :Arguments:
        data: array or list
            Usually a trace from an MCMC sample.

        name: string
            The name of the histogram.

        bins: int or string
            The number of bins, or a preferred binning method. Available methods include
            'doanes', 'sturges' and 'sqrt' (defaults to 'doanes').

        datarange: tuple or list
            Preferred range of histogram (defaults to (None,None)).

        format (optional): string
            Graphic output format (defaults to png).

        suffix (optional): string
            Filename suffix.

        path (optional): string
            Specifies location for saving plots (defaults to local directory).

        fontmap (optional): dict
            Font map for plot.
    """

    # Internal histogram specification for handling nested arrays
    try:

        if fontmap is None:
            fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

        # Stand-alone plot or subplot?
        standalone = rows == 1 and columns == 1 and num == 1
        if standalone:
            if verbose > 0:
                print_('Generating histogram of', name)
            figure()

        subplot(rows, columns, num)

        # Specify number of bins
        uniquevals = len(unique(data))
        if isinstance(bins, int):
            pass
        if bins == 'sturges':
            bins = uniquevals * (uniquevals <= 25) or _sturges(len(data))
        elif bins == 'doanes':
            bins = uniquevals * (uniquevals <= 25) or _doanes(data, len(data))
        elif bins == 'sqrt':
            bins = uniquevals * (uniquevals <= 25) or _sqrt_choice(len(data))
        elif isinstance(bins, int):
            bins = bins
        else:
            raise ValueError('Invalid bins argument in histogram')

        if isnan(bins):
            bins = uniquevals * (uniquevals <= 25) or int(
                4 + 1.5 * log(len(data)))
            print_(
                'Bins could not be calculated using selected method. Setting bins to %i.' %
                bins)

        # Generate histogram
        hist(data.tolist(), bins, histtype='stepfilled')

        xlim(datarange)

        # Plot options
        title(
            '\n\n   %s hist' %
            name,
            x=0.,
            y=1.,
            ha='left',
            va='top',
            fontsize='medium')

        ylabel("Frequency", fontsize='x-small')

        # Plot vertical lines for median and 95% HPD interval
        quant = calc_quantiles(data)
        axvline(x=quant[50], linewidth=2, color='black')
        for q in calc_hpd(data, 0.05):
            axvline(x=q, linewidth=2, color='grey', linestyle='dotted')

        # Smaller tick labels
        tlabels = gca().get_xticklabels()
        setp(tlabels, 'fontsize', fontmap[rows])
        tlabels = gca().get_yticklabels()
        setp(tlabels, 'fontsize', fontmap[rows])

        if standalone:
            if not os.path.exists(path):
                os.mkdir(path)
            if not path.endswith('/'):
                path += '/'
            # Save to file
            savefig("%s%s%s.%s" % (path, name, suffix, format))
            # close()

    except OverflowError:
        print_('... cannot generate histogram')


@plotwrapper
def trace(
    data, name, format='png', datarange=(None, None), suffix='', path='./', rows=1, columns=1,
        num=1, last=True, fontmap = None, verbose=1):
    """
    Generates trace plot from an array of data.

    :Arguments:
        data: array or list
            Usually a trace from an MCMC sample.

        name: string
            The name of the trace.

        datarange: tuple or list
            Preferred y-range of trace (defaults to (None,None)).

        format (optional): string
            Graphic output format (defaults to png).

        suffix (optional): string
            Filename suffix.

        path (optional): string
            Specifies location for saving plots (defaults to local directory).

        fontmap (optional): dict
            Font map for plot.

    """

    if fontmap is None:
        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

    # Stand-alone plot or subplot?
    standalone = rows == 1 and columns == 1 and num == 1

    if standalone:
        if verbose > 0:
            print_('Plotting', name)
        figure()

    subplot(rows, columns, num)
    pyplot(data.tolist())
    ylim(datarange)

    # Plot options
    title(
        '\n\n   %s trace' %
        name,
        x=0.,
        y=1.,
        ha='left',
        va='top',
        fontsize='small')

    # Smaller tick labels
    tlabels = gca().get_xticklabels()
    setp(tlabels, 'fontsize', fontmap[max(rows / 2, 1)])

    tlabels = gca().get_yticklabels()
    setp(tlabels, 'fontsize', fontmap[max(rows / 2, 1)])

    if standalone:
        if not os.path.exists(path):
            os.mkdir(path)
        if not path.endswith('/'):
            path += '/'
        # Save to file
        savefig("%s%s%s.%s" % (path, name, suffix, format))
        # close()


@plotwrapper
def geweke_plot(
    data, name, format='png', suffix='-diagnostic', path='./', fontmap=None,
        verbose=1):
    # Generate Geweke (1992) diagnostic plots

    if fontmap is None:
        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

    # Generate new scatter plot
    figure()
    x, y = transpose(data)
    scatter(x.tolist(), y.tolist())

    # Plot options
    xlabel('First iteration', fontsize='x-small')
    ylabel('Z-score for %s' % name, fontsize='x-small')

    # Plot lines at +/- 2 sd from zero
    pyplot((nmin(x), nmax(x)), (2, 2), '--')
    pyplot((nmin(x), nmax(x)), (-2, -2), '--')

    # Set plot bound
    ylim(min(-2.5, nmin(y)), max(2.5, nmax(y)))
    xlim(0, nmax(x))

    # Save to file
    if not os.path.exists(path):
        os.mkdir(path)
    if not path.endswith('/'):
        path += '/'
    savefig("%s%s%s.%s" % (path, name, suffix, format))
    # close()


@plotwrapper
def discrepancy_plot(
    data, name='discrepancy', report_p=True, format='png', suffix='-gof', path='./',
        fontmap=None, verbose=1):
    # Generate goodness-of-fit deviate scatter plot

    if verbose > 0:
        print_('Plotting', name + suffix)

    if fontmap is None:
        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

    # Generate new scatter plot
    figure()
    try:
        x, y = transpose(data)
    except ValueError:
        x, y = data
    scatter(x, y)

    # Plot x=y line
    lo = nmin(ravel(data))
    hi = nmax(ravel(data))
    datarange = hi - lo
    lo -= 0.1 * datarange
    hi += 0.1 * datarange
    pyplot((lo, hi), (lo, hi))

    # Plot options
    xlabel('Observed deviates', fontsize='x-small')
    ylabel('Simulated deviates', fontsize='x-small')

    if report_p:
        # Put p-value in legend
        count = sum(s > o for o, s in zip(x, y))
        text(lo + 0.1 * datarange, hi - 0.1 * datarange,
             'p=%.3f' % (count / len(x)), horizontalalignment='center',
             fontsize=10)

    # Save to file
    if not os.path.exists(path):
        os.mkdir(path)
    if not path.endswith('/'):
        path += '/'
    savefig("%s%s%s.%s" % (path, name, suffix, format))
    # close()


def gof_plot(
    simdata, trueval, name=None, bins=None, format='png', suffix='-gof', path='./',
        fontmap=None, verbose=1):
    """
    Plots histogram of replicated data, indicating the location of the observed data

    :Arguments:
        simdata: array or PyMC object
            Trace of simulated data or the PyMC stochastic object containing trace.

        trueval: numeric
            True (observed) value of the data

        bins: int or string
            The number of bins, or a preferred binning method. Available methods include
            'doanes', 'sturges' and 'sqrt' (defaults to 'doanes').

        format (optional): string
            Graphic output format (defaults to png).

        suffix (optional): string
            Filename suffix.

        path (optional): string
            Specifies location for saving plots (defaults to local directory).

        fontmap (optional): dict
            Font map for plot.

    """

    if fontmap is None:
        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}
    try:
        if ndim(simdata) == 1:
            simdata = simdata.trace()
    except ValueError:
        pass

    if ndim(trueval) == 1 and ndim(simdata == 2):
        # Iterate over more than one set of data
        for i in range(len(trueval)):
            n = name or 'MCMC'
            gof_plot(
                simdata[
                    :,
                    i],
                trueval[
                    i],
                '%s[%i]' % (
                n,
                    i),
                bins=bins,
                format=format,
                suffix=suffix,
                path=path,
                fontmap=fontmap,
                verbose=verbose)
        return

    if verbose > 0:
        print_('Plotting', (name or 'MCMC') + suffix)

    figure()

    # Specify number of bins
    if bins is None:
        bins = 'sqrt'
    uniquevals = len(unique(simdata))
    if bins == 'sturges':
        bins = uniquevals * (uniquevals <= 25) or _sturges(len(simdata))
    elif bins == 'doanes':
        bins = uniquevals * (
            uniquevals <= 25) or _doanes(simdata,
                                         len(simdata))
    elif bins == 'sqrt':
        bins = uniquevals * (uniquevals <= 25) or _sqrt_choice(len(simdata))
    elif isinstance(bins, int):
        bins = bins
    else:
        raise ValueError('Invalid bins argument in gof_plot')

    # Generate histogram
    hist(simdata, bins)

    # Plot options
    xlabel(name or 'Value', fontsize='x-small')

    ylabel("Frequency", fontsize='x-small')

    # Smaller tick labels
    tlabels = gca().get_xticklabels()
    setp(tlabels, 'fontsize', fontmap[1])
    tlabels = gca().get_yticklabels()
    setp(tlabels, 'fontsize', fontmap[1])

    # Plot vertical line at location of true data value
    axvline(x=trueval, linewidth=2, color='r', linestyle='dotted')

    if not os.path.exists(path):
        os.mkdir(path)
    if not path.endswith('/'):
        path += '/'
    # Save to file
    savefig("%s%s%s.%s" % (path, name or 'MCMC', suffix, format))
    # close()


@plotwrapper
def autocorrelation(
    data, name, maxlags=100, format='png', reflected=False, suffix='-acf', path='./',
        fontmap=None, new=True, last=True, rows=1, columns=1, num=1, verbose=1):
    """
    Generate bar plot of the autocorrelation function for a series (usually an MCMC trace).

    :Arguments:
        data: PyMC object, trace or array
            A trace from an MCMC sample or a PyMC object with one or more traces.

        name: string
            The name of the object.

        maxlags (optional): int
            The largest discrete value for the autocorrelation to be calculated (defaults to 100).

        format (optional): string
            Graphic output format (defaults to png).

        suffix (optional): string
            Filename suffix.

        path (optional): string
            Specifies location for saving plots (defaults to local directory).

        fontmap (optional): dict
            Font mapping for plot labels; most users should not specify this.

        verbose (optional): int
            Level of output verbosity.

    """
    # Internal plotting specification for handling nested arrays

    if fontmap is None:
        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

    # Stand-alone plot or subplot?
    standalone = rows == 1 and columns == 1 and num == 1

    if standalone:
        if verbose > 0:
            print_('Plotting', name)
        figure()

    subplot(rows, columns, num)
    if ndim(data) == 1:
        maxlags = min(len(data) - 1, maxlags)
        try:
            acorr(data, detrend=mlab.detrend_mean, maxlags=maxlags)
        except:
            print_('Cannot plot autocorrelation for %s' % name)
            return

        # Set axis bounds
        ylim(-.1, 1.1)
        xlim(-maxlags * reflected or 0, maxlags)

        # Plot options
        title(
            '\n\n   %s acorr' %
            name,
            x=0.,
            y=1.,
            ha='left',
            va='top',
            fontsize='small')

        # Smaller tick labels
        tlabels = gca().get_xticklabels()
        setp(tlabels, 'fontsize', fontmap[1])

        tlabels = gca().get_yticklabels()
        setp(tlabels, 'fontsize', fontmap[1])
    elif ndim(data) == 2:
        # generate acorr plot for each dimension
        rows = data.shape[1]
        for j in range(rows):
            autocorrelation(
                data[:,
                     j],
                '%s_%d' % (name,
                           j),
                maxlags,
                fontmap=fontmap,
                rows=rows,
                columns=1,
                num=j + 1)
    else:
        raise ValueError(
            'Only 1- and 2- dimensional functions can be displayed')

    if standalone:
        if not os.path.exists(path):
            os.mkdir(path)
        if not path.endswith('/'):
            path += '/'
        # Save to fiel
        savefig("%s%s%s.%s" % (path, name, suffix, format))
        # close()


def zplot(pvalue_dict, name='',
          format='png', path='./', fontmap=None, verbose=1):
    """Plots absolute values of z-scores for model validation output from
    diagnostics.validate()."""

    if verbose:
        print_('\nGenerating model validation plot')

    if fontmap is None:
        fontmap = {1: 10, 2: 8, 3: 6, 4: 5, 5: 4}

    x, y, labels = [], [], []

    for i, var in enumerate(pvalue_dict):

        # Get p-values
        pvals = pvalue_dict[var]
        # Take absolute values of inverse-standard normals
        zvals = abs(special.ndtri(pvals))

        x = append(x, zvals)
        y = append(y, ones(size(zvals)) * (i + 1))

        vname = var
        vname += " (%i)" % size(zvals)
        labels = append(labels, vname)

    # Spawn new figure
    figure()
    subplot(111)
    subplots_adjust(left=0.25, bottom=0.1)
    # Plot scores
    pyplot(x, y, 'o')
    # Set range on axes
    ylim(0, size(pvalue_dict) + 2)
    xlim(xmin=0)
    # Tick labels for y-axis
    yticks(arange(len(labels) + 2), append(append("", labels), ""))
    # X label
    xlabel("Absolute z transformation of p-values")

    if not os.path.exists(path):
        os.mkdir(path)
    if not path.endswith('/'):
        path += '/'

    if name:
        name += '-'

    savefig("%s%svalidation.%s" % (path, name, format))


def var_str(name, shape):
    """Return a sequence of strings naming the element of the tallyable object.

    :Example:
    >>> var_str('theta', (4,))
    ['theta[1]', 'theta[2]', 'theta[3]', 'theta[4]']

    """

    size = prod(shape)
    ind = (indices(shape) + 1).reshape(-1, size)
    names = ['[' + ','.join(map(str, i)) + ']' for i in zip(*ind)]
    # if len(name)>12:
    #     name = '\n'.join(name.split('_'))
    #     name += '\n'
    names[0] = '%s %s' % (name, names[0])
    return names


def summary_plot(
    pymc_obj, name='model', format='png', suffix='-summary', path='./',
    alpha=0.05, quartiles=True, hpd=True, rhat=True, main=None, xlab=None, x_range=None,
        custom_labels=None, chain_spacing=0.05, vline_pos=0):
    """
    Model summary plot

    Generates a "forest plot" of 100*(1-alpha)% credible intervals for either the
    set of nodes in a given model, or a specified set of nodes.

    :Arguments:
        pymc_obj: PyMC object, trace or array
            A trace from an MCMC sample or a PyMC object with one or more traces.

        name (optional): string
            The name of the object.

        format (optional): string
            Graphic output format (defaults to png).

        suffix (optional): string
            Filename suffix.

        path (optional): string
            Specifies location for saving plots (defaults to local directory).

        alpha (optional): float
            Alpha value for (1-alpha)*100% credible intervals (defaults to 0.05).

        quartiles (optional): bool
            Flag for plotting the interquartile range, in addition to the
            (1-alpha)*100% intervals (defaults to True).

        hpd (optional): bool
            Flag for plotting the highest probability density (HPD) interval
            instead of the central (1-alpha)*100% interval (defaults to True).

        rhat (optional): bool
            Flag for plotting Gelman-Rubin statistics. Requires 2 or more
            chains (defaults to True).

        main (optional): string
            Title for main plot. Passing False results in titles being
            suppressed; passing False (default) results in default titles.

        xlab (optional): string
            Label for x-axis. Defaults to no label

        x_range (optional): list or tuple
            Range for x-axis. Defaults to matplotlib's best guess.

        custom_labels (optional): list
            User-defined labels for each node. If not provided, the node
            __name__ attributes are used.

        chain_spacing (optional): float
            Plot spacing between chains (defaults to 0.05).

        vline_pos (optional): numeric
            Location of vertical reference line (defaults to 0).

    """

    if not gridspec:
        print_(
            '\nYour installation of matplotlib is not recent enough to support summary_plot; this function is disabled until matplotlib is updated.')
        return

    # Quantiles to be calculated
    quantiles = [100 * alpha / 2, 50, 100 * (1 - alpha / 2)]
    if quartiles:
        quantiles = [100 * alpha / 2, 25, 50, 75, 100 * (1 - alpha / 2)]

    # Range for x-axis
    plotrange = None

    # Number of chains
    chains = None

    # Gridspec
    gs = None

    # Subplots
    interval_plot = None
    rhat_plot = None

    try:
        # First try Model type
        vars = pymc_obj._variables_to_tally

    except AttributeError:

        try:

            # Try a database object
            vars = pymc_obj._traces

        except AttributeError:

            # Assume an iterable
            vars = pymc_obj

    from .diagnostics import gelman_rubin

    # Calculate G-R diagnostics
    if rhat:
        try:
            R = gelman_rubin(pymc_obj)
        except (ValueError, TypeError):
            try:
                R = {}
                for variable in vars:
                    R[variable.__name__] = gelman_rubin(variable)
            except ValueError:
                print(
                    'Could not calculate Gelman-Rubin statistics. Requires multiple chains of equal length.')
                rhat = False

    # Empty list for y-axis labels
    labels = []
    # Counter for current variable
    var = 1

    # Make sure there is something to print
    if all([v._plot == False for v in vars]):
        print_('No variables to plot')
        return

    for variable in vars:

        # If plot flag is off, do not print
        if variable._plot == False:
            continue

        # Extract name
        varname = variable.__name__

        # Retrieve trace(s)
        i = 0
        traces = []
        while True:
            try:
                # traces.append(pymc_obj.trace(varname, chain=i)[:])
                traces.append(variable.trace(chain=i))
                i += 1
            except (KeyError, IndexError):
                break

        chains = len(traces)

        if gs is None:
            # Initialize plot
            if rhat and chains > 1:
                gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])

            else:

                gs = gridspec.GridSpec(1, 1)

            # Subplot for confidence intervals
            interval_plot = subplot(gs[0])

        # Get quantiles
        data = [calc_quantiles(d, quantiles) for d in traces]
        if hpd:
            # Substitute HPD interval
            for i, d in enumerate(traces):
                hpd_interval = calc_hpd(d, alpha).T
                data[i][quantiles[0]] = hpd_interval[0]
                data[i][quantiles[-1]] = hpd_interval[1]

        data = [[d[q] for q in quantiles] for d in data]
        # Ensure x-axis contains range of current interval
        if plotrange:
            plotrange = [min(
                         plotrange[0],
                         nmin(data)),
                         max(plotrange[1],
                             nmax(data))]
        else:
            plotrange = [nmin(data), nmax(data)]

        try:
            # First try missing-value stochastic
            value = variable.get_stoch_value()
        except AttributeError:
            # All other variable types
            value = variable.value

        # Number of elements in current variable
        k = size(value)

        # Append variable name(s) to list
        if k > 1:
            names = var_str(varname, shape(value))
            labels += names
        else:
            labels.append(varname)
            # labels.append('\n'.join(varname.split('_')))

        # Add spacing for each chain, if more than one
        e = [0] + [(chain_spacing * ((i + 2) / 2)) * (
            -1) ** i for i in range(chains - 1)]

        # Loop over chains
        for j, quants in enumerate(data):

            # Deal with multivariate nodes
            if k > 1:

                for i, q in enumerate(transpose(quants)):

                    # Y coordinate with jitter
                    y = -(var + i) + e[j]

                    if quartiles:
                        # Plot median
                        pyplot(q[2], y, 'bo', markersize=4)
                        # Plot quartile interval
                        errorbar(
                            x=(q[1],
                                q[3]),
                            y=(y,
                                y),
                            linewidth=2,
                            color="blue")

                    else:
                        # Plot median
                        pyplot(q[1], y, 'bo', markersize=4)

                    # Plot outer interval
                    errorbar(
                        x=(q[0],
                            q[-1]),
                        y=(y,
                            y),
                        linewidth=1,
                        color="blue")

            else:

                # Y coordinate with jitter
                y = -var + e[j]

                if quartiles:
                    # Plot median
                    pyplot(quants[2], y, 'bo', markersize=4)
                    # Plot quartile interval
                    errorbar(
                        x=(quants[1],
                            quants[3]),
                        y=(y,
                            y),
                        linewidth=2,
                        color="blue")
                else:
                    # Plot median
                    pyplot(quants[1], y, 'bo', markersize=4)

                # Plot outer interval
                errorbar(
                    x=(quants[0],
                        quants[-1]),
                    y=(y,
                        y),
                    linewidth=1,
                    color="blue")

        # Increment index
        var += k

    if custom_labels is not None:
        labels = custom_labels

    # Update margins
    left_margin = max([len(x) for x in labels]) * 0.015
    gs.update(left=left_margin, right=0.95, top=0.9, bottom=0.05)

    # Define range of y-axis
    ylim(-var + 0.5, -0.5)

    datarange = plotrange[1] - plotrange[0]
    xlim(plotrange[0] - 0.05 * datarange, plotrange[1] + 0.05 * datarange)

    # Add variable labels
    yticks([-(l + 1) for l in range(len(labels))], labels)

    # Add title
    if main is not False:
        plot_title = main or str(int((
            1 - alpha) * 100)) + "% Credible Intervals"
        title(plot_title)

    # Add x-axis label
    if xlab is not None:
        xlabel(xlab)

    # Constrain to specified range
    if x_range is not None:
        xlim(*x_range)

    # Remove ticklines on y-axes
    for ticks in interval_plot.yaxis.get_major_ticks():
        ticks.tick1On = False
        ticks.tick2On = False

    for loc, spine in six.iteritems(interval_plot.spines):
        if loc in ['bottom', 'top']:
            pass
            # spine.set_position(('outward',10)) # outward by 10 points
        elif loc in ['left', 'right']:
            spine.set_color('none')  # don't draw spine

    # Reference line
    axvline(vline_pos, color='k', linestyle='--')

    # Genenerate Gelman-Rubin plot
    if rhat and chains > 1:

        # If there are multiple chains, calculate R-hat
        rhat_plot = subplot(gs[1])

        if main is not False:
            title("R-hat")

        # Set x range
        xlim(0.9, 2.1)

        # X axis labels
        xticks((1.0, 1.5, 2.0), ("1", "1.5", "2+"))
        yticks([-(l + 1) for l in range(len(labels))], "")

        i = 1
        for variable in vars:

            if variable._plot == False:
                continue

            # Extract name
            varname = variable.__name__

            try:
                value = variable.get_stoch_value()
            except AttributeError:
                value = variable.value

            k = size(value)

            if k > 1:
                pyplot([min(r, 2) for r in R[varname]], [-(j + i)
                       for j in range(k)], 'bo', markersize=4)
            else:
                pyplot(min(R[varname], 2), -i, 'bo', markersize=4)

            i += k

        # Define range of y-axis
        ylim(-i + 0.5, -0.5)

        # Remove ticklines on y-axes
        for ticks in rhat_plot.yaxis.get_major_ticks():
            ticks.tick1On = False
            ticks.tick2On = False

        for loc, spine in six.iteritems(rhat_plot.spines):
            if loc in ['bottom', 'top']:
                pass
                # spine.set_position(('outward',10)) # outward by 10 points
            elif loc in ['left', 'right']:
                spine.set_color('none')  # don't draw spine

    savefig("%s%s%s.%s" % (path, name, suffix, format))
