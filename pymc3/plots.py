from collections import OrderedDict
import itertools

import numpy as np
from scipy.stats import kde, mode
from matplotlib import gridspec
import matplotlib.pyplot as plt
from .diagnostics import gelman_rubin
from .stats import quantiles, hpd
from scipy.signal import gaussian, convolve

__all__ = ['traceplot', 'kdeplot', 'kde2plot',
           'forestplot', 'autocorrplot', 'plot_posterior', 'compare_plot']


def get_axis(ax, default_rows, default_columns, **default_kwargs):
    """Verifies the provided axis is of the correct shape, and creates one if needed.

    Args:
        ax: matplotlib axis or None
        default_rows: int, expected rows in axis
        default_columns: int, expected columns in axis
        **default_kwargs: keyword arguments to pass to plt.subplot

    Returns:
        axis, or raises an error
    """

    default_shape = (default_rows, default_columns)
    if ax is None:
        _, ax = plt.subplots(*default_shape, **default_kwargs)
    elif ax.shape != default_shape:
        raise ValueError('Subplots with shape %r required' % (default_shape,))
    return ax


def identity_transform(x):
    """f(x) = x"""
    return x


def get_default_varnames(trace, include_transformed):
    """Helper to extract default varnames from a trace."""
    if include_transformed:
        return [name for name in trace.varnames]
    else:
        return [name for name in trace.varnames if not name.endswith('_')]


def traceplot(trace, varnames=None, transform=identity_transform, figsize=None, lines=None,
              combined=False, plot_transformed=False, grid=False, alpha=0.35, priors=None,
              prior_alpha=1, prior_style='--', ax=None):
    """Plot samples histograms and values

    Parameters
    ----------

    trace : result of MCMC run
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    transform : callable
        Function to transform data (defaults to identity)
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    lines : dict
        Dictionary of variable name / value  to be overplotted as vertical
        lines to the posteriors and horizontal lines on sample values
        e.g. mean of posteriors, true values of a simulation.
        If an array of values, line colors are matched to posterior colors.
        Otherwise, a default red line
    combined : bool
        Flag for combining multiple chains into a single chain. If False
        (default), chains will be plotted separately.
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).
    grid : bool
        Flag for adding gridlines to histogram. Defaults to True.
    alpha : float
        Alpha value for plot line. Defaults to 0.35.
    priors : iterable of PyMC distributions
        PyMC prior distribution(s) to be plotted alongside posterior. Defaults
        to None (no prior plots).
    prior_alpha : float
        Alpha value for prior plot. Defaults to 1.
    prior_style : str
        Line style for prior plot. Defaults to '--' (dashed line).
    ax : axes
        Matplotlib axes. Accepts an array of axes, e.g.:

        >>> fig, axs = plt.subplots(3, 2) # 3 RVs
        >>> pymc3.traceplot(trace, ax=axs)

        Creates own axes by default.

    Returns
    -------

    ax : matplotlib axes

    """

    if varnames is None:
        varnames = get_default_varnames(trace, plot_transformed)

    if figsize is None:
        figsize = (12, len(varnames) * 2)

    ax = get_axis(ax, len(varnames), 2, squeeze=False, figsize=figsize)

    for i, v in enumerate(varnames):
        if priors is not None:
            prior = priors[i]
        else:
            prior = None
        for d in trace.get_values(v, combine=combined, squeeze=False):
            d = np.squeeze(transform(d))
            d = make_2d(d)
            if d.dtype.kind == 'i':
                hist_objs = histplot_op(ax[i, 0], d, alpha=alpha)
                colors = [h[-1][0].get_facecolor() for h in hist_objs]
            else:
                artists = kdeplot_op(ax[i, 0], d, prior, prior_alpha, prior_style)[0]
                colors = [a[0].get_color() for a in artists]
            ax[i, 0].set_title(str(v))
            ax[i, 0].grid(grid)
            ax[i, 1].set_title(str(v))
            ax[i, 1].plot(d, alpha=alpha)

            ax[i, 0].set_ylabel("Frequency")
            ax[i, 1].set_ylabel("Sample value")

            if lines:
                try:
                    if isinstance(lines[v], (float, int)):
                        line_values, colors = [lines[v]], ['r']
                    else:
                        line_values = np.atleast_1d(lines[v]).ravel()
                        if len(colors) != len(line_values):
                            raise AssertionError("An incorrect number of lines was specified for "
                                                 "'{}'. Expected an iterable of length {} or to "
                                                 " a scalar".format(v, len(colors)))
                    for c, l in zip(colors, line_values):
                        ax[i, 0].axvline(x=l, color=c, lw=1.5, alpha=0.75)
                        ax[i, 1].axhline(y=l, color=c,
                                         lw=1.5, alpha=alpha)
                except KeyError:
                    pass
        ax[i, 0].set_ylim(ymin=0)
    plt.tight_layout()
    return ax


def histplot_op(ax, data, alpha=.35):
    hs = []
    data_min = np.min(data)
    data_max = np.max(data)
    for i in range(data.shape[1]):
        d = data[:, i]

        mind = np.min(d)
        maxd = np.max(d)
        step = max((maxd - mind) // 100, 1)
        hs.append(ax.hist(d, bins=range(mind, maxd + 2, step), alpha=alpha, align='left'))
    ax.set_xlim(data_min - .5, data_max + .5)
    return hs


def kdeplot_op(ax, data, prior=None, prior_alpha=1, prior_style='--'):
    ls = []
    pls = []
    errored = []
    for i in range(data.shape[1]):
        d = data[:, i]
        try:
            density, l, u = fast_kde(d)
            x = np.linspace(l, u, len(density))

            if prior is not None:
                p = prior.logp(x).eval()
                pls.append(
                    ax.plot(x, np.exp(p), alpha=prior_alpha, ls=prior_style))

            ls.append(ax.plot(x, density))
        except ValueError:
            errored.append(str(i))

    if errored:
        ax.text(.27, .47, 'WARNING: KDE plot failed for: ' + ','.join(errored),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                style='italic')

    return ls, pls


def make_2d(a):
    """Ravel the dimensions after the first."""
    a = np.atleast_2d(a.T).T
    # flatten out dimensions beyond the first
    n = a.shape[0]
    newshape = np.product(a.shape[1:]).astype(int)
    a = a.reshape((n, newshape), order='F')
    return a


def kde2plot_op(ax, x, y, grid=200, **kwargs):

    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    extent = kwargs.pop('extent', [])
    if len(extent) != 4:
        extent = [xmin, xmax, ymin, ymax]

    grid = grid * 1j
    X, Y = np.mgrid[xmin:xmax:grid, ymin:ymax:grid]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = kde.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.imshow(np.rot90(Z), extent=extent, **kwargs)


def kdeplot(data, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True)
    kdeplot_op(ax, data)
    return ax


def kde2plot(x, y, grid=200, ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1, squeeze=True)
    kde2plot_op(ax, x, y, grid, **kwargs)
    return ax


def autocorrplot(trace, varnames=None, max_lag=100, burn=0, plot_transformed=False,
                 symmetric_plot=False, ax=None, figsize=None):
    """Bar plot of the autocorrelation function for a trace

    Parameters
    ----------
    trace : result of MCMC run
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted.
        Vector-value stochastics are handled automatically.
    max_lag : int
        Maximum lag to calculate autocorrelation. Defaults to 100.
    burn : int
        Number of samples to discard from the beginning of the trace.
        Defaults to 0.
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).
    symmetric_plot : boolean
        Plot from either [0, +lag] or [-lag, lag]. Defaults to False, [-, +lag].
    ax : axes
        Matplotlib axes. Defaults to None.
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inches.
        Note this is not used if ax is supplied.

    Returns
    -------
    ax : matplotlib axes
    """

    def _handle_array_varnames(varname):
        if trace[0][varname].__class__ is np.ndarray:
            k = trace[varname].shape[1]
            for i in range(k):
                yield varname + '_{0}'.format(i)
        else:
            yield varname

    if varnames is None:
        varnames = get_default_varnames(trace, plot_transformed)

    varnames = list(itertools.chain.from_iterable(map(_handle_array_varnames, varnames)))

    nchains = trace.nchains

    if figsize is None:
        figsize = (12, len(varnames) * 2)

    ax = get_axis(ax, len(varnames), nchains,
                  squeeze=False, sharex=True, sharey=True, figsize=figsize)

    max_lag = min(len(trace) - 1, max_lag)

    for i, v in enumerate(varnames):
        for j in range(nchains):
            try:
                d = np.squeeze(trace.get_values(v, chains=[j], burn=burn,
                                                combine=False))
            except KeyError:
                k = int(v.split('_')[-1])
                v_use = '_'.join(v.split('_')[:-1])
                d = np.squeeze(trace.get_values(v_use, chains=[j],
                                                burn=burn, combine=False)[:, k])

            ax[i, j].acorr(d, detrend=plt.mlab.detrend_mean, maxlags=max_lag)

            if j == 0:
                ax[i, j].set_ylabel("correlation")

            if i == len(varnames) - 1:
                ax[i, j].set_xlabel("lag")

            ax[i, j].set_title(v)

            if not symmetric_plot:
                ax[i, j].set_xlim(0, max_lag)

            if nchains > 1:
                ax[i, j].set_title("chain {0}".format(j + 1))

    return ax


def var_str(name, shape):
    """Return a sequence of strings naming the element of the tallyable object.
    This is a support function for forestplot.

    :Example:
    >>> var_str('theta', (4,))
    ['theta[1]', 'theta[2]', 'theta[3]', 'theta[4]']

    """

    size = np.prod(shape)
    ind = (np.indices(shape) + 1).reshape(-1, size)
    names = ['[' + ','.join(map(str, i)) + ']' for i in zip(*ind)]
    names[0] = '%s %s' % (name, names[0])
    return names


def make_rhat_plot(trace, ax, title, labels, varnames, include_transformed):
    if varnames is None:
        varnames = get_default_varnames(trace, include_transformed)

    R = gelman_rubin(trace)
    R = {v: R[v] for v in varnames}

    ax.set_title(title)

    # Set x range
    ax.set_xlim(0.9, 2.1)

    # X axis labels
    ax.set_xticks((1.0, 1.5, 2.0), ("1", "1.5", "2+"))
    ax.set_yticks([-(l + 1) for l in range(len(labels))], "")

    i = 1
    for varname in varnames:
        chain = trace.chains[0]
        value = trace.get_values(varname, chains=[chain])[0]
        k = np.size(value)

        if k > 1:
            ax.plot([min(r, 2) for r in R[varname]],
                    [-(j + i) for j in range(k)], 'bo', markersize=4)
        else:
            ax.plot(min(R[varname], 2), -i, 'bo', markersize=4)

        i += k

    # Define range of y-axis
    ax.set_ylim(-i + 0.5, -0.5)

    # Remove ticklines on y-axes
    ax.set_yticks([])

    for loc, spine in ax.spines.items():
        if loc in ['left', 'right']:
            spine.set_color('none')  # don't draw spine
    return ax


def plot_tree(ax, y, ntiles, show_quartiles):
    """ Helper to plot errorbars for the forestplot.

    Parameters
    ----------
    ax: Matplotlib.Axes
    y: float
        y value to add error bar to
    ntiles: iterable
        A list or array of length 5 or 3
    show_quartiles: boolean
        Whether to plot the interquartile range

    Returns
    -------

    Matplotlib.Axes with a single error bar added

    """
    if show_quartiles:
        # Plot median
        ax.plot(ntiles[2], y, 'bo', markersize=4)
        # Plot quartile interval
        ax.errorbar(x=(ntiles[1], ntiles[3]), y=(y, y), linewidth=2, color='b')

    else:
        # Plot median
        ax.plot(ntiles[1], y, 'bo', markersize=4)

    # Plot outer interval
    ax.errorbar(x=(ntiles[0], ntiles[-1]), y=(y, y), linewidth=1, color='b')
    return ax


def forestplot(trace_obj, varnames=None, transform=identity_transform, alpha=0.05, quartiles=True,
               rhat=True, main=None, xtitle=None, xlim=None, ylabels=None,
               chain_spacing=0.05, vline=0, gs=None, plot_transformed=False):
    """
    Forest plot (model summary plot)
    Generates a "forest plot" of 100*(1-alpha)% credible intervals for either
    the set of variables in a given model, or a specified set of nodes.

    Parameters
    ----------

    trace_obj: NpTrace or MultiTrace object
        Trace(s) from an MCMC sample.
    varnames: list
        List of variables to plot (defaults to None, which results in all
        variables plotted).
    transform : callable
        Function to transform data (defaults to identity)
    alpha (optional): float
        Alpha value for (1-alpha)*100% credible intervals (defaults to 0.05).
    quartiles (optional): bool
        Flag for plotting the interquartile range, in addition to the
        (1-alpha)*100% intervals (defaults to True).
    rhat (optional): bool
        Flag for plotting Gelman-Rubin statistics. Requires 2 or more chains
        (defaults to True).
    main (optional): string
        Title for main plot. Passing False results in titles being suppressed;
        passing None (default) results in default titles.
    xtitle (optional): string
        Label for x-axis. Defaults to no label
    xlim (optional): list or tuple
        Range for x-axis. Defaults to matplotlib's best guess.
    ylabels (optional): list or array
        User-defined labels for each variable. If not provided, the node
        __name__ attributes are used.
    chain_spacing (optional): float
        Plot spacing between chains (defaults to 0.05).
    vline (optional): numeric
        Location of vertical reference line (defaults to 0).
    gs : GridSpec
        Matplotlib GridSpec object. Defaults to None.
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).

    Returns
    -------

    gs : matplotlib GridSpec

    """

    # Quantiles to be calculated
    if quartiles:
        qlist = [100 * alpha / 2, 25, 50, 75, 100 * (1 - alpha / 2)]
    else:
        qlist = [100 * alpha / 2, 50, 100 * (1 - alpha / 2)]

    # Range for x-axis
    plotrange = None

    # Subplots
    interval_plot = None

    nchains = trace_obj.nchains

    if varnames is None:
        varnames = get_default_varnames(trace_obj, plot_transformed)

    plot_rhat = (rhat and nchains > 1)
    # Empty list for y-axis labels
    if gs is None:
        # Initialize plot
        if plot_rhat:
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        else:
            gs = gridspec.GridSpec(1, 1)

        # Subplot for confidence intervals
        interval_plot = plt.subplot(gs[0])

    trace_quantiles = quantiles(trace_obj, qlist, transform=transform, squeeze=False)
    hpd_intervals = hpd(trace_obj, alpha, transform=transform, squeeze=False)

    labels = []
    for j, chain in enumerate(trace_obj.chains):
        # Counter for current variable
        var = 0
        for varname in varnames:
            var_quantiles = trace_quantiles[chain][varname]

            quants = [var_quantiles[v] for v in qlist]
            var_hpd = hpd_intervals[chain][varname].T

            # Substitute HPD interval for quantile
            quants[0] = var_hpd[0].T
            quants[-1] = var_hpd[1].T

            # Ensure x-axis contains range of current interval
            if plotrange:
                plotrange = [min(plotrange[0], np.min(quants)),
                             max(plotrange[1], np.max(quants))]
            else:
                plotrange = [np.min(quants), np.max(quants)]

            # Number of elements in current variable
            value = trace_obj.get_values(varname, chains=[chain])[0]
            k = np.size(value)

            # Append variable name(s) to list
            if j == 0:
                if k > 1:
                    names = var_str(varname, np.shape(value))
                    labels += names
                else:
                    labels.append(varname)

            # Add spacing for each chain, if more than one
            offset = [0] + [(chain_spacing * ((i + 2) / 2)) * (-1) ** i for i in range(nchains - 1)]

            # Y coordinate with offset
            y = -var + offset[j]

            # Deal with multivariate nodes
            if k > 1:
                for q in np.transpose(quants).squeeze():
                    # Multiple y values
                    y -= 1
                    interval_plot = plot_tree(interval_plot, y, q, quartiles)
            else:
                interval_plot = plot_tree(interval_plot, y, quants, quartiles)

            # Increment index
            var += k

    labels = ylabels if ylabels is not None else labels

    # Update margins
    left_margin = np.max([len(x) for x in labels]) * 0.015
    gs.update(left=left_margin, right=0.95, top=0.9, bottom=0.05)

    # Define range of y-axis
    interval_plot.set_ylim(-var + 0.5, -0.5)

    datarange = plotrange[1] - plotrange[0]
    interval_plot.set_xlim(plotrange[0] - 0.05 * datarange, plotrange[1] + 0.05 * datarange)

    # Add variable labels
    interval_plot.set_yticks([-(l + 1) for l in range(len(labels))])
    interval_plot.set_yticklabels(labels)

    # Add title
    plot_title = ""
    if main is None:
        plot_title = "{:.0f}% Credible Intervals".format((1 - alpha) * 100)
    elif main:
        plot_title = main
    if plot_title:
        interval_plot.set_title(plot_title)

    # Add x-axis label
    if xtitle is not None:
        interval_plot.set_xlabel(xtitle)

    # Constrain to specified range
    if xlim is not None:
        interval_plot.set_xlim(*xlim)

    # Remove ticklines on y-axes
    for ticks in interval_plot.yaxis.get_major_ticks():
        ticks.tick1On = False
        ticks.tick2On = False

    for loc, spine in interval_plot.spines.items():
        if loc in ['left', 'right']:
            spine.set_color('none')  # don't draw spine

    # Reference line
    interval_plot.axvline(vline, color='k', linestyle='--')

    # Genenerate Gelman-Rubin plot
    if plot_rhat:
        make_rhat_plot(trace_obj, plt.subplot(gs[1]), "R-hat", labels, varnames, plot_transformed)

    return gs


def plot_posterior(trace, varnames=None, transform=identity_transform, figsize=None,
                   alpha_level=0.05, round_to=3, point_estimate='mean', rope=None,
                   ref_val=None, kde_plot=False, plot_transformed=False, ax=None, **kwargs):
    """Plot Posterior densities in style of John K. Kruschke book

    Parameters
    ----------

    trace : result of MCMC run
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    transform : callable
        Function to transform data (defaults to identity)
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    alpha_level : float
        Defines range for High Posterior Density
    round_to : int
        Controls formatting for floating point numbers
    point_estimate: str
        Must be in ('mode', 'mean', 'median')
    rope: list or numpy array
        Lower and upper values of the Region Of Practical Equivalence
    ref_val: bool
        display the percentage below and above ref_val
    kde_plot: bool
        if True plot a KDE instead of a histogram
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).
    ax : axes
        Matplotlib axes. Defaults to None.
    **kwargs
        Passed as-is to plt.hist() or plt.plot() function, depending on the
        value of the argument kde_plot
        Some defaults are added, if not specified
        color='#87ceeb' will match the style in the book


    Returns
    -------

    ax : matplotlib axes

    """

    def plot_posterior_op(trace_values, ax):
        def format_as_percent(x, round_to=0):
            return '{0:.{1:d}f}%'.format(100 * x, round_to)

        def display_ref_val(ref_val):
            less_than_ref_probability = (trace_values < ref_val).mean()
            greater_than_ref_probability = (trace_values >= ref_val).mean()
            ref_in_posterior = "{} <{:g}< {}".format(
                format_as_percent(less_than_ref_probability, 1),
                ref_val,
                format_as_percent(greater_than_ref_probability, 1))
            ax.axvline(ref_val, ymin=0.02, ymax=.75, color='g',
                       linewidth=4, alpha=0.65)
            ax.text(trace_values.mean(), plot_height * 0.6, ref_in_posterior,
                    size=14, horizontalalignment='center')

        def display_rope(rope):
            ax.plot(rope, (plot_height * 0.02, plot_height * 0.02),
                    linewidth=20, color='r', alpha=0.75)
            text_props = dict(size=16, horizontalalignment='center', color='r')
            ax.text(rope[0], plot_height * 0.14, rope[0], **text_props)
            ax.text(rope[1], plot_height * 0.14, rope[1], **text_props)

        def display_point_estimate():
            if not point_estimate:
                return
            if point_estimate not in ('mode', 'mean', 'median'):
                raise ValueError(
                    "Point Estimate should be in ('mode','mean','median', None)")
            if point_estimate == 'mean':
                point_value = trace_values.mean()
                point_text = '{}={}'.format(
                    point_estimate, point_value.round(round_to))
            elif point_estimate == 'mode':
                point_value = mode(trace_values.round(round_to))[0][0]
                point_text = '{}={}'.format(
                    point_estimate, point_value.round(round_to))
            elif point_estimate == 'median':
                point_value = np.median(trace_values)
                point_text = '{}={}'.format(
                    point_estimate, point_value.round(round_to))

            ax.text(point_value, plot_height * 0.8, point_text,
                    size=16, horizontalalignment='center')

        def display_hpd():
            hpd_intervals = hpd(trace_values, alpha=alpha_level)
            ax.plot(hpd_intervals, (plot_height * 0.02,
                                    plot_height * 0.02), linewidth=4, color='k')
            text_props = dict(size=16, horizontalalignment='center')
            ax.text(hpd_intervals[0], plot_height * 0.07,
                    hpd_intervals[0].round(round_to), **text_props)
            ax.text(hpd_intervals[1], plot_height * 0.07,
                    hpd_intervals[1].round(round_to), **text_props)
            ax.text((hpd_intervals[0] + hpd_intervals[1]) / 2, plot_height * 0.2,
                    format_as_percent(1 - alpha_level) + ' HPD', **text_props)

        def format_axes():
            ax.yaxis.set_ticklabels([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.tick_params(axis='x', direction='out', width=1, length=3,
                           color='0.5')
            ax.spines['bottom'].set_color('0.5')

        def set_key_if_doesnt_exist(d, key, value):
            if key not in d:
                d[key] = value

        if kde_plot:
            density, l, u = fast_kde(trace_values)
            x = np.linspace(l, u, len(density))
            ax.plot(x, density, **kwargs)
        else:
            set_key_if_doesnt_exist(kwargs, 'bins', 30)
            set_key_if_doesnt_exist(kwargs, 'edgecolor', 'w')
            set_key_if_doesnt_exist(kwargs, 'align', 'right')
            ax.hist(trace_values, **kwargs)

        plot_height = ax.get_ylim()[1]

        format_axes()
        display_hpd()
        display_point_estimate()
        if ref_val is not None:
            display_ref_val(ref_val)
        if rope is not None:
            display_rope(rope)

    def create_axes_grid(figsize, traces):
        n = np.ceil(len(traces) / 2.0).astype(int)
        if figsize is None:
            figsize = (12, n * 2.5)
        fig, ax = plt.subplots(n, 2, figsize=figsize)
        ax = ax.reshape(2 * n)
        if len(traces) % 2 == 1:
            ax[-1].set_axis_off()
            ax = ax[:-1]
        return ax, fig

    def get_trace_dict(tr, varnames):
        traces = OrderedDict()
        for v in varnames:
            vals = tr.get_values(v, combine=True, squeeze=True)
            if vals.ndim > 1:
                vals_flat = vals.reshape(vals.shape[0], -1).T
                for i, vi in enumerate(vals_flat):
                    traces['_'.join([v, str(i)])] = vi
            else:
                traces[v] = vals
        return traces

    if isinstance(trace, np.ndarray):
        if figsize is None:
            figsize = (6, 2)
        if ax is None:
            fig, ax = plt.subplots()
        plot_posterior_op(transform(trace), ax)
    else:
        if varnames is None:
            varnames = get_default_varnames(trace, plot_transformed)

        trace_dict = get_trace_dict(trace, varnames)

        if ax is None:
            ax, fig = create_axes_grid(figsize, trace_dict)

        for a, v in zip(ax, trace_dict):
            tr_values = transform(trace_dict[v])
            plot_posterior_op(tr_values, ax=a)
            a.set_title(v)

        fig.tight_layout()
    return ax


def fast_kde(x):
    """
    A fft-based Gaussian kernel density estimate (KDE) for computing
    the KDE on a regular grid.
    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------

    x : Numpy array or list

    Returns
    -------

    grid: A gridded 1D KDE of the input points (x).
    xmin: minimum value of x
    xmax: maximum value of x

    """
    x = x[~np.isnan(x)]
    x = x[~np.isinf(x)]
    n = len(x)
    nx = 200

    # add small jitter in case input values are the same
    x += np.random.uniform(-1E-12, 1E-12, size=n)
    xmin, xmax = np.min(x), np.max(x)

    # compute histogram
    bins = np.linspace(xmin, xmax, nx)
    xyi = np.digitize(x, bins)
    dx = (xmax - xmin) / (nx - 1)
    grid = np.histogram(x, bins=nx)[0]

    # Scaling factor for bandwidth
    scotts_factor = n ** (-0.2)
    # Determine the bandwidth using Scott's rule
    std_x = np.std(xyi)
    kern_nx = int(np.round(scotts_factor * 2 * np.pi * std_x))

    # Evaluate the gaussian function on the kernel grid
    kernel = np.reshape(gaussian(kern_nx, scotts_factor * std_x), kern_nx)

    # Compute the KDE
    # use symmetric padding to correct for data boundaries in the kde
    npad = np.min((nx, 2 * kern_nx))

    grid = np.concatenate([grid[npad: 0: -1], grid, grid[nx: nx - npad: -1]])
    grid = convolve(grid, kernel, mode='same')[npad: npad + nx]

    norm_factor = n * dx * (2 * np.pi * std_x ** 2 * scotts_factor ** 2) ** 0.5

    grid /= norm_factor

    return grid, xmin, xmax


def compare_plot(comp_df, ax=None):
    """
    Model comparison summary plot in the style of the one used in the book
    Statistical Rethinking by Richard McElreath.

    Parameters
    ----------

    comp_df: DataFrame
        The result of the pm.compare() function
    ax : axes
        Matplotlib axes. Defaults to None.

    Returns
    -------

    ax : matplotlib axes

    """
    if ax is None:
        _, ax = plt.subplots()

    yticks_pos = np.linspace(0, -1, (comp_df.shape[0] * 2) - 1)
    yticks_labels = np.repeat(comp_df.index, 2)[1:]

    data = comp_df.values
    min_ic = data[0, 0]

    ax.errorbar(x=data[:, 0], y=yticks_pos[::2], xerr=data[:, 4], fmt='ko', mfc='None', mew=1)
    ax.errorbar(x=data[1:, 0], y=yticks_pos[1::2], xerr=data[1:, 5], fmt='^', color='grey')

    ax.plot(data[:, 0] - (2 * data[:, 1]), yticks_pos[::2], 'ko')
    ax.axvline(min_ic, ls='--', color='grey')

    ax.set_yticks(yticks_pos)
    ax.set_yticklabels(yticks_labels)
    ax.set_xlabel('Deviance')

    return ax

