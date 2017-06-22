from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from pymc3.diagnostics import gelman_rubin
from pymc3.stats import quantiles, hpd
from .utils import identity_transform, get_default_varnames


def _var_str(name, shape):
    """Return a sequence of strings naming the element of the tallyable object.

    :Example:
    >>> _var_str('theta', (4,))
    ['theta[1]', 'theta[2]', 'theta[3]', 'theta[4]']
    """
    size = np.prod(shape)
    ind = (np.indices(shape) + 1).reshape(-1, size)
    names = ['[' + ','.join(map(str, i)) + ']' for i in zip(*ind)]
    names[0] = '%s %s' % (name, names[0])
    return names


def _make_rhat_plot(trace, ax, title, labels, varnames, include_transformed):
    """Helper to plot rhat for multiple chains.

    Parameters
    ----------
    trace: pymc3 trace
    ax: Matplotlib.Axes
    title: str
    labels: iterable
        Same length as the number of chains
    include_transformed: bool
        Whether to include the transformed variables

    Returns
    -------

    Matplotlib.Axes with a single error bar added

    """
    if varnames is None:
        varnames = get_default_varnames(trace.varnames, include_transformed)

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


def _plot_tree(ax, y, ntiles, show_quartiles, **plot_kwargs):
    """Helper to plot errorbars for the forestplot.

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
        ax.plot(ntiles[2], y, color=plot_kwargs.get('color', 'blue'), 
                        marker=plot_kwargs.get('marker', 'o'), 
                        markersize=plot_kwargs.get('markersize', 4))
        # Plot quartile interval
        ax.errorbar(x=(ntiles[1], ntiles[3]), y=(y, y), 
                        linewidth=plot_kwargs.get('linewidth', 2), 
                        color=plot_kwargs.get('color', 'blue'))

    else:
        # Plot median
        ax.plot(ntiles[1], y, marker=plot_kwargs.get('marker', 'o'), 
                            color=plot_kwargs.get('color', 'blue'),
                            markersize=plot_kwargs.get('markersize', 4))

    # Plot outer interval
    ax.errorbar(x=(ntiles[0], ntiles[-1]), y=(y, y), 
                linewidth=int(plot_kwargs.get('linewidth', 2)/2), 
                color=plot_kwargs.get('color', 'blue'))
                
    return ax


def forestplot(trace_obj, varnames=None, transform=identity_transform, alpha=0.05, quartiles=True,
               rhat=True, main=None, xtitle=None, xlim=None, ylabels=None,
               chain_spacing=0.05, vline=0, gs=None, plot_transformed=False,
               **plot_kwargs):
    """
    Forest plot (model summary plot).

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
    plot_kwargs : dict
        Optional arguments for plot elements. Currently accepts 'fontsize',
        'linewidth', 'color', 'marker', and 'markersize'.

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
        varnames = get_default_varnames(trace_obj.varnames, plot_transformed)

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
                    names = _var_str(varname, np.shape(value))
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
                    interval_plot = _plot_tree(interval_plot, y, q, quartiles,
                                    **plot_kwargs)
                    y -= 1
            else:
                interval_plot = _plot_tree(interval_plot, y, quants, quartiles,
                                **plot_kwargs)

            # Increment index
            var += k

    labels = ylabels if ylabels is not None else labels

    # Update margins
    left_margin = np.max([len(x) for x in labels]) * 0.015
    gs.update(left=left_margin, right=0.95, top=0.9, bottom=0.05)

    # Define range of y-axis
    interval_plot.set_ylim(-var + 0.5, 0.5)

    datarange = plotrange[1] - plotrange[0]
    interval_plot.set_xlim(plotrange[0] - 0.05 * datarange, plotrange[1] + 0.05 * datarange)

    # Add variable labels
    interval_plot.set_yticks([-l for l in range(len(labels))])
    interval_plot.set_yticklabels(labels, fontsize=plot_kwargs.get('fontsize', None))

    # Add title
    plot_title = ""
    if main is None:
        plot_title = "{:.0f}% Credible Intervals".format((1 - alpha) * 100)
    elif main:
        plot_title = main
    if plot_title:
        interval_plot.set_title(plot_title, fontsize=plot_kwargs.get('fontsize', None))

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
    interval_plot.axvline(vline, color='k', linestyle=':')

    # Genenerate Gelman-Rubin plot
    if plot_rhat:
        _make_rhat_plot(trace_obj, plt.subplot(gs[1]), "R-hat", labels, varnames, plot_transformed)

    return gs
