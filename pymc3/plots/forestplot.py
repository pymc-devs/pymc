try:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
except ImportError:  # mpl is optional
    pass
import numpy as np
from pymc3.diagnostics import gelman_rubin
from pymc3.stats import quantiles, hpd, dict2pd
from .utils import identity_transform, get_default_varnames

def _var_str(name, shape):
    """Return a sequence of strings naming the element of the tallyable object.

    :Example:
    >>> _var_str('theta', (4,))
    ['theta[0]', 'theta[1]', 'theta[2]', 'theta[3]']
    """
    size = np.prod(shape)
    ind = (np.indices(shape)).reshape(-1, size)
    names = ['[' + ','.join(map(str, i)) + ']' for i in zip(*ind)]
    names[0] = '%s %s' % (name, names[0])
    return names


def _plot_tree(ax, y, ntiles, show_quartiles, c, plot_kwargs):
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
    c : string
        color
    Returns
    -------

    Matplotlib.Axes with a single error bar added

    """
    if show_quartiles:
        # Plot median
        ax.plot(ntiles[2], y, color=c,
                marker=plot_kwargs.get('marker', 'o'),
                markersize=plot_kwargs.get('markersize', 4))
        # Plot quartile interval
        ax.errorbar(x=(ntiles[1], ntiles[3]), y=(y, y),
                    linewidth=plot_kwargs.get('linewidth', 2),
                    color=c)

    else:
        # Plot median
        ax.plot(ntiles[1], y, marker=plot_kwargs.get('marker', 'o'),
                color=c, markersize=plot_kwargs.get('markersize', 4))

    # Plot outer interval
    ax.errorbar(x=(ntiles[0], ntiles[-1]), y=(y, y),
                linewidth=int(plot_kwargs.get('linewidth', 2)/2),
                color=c)

    return ax


def forestplot(trace, models=None, varnames=None, transform=identity_transform,
               alpha=0.05, quartiles=True, rhat=True, main=None, xtitle=None,
               xlim=None, ylabels=None, colors='C0', chain_spacing=0.1, vline=0,
               gs=None, plot_transformed=False, plot_kwargs=None):
    """
    Forest plot (model summary plot).

    Generates a "forest plot" of 100*(1-alpha)% credible intervals from a trace
    or list of traces.

    Parameters
    ----------

    trace : trace or list of traces
        Trace(s) from an MCMC sample.
    models : list (optional)
        List with names for the models in the list of traces. Useful when
        plotting more that one trace.
    varnames: list
        List of variables to plot (defaults to None, which results in all
        variables plotted).
    transform : callable
        Function to transform data (defaults to identity)
    alpha : float, optional
        Alpha value for (1-alpha)*100% credible intervals (defaults to 0.05).
    quartiles : bool, optional
        Flag for plotting the interquartile range, in addition to the
        (1-alpha)*100% intervals (defaults to True).
    rhat : bool, optional
        Flag for plotting Gelman-Rubin statistics. Requires 2 or more chains
        (defaults to True).
    main : string, optional
        Title for main plot. Passing False results in titles being suppressed;
        passing None (default) results in default titles.
    xtitle : string, optional
        Label for x-axis. Defaults to no label
    xlim : list or tuple, optional
        Range for x-axis. Defaults to matplotlib's best guess.
    ylabels : list or array, optional
        User-defined labels for each variable. If not provided, the node
        __name__ attributes are used.
    colors : list or string, optional
        list with valid matplotlib colors, one color per model. Alternative a
        string can be passed. If the string is `cycle `, it will automatically
        chose a color per model from the matyplolib's cycle. If a single color
        is passed, eg 'k', 'C2', 'red' this color will be used for all models.
        Defauls to 'C0' (blueish in most matplotlib styles)
    chain_spacing : float, optional
        Plot spacing between chains (defaults to 0.1).
    vline : numeric, optional
        Location of vertical reference line (defaults to 0).
    gs : GridSpec
        Matplotlib GridSpec object. Defaults to None.
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).
    plot_kwargs : dict
        Optional arguments for plot elements. Currently accepts 'fontsize',
        'linewidth', 'marker', and 'markersize'.

    Returns
    -------

    gs : matplotlib GridSpec

    """
    if plot_kwargs is None:
        plot_kwargs = {}

    if not isinstance(trace, (list, tuple)):
        trace = [trace]

    if models is None:
        if len(trace) > 1:
            models = ['m_{}'.format(i) for i in range(len(trace))]
        else:
            models = ['']
    elif len(models) != len(trace):
        raise ValueError("The number of names for the models does not match "
                         "the number of models")

    if colors == 'cycle':
        colors = ['C{}'.format(i % 10) for i in range(len(models))]
    elif isinstance(colors, str):
        colors = [colors for i in range(len(models))]

    # Quantiles to be calculated
    if quartiles:
        qlist = [100 * alpha / 2, 25, 50, 75, 100 * (1 - alpha / 2)]
    else:
        qlist = [100 * alpha / 2, 50, 100 * (1 - alpha / 2)]

    nchains = [tr.nchains for tr in trace]

    if varnames is None:
        varnames = []
        for idx, tr in enumerate(trace):
            varnames_tmp = get_default_varnames(tr.varnames, plot_transformed)
            for v in varnames_tmp:
                if v not in varnames:
                    varnames.append(v)

    plot_rhat = [rhat and nch > 1 for nch in nchains]
    # Empty list for y-axis labels
    if gs is None:
        # Initialize plot
        if np.any(plot_rhat):
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            gr_plot = plt.subplot(gs[1])
            gr_plot.set_xticks((1.0, 1.5, 2.0), ("1", "1.5", "2+"))
            gr_plot.set_xlim(0.9, 2.1)
            gr_plot.set_yticks([])
            gr_plot.set_title('R-hat')
        else:
            gs = gridspec.GridSpec(1, 1)

    # Subplot for confidence intervals
    interval_plot = plt.subplot(gs[0])

    trace_quantiles = []
    hpd_intervals = []
    for tr in trace:
        trace_quantiles.append(quantiles(tr, qlist, transform=transform,
                                         squeeze=False))
        hpd_intervals.append(hpd(tr, alpha, transform=transform,
                                 squeeze=False))

    labels = []
    var = 0
    all_quants = []
    bands = [0.05, 0] * len(varnames)
    var_old = 0.5
    for v_idx, v in enumerate(varnames):
        for h, tr in enumerate(trace):
            if v not in tr.varnames:
                labels.append(models[h] + ' ' + v)
                var += 1
            else:
                for j, chain in enumerate(tr.chains):
                    var_quantiles = trace_quantiles[h][chain][v]

                    quants = [var_quantiles[vq] for vq in qlist]
                    var_hpd = hpd_intervals[h][chain][v].T

                    # Substitute HPD interval for quantile
                    quants[0] = var_hpd[0].T
                    quants[-1] = var_hpd[1].T

                    # Ensure x-axis contains range of current interval
                    all_quants.extend(np.ravel(quants))

                    # Number of elements in current variable
                    value = tr.get_values(v, chains=[chain])[0]
                    k = np.size(value)

                    # Append variable name(s) to list
                    if j == 0:
                        if k > 1:
                            names = _var_str(v, np.shape(value))
                            names[0] = models[h] + ' ' + names[0]
                            labels += names
                        else:
                            labels.append(models[h] + ' ' + v)

                    # Add spacing for each chain, if more than one
                    offset = [0] + [(chain_spacing * ((i + 2) / 2)) *
                                    (-1) ** i for i in range(nchains[h] - 1)]

                    # Y coordinate with offset
                    y = - var + offset[j]

                    # Deal with multivariate nodes

                    if k > 1:
                        qs = np.moveaxis(np.array(quants), 0, -1).squeeze()
                        for q in qs.reshape(-1, len(quants)):
                            # Multiple y values
                            interval_plot = _plot_tree(interval_plot, y, q,
                                                       quartiles, colors[h],
                                                       plot_kwargs)
                            y -= 1
                    else:
                        interval_plot = _plot_tree(interval_plot, y, quants,
                                                   quartiles, colors[h],
                                                   plot_kwargs)

                # Genenerate Gelman-Rubin plot
                if plot_rhat[h] and v in tr.varnames:
                    R = gelman_rubin(tr, [v])
                    if k > 1:
                        Rval = dict2pd(R, 'rhat').values
                        gr_plot.plot([min(r, 2) for r in Rval],
                                     [-(j + var) for j in range(k)], 'o',
                                     color=colors[h], markersize=4)
                    else:
                        gr_plot.plot(min(R[v], 2), -var, 'o', color=colors[h],
                                     markersize=4)
                var += k

        if len(trace) > 1:
            interval_plot.axhspan(var_old, y - chain_spacing - 0.5,
                                  facecolor='k', alpha=bands[v_idx])
            if np.any(plot_rhat):
                gr_plot.axhspan(var_old, y - chain_spacing - 0.5,
                                facecolor='k', alpha=bands[v_idx])
            var_old = y - chain_spacing - 0.5

    if ylabels is not None:
        labels = ylabels

    # Update margins
    left_margin = np.max([len(x) for x in labels]) * 0.015
    gs.update(left=left_margin, right=0.95, top=0.9, bottom=0.05)

    # Define range of y-axis for forestplot and R-hat
    interval_plot.set_ylim(- var + 0.5, 0.5)
    if np.any(plot_rhat):
        gr_plot.set_ylim(- var + 0.5, 0.5)

    plotrange = [np.min(all_quants), np.max(all_quants)]
    datarange = plotrange[1] - plotrange[0]
    interval_plot.set_xlim(plotrange[0] - 0.05 * datarange,
                           plotrange[1] + 0.05 * datarange)

    # Add variable labels
    interval_plot.set_yticks([- l for l in range(len(labels))])
    interval_plot.set_yticklabels(labels,
                                  fontsize=plot_kwargs.get('fontsize', None))

    # Add title
    if main is None:
        plot_title = "{:.0f}% Credible Intervals".format((1 - alpha) * 100)
    elif main:
        plot_title = main
    else:
        plot_title = ""

    interval_plot.set_title(plot_title,
                            fontsize=plot_kwargs.get('fontsize', None))

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

    return gs
