import import plot as plt, subplots
try:
    import matplotlib.gridspec as gridspec
except ImportError:
    gridspec = None
import numpy as np
from scipy.stats import kde
from .stats import *

__all__ = ['traceplot', 'kdeplot', 'kde2plot', 'forestplot']


def traceplot(trace, vars=None):
    if vars is None:
        vars = trace.samples.keys()

    n = len(vars)
    f, ax = plt.subplots(n, 2, squeeze=False)

    for i, v in enumerate(vars):
        d = np.squeeze(trace[v])

        kdeplot_op(ax[i, 0], d)
        ax[i, 0].set_title(str(v))
        ax[i, 1].plot(d, alpha=.35)

        ax[i, 0].set_ylabel("frequency")
        ax[i, 1].set_ylabel("sample value")

    return f


def kdeplot_op(ax, data):
    data = np.atleast_2d(data.T).T
    for i in range(data.shape[1]):
        d = data[:, i]
        density = kde.gaussian_kde(d)
        l = np.min(d)
        u = np.max(d)
        x = np.linspace(0, 1, 100) * (u - l) + l

        ax.plot(x, density(x))


def kde2plot_op(ax, x, y, grid=200):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    grid = grid * 1j
    X, Y = np.mgrid[xmin:xmax:grid, ymin:ymax:grid]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = kde.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
              extent=[xmin, xmax, ymin, ymax])


def kdeplot(data):
    f, ax = plt.subplots(1, 1, squeeze=True)
    kdeplot_op(ax, data)
    return f


def kde2plot(x, y, grid=200):
    f, ax = plt.subplots(1, 1, squeeze=True)
    kde2plot_op(ax, x, y, grid)
    return f


def forestplot(trace_obj, vars=None, alpha=0.05,  quartiles=True, hpd=True, rhat=True,
    main=None, xtitle=None, xrange=None, labels=None, spacing=0.05, vline=0):
    """ Forest plot (model summary plot)

    Generates a "forest plot" of 100*(1-alpha)% credible intervals for either the
    set of variables in a given model, or a specified set of nodes.

    :Arguments:
        trace_obj: NpTrace or MultiTrace object
            Trace(s) from an MCMC sample.

        vars: list
            List of variables to plot (defaults to None, which results in all
            variables plotted).

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

        xtitle (optional): string
            Label for x-axis. Defaults to no label

        xrange (optional): list or tuple
            Range for x-axis. Defaults to matplotlib's best guess.

        labels (optional): list
            User-defined labels for each variable. If not provided, the node
            __name__ attributes are used.

        spacing (optional): float
            Plot spacing between chains (defaults to 0.05).

        vline (optional): numeric
            Location of vertical reference line (defaults to 0).

    """

    if not gridspec:
        print_('\nYour installation of matplotlib is not recent enough to support summary_plot; this function is disabled until matplotlib is updated.')
        return

    # Quantiles to be calculated
    quantiles = [100*alpha/2, 50, 100*(1-alpha/2)]
    if quartiles:
        quantiles = [100*alpha/2, 25, 50, 75, 100*(1-alpha/2)]

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
        # First try MultiTrace type
        traces = trace_obj.traces

        if rhat and len(traces) > 1:

            from .diagnostics import gelman_rubin

            R = gelman_rubin(trace_obj)
            if vars is not None:
                R = {v: R[v] for v in vars}

        else:
            print 'Could not calculate Gelman-Rubin statistics. Requires multiple chains \
                of equal length.'
            rhat = False


    except AttributeError:

        # Single NpTrace
        traces = [trace_obj]

        # Can't calculate Gelman-Rubin with a single trace
        rhat = False


    if vars is None:
        vars = traces[0].varnames

    # Empty list for y-axis labels
    labels = []
    # Counter for current variable
    var = 1

    for varname in vars:

        chains = len(traces)

        if gs is None:
            # Initialize plot
            if rhat and chains>1:
                gs = gridspec.GridSpec(1, 2, width_ratios=[3,1])

            else:

                gs = gridspec.GridSpec(1, 1)

            # Subplot for confidence intervals
            interval_plot = subplot(gs[0])

        # Get quantiles
        data = [calc_quantiles(d, quantiles) for d in traces]
        if hpd:
            # Substitute HPD interval
            for i,d in enumerate(traces):
                hpd_interval = hpd(d, alpha).T
                data[i][quantiles[0]] = hpd_interval[0]
                data[i][quantiles[-1]] = hpd_interval[1]

        data = [[d[q] for q in quantiles] for d in data]
        # Ensure x-axis contains range of current interval
        if plotrange:
            plotrange = [min(plotrange[0], nmin(data)), max(plotrange[1], nmax(data))]
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
        if k>1:
            names = var_str(varname, shape(value))
            labels += names
        else:
            labels.append(varname)
            #labels.append('\n'.join(varname.split('_')))

        # Add spacing for each chain, if more than one
        e = [0] + [(chain_spacing * ((i+2)/2))*(-1)**i for i in range(chains-1)]

        # Loop over chains
        for j,quants in enumerate(data):

            # Deal with multivariate nodes
            if k>1:

                for i,q in enumerate(transpose(quants)):

                    # Y coordinate with jitter
                    y = -(var+i) + e[j]

                    if quartiles:
                        # Plot median
                        plt.plot(q[2], y, 'bo', markersize=4)
                        # Plot quartile interval
                        errorbar(x=(q[1],q[3]), y=(y,y), linewidth=2, color="blue")

                    else:
                        # Plot median
                        plt.plot(q[1], y, 'bo', markersize=4)

                    # Plot outer interval
                    errorbar(x=(q[0],q[-1]), y=(y,y), linewidth=1, color="blue")

            else:

                # Y coordinate with jitter
                y = -var + e[j]

                if quartiles:
                    # Plot median
                    plt.plot(quants[2], y, 'bo', markersize=4)
                    # Plot quartile interval
                    errorbar(x=(quants[1],quants[3]), y=(y,y), linewidth=2, color="blue")
                else:
                    # Plot median
                    plt.plot(quants[1], y, 'bo', markersize=4)

                # Plot outer interval
                errorbar(x=(quants[0],quants[-1]), y=(y,y), linewidth=1, color="blue")

        # Increment index
        var += k

    if custom_labels is not None:
        labels = custom_labels

    # Update margins
    left_margin = max([len(x) for x in labels])*0.015
    gs.update(left=left_margin, right=0.95, top=0.9, bottom=0.05)

    # Define range of y-axis
    ylim(-var+0.5, -0.5)

    datarange = plotrange[1] - plotrange[0]
    xlim(plotrange[0] - 0.05*datarange, plotrange[1] + 0.05*datarange)

    # Add variable labels
    yticks([-(l+1) for l in range(len(labels))], labels)

    # Add title
    if main is not False:
        plot_title = main or str(int((1-alpha)*100)) + "% Credible Intervals"
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
        if loc in ['bottom','top']:
            pass
            #spine.set_position(('outward',10)) # outward by 10 points
        elif loc in ['left','right']:
            spine.set_color('none') # don't draw spine

    # Reference line
    axvline(vline_pos, color='k', linestyle='--')

    # Genenerate Gelman-Rubin plot
    if rhat and chains>1:

        # If there are multiple chains, calculate R-hat
        rhat_plot = subplot(gs[1])

        if main is not False:
            title("R-hat")

        # Set x range
        xlim(0.9,2.1)

        # X axis labels
        xticks((1.0,1.5,2.0), ("1", "1.5", "2+"))
        yticks([-(l+1) for l in range(len(labels))], "")

        i = 1
        for variable in vars:

            if variable._plot==False:
                continue

            # Extract name
            varname = variable.__name__

            try:
                value = variable.get_stoch_value()
            except AttributeError:
                value = variable.value

            k = size(value)

            if k>1:
                plt.plot([min(r, 2) for r in R[varname]], [-(j+i) for j in range(k)], 'bo', markersize=4)
            else:
                plt.plot(min(R[varname], 2), -i, 'bo', markersize=4)

            i += k

        # Define range of y-axis
        ylim(-i+0.5, -0.5)

        # Remove ticklines on y-axes
        for ticks in rhat_plot.yaxis.get_major_ticks():
            ticks.tick1On = False
            ticks.tick2On = False

        for loc, spine in six.iteritems(rhat_plot.spines):
            if loc in ['bottom','top']:
                pass
                #spine.set_position(('outward',10)) # outward by 10 points
            elif loc in ['left','right']:
                spine.set_color('none') # don't draw spine

    return fig