import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:  # mpl is optional
    pass
from .kdeplot import fast_kde
from .utils import get_default_varnames
from ..stats import hpd


def densityplot(trace, models=None, varnames=None, alpha=0.05, point_estimate='mean',
                colors='cycle', outline=True, hpd_markers='', shade=0., figsize=None, textsize=12,
                plot_transformed=False, ax=None):
    """
    Generates KDE plots truncated at their 100*(1-alpha)% credible intervals from a trace or list of
    traces. KDE plots are grouped per variable and colors assigned to models.

    Parameters
    ----------
    trace : trace or list of traces
        Trace(s) from an MCMC sample.
    models : list
        List with names for the models in the list of traces. Useful when
        plotting more that one trace. 
    varnames: list
        List of variables to plot (defaults to None, which results in all
        variables plotted).
    alpha : float
        Alpha value for (1-alpha)*100% credible intervals (defaults to 0.05).
    point_estimate : str or None
        Plot point estimate per variable. Values should be 'mean', 'median' or None.
        Defaults to 'mean'.
    colors : list or string, optional
        List with valid matplotlib colors, one color per model. Alternative a string can be passed.
        If the string is `cycle `, it will automatically choose a color per model from matplolib's
        cycle. If a single color is passed, e.g. 'k', 'C2' or 'red' this color will be used for all
        models. Defaults to 'C0' (blueish in most matplotlib styles)
    outline : boolean
        Use a line to draw the truncated KDE and. Defaults to True
    hpd_markers : str
        A valid `matplotlib.markers` like 'v', used to indicate the limits of the hpd interval.
        Defaults to empty string (no marker).
    shade : float
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    figsize : tuple
        Figure size. If None, size is (6, number of variables * 2)
    textsize : int
        Text size of the legend. Default 12.
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to original variables
        Defaults to False.
    ax : axes
        Matplotlib axes.

    Returns
    -------

    ax : Matplotlib axes

    """
    if point_estimate not in ('mean', 'median', None):
        raise ValueError("Point estimate should be 'mean' or 'median'")

    if not isinstance(trace, (list, tuple)):
        trace = [trace]

    lenght_trace = len(trace)

    if models is None:
        if lenght_trace > 1:
            models = ['m_{}'.format(i) for i in range(lenght_trace)]
        else:
            models = ['']
    elif len(models) != lenght_trace:
        raise ValueError("The number of names for the models does not match the number of models")

    lenght_models = len(models)

    if colors == 'cycle':
        colors = ['C{}'.format(i % 10) for i in range(lenght_models)]
    elif isinstance(colors, str):
        colors = [colors for i in range(lenght_models)]

    if varnames is None:
        varnames = []
        for tr in trace:
            varnames_tmp = get_default_varnames(tr.varnames, plot_transformed)
            for v in varnames_tmp:
                if v not in varnames:
                    varnames.append(v)

    if figsize is None:
        figsize = (6, len(varnames) * 2)

    fig, kplot = plt.subplots(len(varnames), 1, squeeze=False, figsize=figsize)
    kplot = kplot.flatten()

    for v_idx, vname in enumerate(varnames):
        for t_idx, tr in enumerate(trace):
            if vname in tr.varnames:
                vec = tr.get_values(vname)
                k = np.size(vec[0])
                if k > 1:
                    vec = np.split(vec.T.ravel(), k)
                    for i in range(k):
                        _kde_helper(vec[i], vname, colors[t_idx], alpha, point_estimate,
                                    hpd_markers, outline, shade, kplot[v_idx])
                else:
                    _kde_helper(vec, vname, colors[t_idx], alpha, point_estimate,
                                hpd_markers, outline, shade, kplot[v_idx])

    if lenght_trace > 1:
        for m_idx, m in enumerate(models):
            kplot[0].plot([], label=m, c=colors[m_idx])
        kplot[0].legend(fontsize=textsize)

    fig.tight_layout()

    return kplot


def _kde_helper(vec, vname, c, alpha, point_estimate, hpd_markers,
                outline, shade, ax):
    """
    vec : array
        1D array from trace
    vname : str
        variable name
    c : str
        matplotlib color
    alpha : float
        Alpha value for (1-alpha)*100% credible intervals (defaults to 0.05).
    point_estimate : str or None
        'mean' or 'median'
    shade : float
        Alpha blending value for the shaded area under the curve, between 0 (no shade) and 1
        (opaque). Defaults to 0.
    ax : matplotlib axes
    """
    density, l, u = fast_kde(vec)
    x = np.linspace(l, u, len(density))
    hpd_ = hpd(vec, alpha)
    cut = (x >= hpd_[0]) & (x <= hpd_[1])

    xmin = x[cut][0]
    xmax = x[cut][-1]
    ymin = density[cut][0]
    ymax = density[cut][-1]

    if outline:
        ax.plot(x[cut], density[cut], color=c)
        ax.plot([xmin, xmin], [-0.5, ymin], color=c, ls='-')
        ax.plot([xmax, xmax], [-0.5, ymax], color=c, ls='-')

    if hpd_markers:
        ax.plot(xmin, 0, 'v', color=c, markeredgecolor='k')
        ax.plot(xmax, 0, 'v', color=c, markeredgecolor='k')

    if shade:
        ax.fill_between(x, density, where=cut, color=c, alpha=shade)

    if point_estimate is not None:
        if point_estimate == 'mean':
            ps = np.mean(vec)
        if point_estimate == 'median':
            ps = np.median(vec)
        ax.plot(ps, 0, 'o', color=c, markeredgecolor='k')

    ax.set_yticks([])
    ax.set_title(vname)
    for pos in ['left', 'right', 'top']:
        ax.spines[pos].set_visible(0)
