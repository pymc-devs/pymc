import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:  # mpl is optional
    pass
from .kdeplot import fast_kde
from .utils import get_default_varnames
from ..stats import hpd


def densityplot(trace, models=None, varnames=None, alpha=0.05,
                point_estimate='mean', colors='cycle', opacity=0.75,
                figsize=None, textsize=12, plot_transformed=False, ax=None):
    """
    Generates KDE plots truncated at their 100*(1-alpha) % credible intervals
    from a trace or list of traces. KDE plots are grouped per variable and
    different colors assigned to models.

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
        Plot a point estimate per variable. Values should be 'mean' (default),
        'median' or None. Defaults to 'mean'.
    colors : list or string, optional
        list with valid matplotlib colors, one color per model. Alternative a
        string can be passed. If the string is `cycle `, it will automatically
        chose a color per model from the matyplolib's cycle. If a single color
        is passed, eg 'k', 'C2', 'red' this color will be used for all models.
        Defauls to 'C0' (blueish in most matplotlib styles)
    opacity : float
        opacity level for the density plots, opaqueness increase from 0 to 1.
        Default value is 0.75.
    figsize : tuple
        Figure size. If None, size is (6, number of variables * 2)
    textsize : int
        Text size of the lenged. Default 12
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).
    ax : axes
        Matplotlib axes.

    Returns
    -------

    ax : Matplotlib axes

    """
    if point_estimate not in ('mean', 'median', None):
        raise ValueError("Point Estimate should be 'mean' or 'median'")

    if not isinstance(trace, (list, tuple)):
        trace = [trace]

    lenght_trace = len(trace)

    if models is None:
        if lenght_trace > 1:
            models = ['m_{}'.format(i) for i in range(lenght_trace)]
        else:
            models = ['']
    elif len(models) != lenght_trace:
        raise ValueError("The number of names for the models does not "
                         "match the number of models")

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
                        _kde_helper(vec[i], vname, colors[t_idx], alpha,
                                    point_estimate, opacity, kplot[v_idx])
                else:
                    _kde_helper(vec, vname, colors[t_idx], alpha,
                                point_estimate, opacity, kplot[v_idx])
    if lenght_trace > 1:
        for m_idx, m in enumerate(models):
            kplot[0].plot([], label=m, c=colors[m_idx])
        kplot[0].legend(fontsize=textsize)

    fig.tight_layout()

    return kplot


def _kde_helper(vec, vname, c, alpha, point_estimate, opacity, ax):
    """
    Helper function to plot truncated kde plots with point estimates.

    Parameters
    ----------

    vec : array
        1D array from trace
    vname : str
        variable name
    c : str
        matplotlib color
    point_estimate : str or None
        'mean' or 'median'
    opacity : float
        0 to 1 value controling the opacity of the kdeplot
    ax : matplotlib axes
    """
    density, l, u = fast_kde(vec)
    x = np.linspace(l, u, len(density))
    hpd_ = hpd(vec, alpha)
    cut = (x > hpd_[0]) & (x < hpd_[1])
    ax.fill_between(x[cut], density[cut], color=c, alpha=opacity)

    if point_estimate is not None:
        if point_estimate == 'mean':
            ps = np.mean(vec)
        if point_estimate == 'median':
            ps = np.median(vec)
        ax.plot(ps, 0, 'o', color=c, markeredgecolor='k')

    ax.set_yticks([])
    ax.set_title(vname)
    for pos in ['left', 'right', 'top']:
        ax.spines[pos].set_visible(False)
