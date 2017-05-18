import matplotlib.pyplot as plt
import numpy as np

from .kdeplot import kdeplot


def energyplot(trace, kind='kde', figsize=None, ax=None, legend=True, lw=0,
               alpha=0.35, frame=True, **kwargs):
    """Plot energy transition distribution and marginal energy distribution in order
    to diagnose poor exploration by HMC algorithms.

    Parameters
    ----------

    trace : result of MCMC run
    kind : str
        Type of plot to display (kde or histogram)
    figsize : figure size tuple
        If None, size is (8 x 6)
    ax : axes
        Matplotlib axes.
    legend : bool
        Flag for plotting legend (defaults to True)
    lw : int
        Line width
    alpha : float
        Alpha value for plot line. Defaults to 0.35.
    frame : bool
        Flag for plotting frame around figure.

    Returns
    -------

    ax : matplotlib axes
    """

    try:
        energy = trace['energy']
    except KeyError:
        print('There is no energy information in the passed trace.')
        return ax
    series = [('Marginal energy distribution', energy - energy.mean()),
              ('Energy transition distribution', np.diff(energy))]

    if figsize is None:
        figsize = (8, 6)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if kind == 'kde':
        for label, value in series:
            kdeplot(value, label=label, alpha=alpha, shade=True, ax=ax,
                    **kwargs)

    elif kind == 'hist':
        for label, value in series:
            ax.hist(value, lw=lw, alpha=alpha, label=label, **kwargs)

    else:
        raise ValueError('Plot type {} not recognized.'.format(kind))

    ax.set_xticks([])
    ax.set_yticks([])

    if not frame:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if legend:
        ax.legend()

    return ax
