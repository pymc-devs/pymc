import matplotlib.pyplot as plt
import numpy as np

from .utils import fast_kde

def energyplot(trace, kind='kde', figsize=None, ax=None, legend=True, lw=0, alpha=0.5, frame=False, **kwargs):
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
    
    energy = trace['energy']
    series_dict = {'Energy': energy - energy.mean(),
                'Energy difference': np.diff(energy)}

    if figsize is None:
        figsize = (8, 6)
        
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if kind == 'kde':
        for series in series_dict:
            density, l, u = fast_kde(series_dict[series])
            x = np.linspace(l, u, len(density))
            ax.plot(x, density, label=series, **kwargs)
            ax.fill_between(x, density, alpha=alpha)
            
    elif kind == 'hist':
        for series in series_dict:
            ax.hist(series_dict[series], lw=lw, alpha=alpha, label=series, **kwargs)
            
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
