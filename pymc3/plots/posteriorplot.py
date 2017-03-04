from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from .artists import plot_posterior_op
from .utils import identity_transform, get_default_varnames


def plot_posterior(trace, varnames=None, transform=identity_transform, figsize=None,
                   alpha_level=0.05, round_to=3, point_estimate='mean', rope=None,
                   ref_val=None, kde_plot=False, plot_transformed=False, ax=None, **kwargs):
    """Plot Posterior densities in style of John K. Kruschke book.

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
        plot_posterior_op(transform(trace), ax=ax, kde_plot=kde_plot,
                          point_estimate=point_estimate, round_to=round_to,
                          alpha_level=alpha_level, ref_val=ref_val, rope=rope, **kwargs)
    else:
        if varnames is None:
            varnames = get_default_varnames(trace, plot_transformed)

        trace_dict = get_trace_dict(trace, varnames)

        if ax is None:
            ax, fig = create_axes_grid(figsize, trace_dict)

        for a, v in zip(np.atleast_1d(ax), trace_dict):
            tr_values = transform(trace_dict[v])
            plot_posterior_op(tr_values, ax=a, kde_plot=kde_plot,
                              point_estimate=point_estimate, round_to=round_to,
                              alpha_level=alpha_level, ref_val=ref_val, rope=rope, **kwargs)
            a.set_title(v)

        plt.tight_layout()
    return ax
