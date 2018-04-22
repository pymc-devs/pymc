try:
    import matplotlib.pyplot as plt
except ImportError:  # mpl is optional
    pass
import numpy as np

from .artists import plot_posterior_op, get_trace_dict, scale_text

from .utils import identity_transform, get_default_varnames


def plot_posterior(trace, varnames=None, transform=identity_transform, figsize=None, text_size=None,
                   alpha_level=0.05, round_to=3, point_estimate='mean', rope=None,
                   ref_val=None, kde_plot=False, plot_transformed=False, bw=4.5, ax=None, **kwargs):
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
    text_size : int
        Text size of the point_estimates, axis ticks, and HPD (Default:16)
    alpha_level : float
        Defines range for High Posterior Density
    round_to : int
        Controls formatting for floating point numbers
    point_estimate: str
        Must be in ('mode', 'mean', 'median')
    rope: list or numpy array
        Lower and upper values of the Region Of Practical Equivalence
    ref_val: float or list-like
        display the percentage below and above the values in ref_val.
        If a list is provided, its length should match the number of variables.
    kde_plot: bool
        if True plot a KDE instead of a histogram. For discrete variables this
        argument is ignored.
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy). Only works if `kde_plot` is True.
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
        l_trace = len(traces)
        if l_trace == 1:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            n = np.ceil(l_trace / 2.0).astype(int)
            if figsize is None:
                figsize = (12, n * 2.5)
            fig, ax = plt.subplots(n, 2, figsize=figsize)
            ax = ax.reshape(2 * n)
            if l_trace % 2 == 1:
                ax[-1].set_axis_off()
                ax = ax[:-1]
        return fig, ax

    if isinstance(trace, np.ndarray):
        if figsize is None:
            figsize = (6, 2)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)


        plot_posterior_op(transform(trace), ax=ax, bw=bw, kde_plot=kde_plot,
                          point_estimate=point_estimate, round_to=round_to, alpha_level=alpha_level,
                          ref_val=ref_val, rope=rope, text_size=scale_text(figsize, text_size), **kwargs)

    else:
        if varnames is None:
            varnames = get_default_varnames(trace.varnames, plot_transformed)

        trace_dict = get_trace_dict(trace, varnames)

        if ax is None:
            fig, ax = create_axes_grid(figsize, trace_dict)

        var_num = len(trace_dict)
        if ref_val is None:
            ref_val = [None] * var_num
        elif np.isscalar(ref_val):
            ref_val = [ref_val for _ in range(var_num)]

        if rope is None:
            rope = [None] * var_num
        elif np.ndim(rope) == 1:
            rope = [rope] * var_num

        for idx, (a, v) in enumerate(zip(np.atleast_1d(ax), trace_dict)):
            tr_values = transform(trace_dict[v])
            plot_posterior_op(tr_values, ax=a, bw=bw, kde_plot=kde_plot,
                              point_estimate=point_estimate, round_to=round_to,
                              alpha_level=alpha_level, ref_val=ref_val[idx],
                              rope=rope[idx], text_size=scale_text(figsize, text_size), **kwargs)
            a.set_title(v, fontsize=scale_text(figsize, text_size))

        plt.tight_layout()
    return ax


def plot_posterior_predictive_glm(trace, eval=None, lm=None, samples=30, **kwargs):
    """Plot posterior predictive of a linear model.
    :Arguments:
        trace : <array>
            Array of posterior samples with columns
        eval : <array>
            Array over which to evaluate lm
        lm : function <default: linear function>
            Function mapping parameters at different points
            to their respective outputs.
            input: point, sample
            output: estimated value
        samples : int <default=30>
            How many posterior samples to draw.
    Additional keyword arguments are passed to pylab.plot().
    """
    if lm is None:
        lm = lambda x, sample: sample['Intercept'] + sample['x'] * x

    if eval is None:
        eval = np.linspace(0, 1, 100)

    # Set default plotting arguments
    if 'lw' not in kwargs and 'linewidth' not in kwargs:
        kwargs['lw'] = .2
    if 'c' not in kwargs and 'color' not in kwargs:
        kwargs['c'] = 'k'

    for rand_loc in np.random.randint(0, len(trace), samples):
        rand_sample = trace[rand_loc]
        plt.plot(eval, lm(eval, rand_sample), **kwargs)
    # Make sure to not plot label multiple times
        kwargs.pop('label', None)

    plt.title('Posterior predictive')
