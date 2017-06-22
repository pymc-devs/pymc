import itertools
import matplotlib.pyplot as plt
import numpy as np

from .utils import get_default_varnames, get_axis


def autocorrplot(trace, varnames=None, max_lag=100, burn=0, plot_transformed=False,
                 symmetric_plot=False, ax=None, figsize=None):
    """Bar plot of the autocorrelation function for a trace.

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
        varnames = get_default_varnames(trace.varnames, plot_transformed)

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
