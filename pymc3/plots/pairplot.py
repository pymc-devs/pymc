import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
except ImportError:  # mpl is optional
    pass
from ..util import get_default_varnames, is_transformed_name, get_untransformed_name
from .artists import get_trace_dict, scale_text


def pairplot(trace, varnames=None, figsize=None, text_size=None,
             gs=None, ax=None, hexbin=False, plot_transformed=False,
             divergences=False, kwargs_divergence=None,
             sub_varnames=None, **kwargs):
    """
    Plot a scatter or hexbin matrix of the sampled parameters.

    Parameters
    ----------

    trace : result of MCMC run
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    figsize : figure size tuple
        If None, size is (8 + numvars, 8 + numvars)
    text_size: int
        Text size for labels
    gs : Grid spec
        Matplotlib Grid spec.
    ax: axes
        Matplotlib axes
    hexbin : Boolean
        If True draws an hexbin plot
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False). Applies when varnames = None.
        When a list of varnames is passed, transformed variables can be passed
        using their names.
    divergences : Boolean
        If True divergences will be plotted in a diferent color
    kwargs_divergence : dicts, optional
        Aditional keywords passed to ax.scatter for divergences
    sub_varnames : list
        Aditional varnames passed for plotting subsets of multidimensional
        variables
    Returns
    -------

    ax : matplotlib axes
    gs : matplotlib gridspec

    """
    if varnames is None:
        if plot_transformed:

            varnames_copy = list(trace.varnames)
            remove = [get_untransformed_name(var) for var in trace.varnames
                      if is_transformed_name(var)]

            try:
                [varnames_copy.remove(i) for i in remove]
                varnames = varnames_copy
            except ValueError:
                varnames = varnames_copy

            trace_dict = get_trace_dict(
                trace, get_default_varnames(
                    varnames, plot_transformed))

        else:
            trace_dict = get_trace_dict(
                trace, get_default_varnames(
                    trace.varnames, plot_transformed))

        if sub_varnames is None:
            varnames = list(trace_dict.keys())

        else:
            trace_dict = get_trace_dict(
                trace, get_default_varnames(
                    trace.varnames, True))
            varnames = sub_varnames

    else:
        trace_dict = get_trace_dict(trace, varnames)
        varnames = list(trace_dict.keys())

    if text_size is None:
        text_size = scale_text(figsize, text_size=text_size)

    if kwargs_divergence is None:
        kwargs_divergence = {}

    numvars = len(varnames)

    if figsize is None:
        figsize = (8 + numvars, 8 + numvars)

    if numvars < 2:
        raise Exception(
            'Number of variables to be plotted must be 2 or greater.')

    if numvars == 2 and ax is not None:
        if hexbin:
            ax.hexbin(trace_dict[varnames[0]],
                      trace_dict[varnames[1]], mincnt=1, **kwargs)
        else:
            ax.scatter(trace_dict[varnames[0]],
                       trace_dict[varnames[1]], **kwargs)

        if divergences:
            try:
                divergent = trace['diverging']
            except KeyError:
                warnings.warn('No divergences were found.')

            diverge = (divergent == 1)
            ax.scatter(trace_dict[varnames[0]][diverge],
                       trace_dict[varnames[1]][diverge], **kwargs_divergence)
        ax.set_xlabel('{}'.format(varnames[0]),
                      fontsize=text_size)
        ax.set_ylabel('{}'.format(
            varnames[1]), fontsize=text_size)
        ax.tick_params(labelsize=text_size)

    if gs is None and ax is None:
        plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(numvars - 1, numvars - 1)

        for i in range(0, numvars - 1):
            var1 = trace_dict[varnames[i]]

            for j in range(i, numvars - 1):
                var2 = trace_dict[varnames[j + 1]]

                ax = plt.subplot(gs[j, i])

                if hexbin:
                    ax.hexbin(var1, var2, mincnt=1, **kwargs)
                else:
                    ax.scatter(var1, var2, **kwargs)

                if divergences:
                    try:
                        divergent = trace['diverging']
                    except KeyError:
                        warnings.warn('No divergences were found.')
                        return ax

                    diverge = (divergent == 1)
                    ax.scatter(var1[diverge],
                               var2[diverge],
                               **kwargs_divergence)

                if j + 1 != numvars - 1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('{}'.format(varnames[i]),
                                  fontsize=text_size)
                if i != 0:
                    ax.set_yticks([])
                else:
                    ax.set_ylabel('{}'.format(
                        varnames[j + 1]), fontsize=text_size)

                ax.tick_params(labelsize=text_size)

    plt.tight_layout()
    return ax, gs
