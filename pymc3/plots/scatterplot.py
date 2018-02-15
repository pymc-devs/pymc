import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:  # mpl is optional
    pass
import numpy as np
from .utils import get_default_varnames
from .artists import get_trace_dict, scale_text


def scatterplot(trace, varnames=None, figsize=None, text_size=None, 
                ax=None, hexbin=False, plot_transformed=False, divergences=False, 
                kwargs_divergence=None, sub_varnames=None, **kwargs):

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
    ax : axes
        Matplotlib axes.
    hexbin : Boolean
        If True draws an hexbin plot
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False). Applies when varnames/sub_varnames = None.
        When a list of varnames/sub_varnames is passed, transformed variables can be passed
        using their names.
    divergences : Boolean
        If True divergences will be plotted in a diferent color
    kwargs_divergence : dicts, optional
        Aditional keywords passed to ax.scatter for divergences
    sub_varnames : list, optional        
        Aditional varnames passed for plotting subsets of multidimensional variables
    Returns
    -------

    ax : matplotlib axes

    """    
    
    if varnames is None:
        if plot_transformed:
            trace_dict = get_trace_dict(trace, get_default_varnames(trace.varnames, True))
        else:
            trace_dict = get_trace_dict(trace, get_default_varnames(trace.varnames, False))
        if sub_varnames is None:
            varnames = list(trace_dict.keys())
        else:
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
        figsize = (8+numvars, 8+numvars)    

    if numvars < 2:
        raise Exception('Number of variables to be plotted must be 2 or greater.')
        
    if numvars == 2 and ax is not None:
        if hexbin:
            ax.hexbin(trace_dict[varnames[0]], trace_dict[varnames[1]], mincnt=1, **kwargs)
        else:
            ax.scatter(trace_dict[varnames[0]], trace_dict[varnames[1]], **kwargs)
        
        if divergences:
            try:
                divergent = trace['diverging']
                if np.any(divergent):
                    ax.scatter(trace_dict[varnames[0]][divergent == 1], trace_dict[varnames[1]][divergent == 1], **kwargs_divergence)
                else:
                    print('No divergences were found.')
            except KeyError:
                warnings.warn('There is no divergence information in the passed trace.')
                return ax

    if ax is None:
        _, ax = plt.subplots(nrows=numvars, ncols=numvars, figsize=figsize)

        for i in range(numvars):
            var1 = trace_dict[varnames[i]]            
            for j in range(i, numvars):
                var2 = trace_dict[varnames[j]]

                if i == j:
                    ax[i, j].axes.remove()

                else:
                    if hexbin:
                        ax[j, i].hexbin(var1, var2, mincnt=1, **kwargs)
                    else:
                        ax[j, i].scatter(var1, var2, **kwargs)
                    
                    if divergences:
                        try:
                            divergent = trace['diverging']
                            if np.any(divergent):
                                ax[j, i].scatter(var1[divergent == 1], var2[divergent == 1], **kwargs_divergence)
                            else:
                                print('No divergences were found.')
                        except KeyError:
                            warnings.warn('There is no divergence information in the passed trace.')
                            return ax
                        
                    ax[i, j].axes.remove()

                    if j != numvars-1:
                        ax[j, i].set_xticks([])
                    if i != 0:
                        ax[j, i].set_yticks([])

                ax[numvars-1, j].set_xlabel('{}'.format(varnames[i]), fontsize=text_size)
                ax[j, 0].set_ylabel('{}'.format(varnames[j]), fontsize=text_size)
                ax[numvars-1, j].tick_params(labelsize=text_size)
                ax[j, 0].tick_params(labelsize=text_size)
           
    plt.tight_layout()  
    return ax      