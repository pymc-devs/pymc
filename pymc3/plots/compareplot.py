import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:  # mpl is optional
    pass


def compareplot(comp_df, insample_dev=True, se=True, dse=True, ax=None,
                plot_kwargs=None):
    """
    Model comparison summary plot in the style of the one used in the book
    Statistical Rethinking by Richard McElreath.

    Parameters
    ----------

    comp_df: DataFrame
        the result of the `pm.compare()` function
    insample_dev : bool
        plot the in-sample deviance, that is the value of the IC without the
        penalization given by the effective number of parameters (pIC).
        Defaults to True
    se : bool
        plot the standard error of the IC estimate. Defaults to True
    dse : bool
        plot standard error of the difference in IC between each model and the
        top-ranked model. Defaults to True
    plot_kwargs : dict
        Optional arguments for plot elements. Currently accepts 'color_ic',
        'marker_ic', 'color_insample_dev', 'marker_insample_dev', 'color_dse',
        'marker_dse', 'ls_min_ic' 'color_ls_min_ic',  'fontsize'
    ax : axes
        Matplotlib axes. Defaults to None

    Returns
    -------

    ax : matplotlib axes

    """
    if ax is None:
        _, ax = plt.subplots()

    if plot_kwargs is None:
        plot_kwargs = {}

    yticks_pos, step = np.linspace(0, -1, (comp_df.shape[0] * 2) - 1,
                                   retstep=True)
    yticks_pos[1::2] = yticks_pos[1::2] + step / 2

    yticks_labels = [''] * len(yticks_pos)
    
    ic = 'WAIC'
    if ic not in comp_df.columns:
        ic = 'LOO'

    if dse:
        yticks_labels[0] = comp_df.index[0]
        yticks_labels[2::2] = comp_df.index[1:]
        ax.set_yticks(yticks_pos)
        ax.errorbar(x=comp_df[ic].iloc[1:],
                    y=yticks_pos[1::2],
                    xerr=comp_df.dSE[1:],
                    color=plot_kwargs.get('color_dse', 'grey'),
                    fmt=plot_kwargs.get('marker_dse', '^'))

    else:
        yticks_labels = comp_df.index
        ax.set_yticks(yticks_pos[::2])

    if se:
        ax.errorbar(x=comp_df[ic],
                    y=yticks_pos[::2],
                    xerr=comp_df.SE,
                    color=plot_kwargs.get('color_ic', 'k'),
                    fmt=plot_kwargs.get('marker_ic', 'o'),
                    mfc='None',
                    mew=1)
    else:
        ax.plot(comp_df[ic],
                yticks_pos[::2],
                color=plot_kwargs.get('color_ic', 'k'),
                marker=plot_kwargs.get('marker_ic', 'o'),
                mfc='None',
                mew=1,
                lw=0)

    if insample_dev:
        ax.plot(comp_df[ic] - (2 * comp_df['p'+ic]),
                yticks_pos[::2],
                color=plot_kwargs.get('color_insample_dev', 'k'),
                marker=plot_kwargs.get('marker_insample_dev', 'o'),
                lw=0)

    ax.axvline(comp_df[ic].iloc[0],
               ls=plot_kwargs.get('ls_min_ic', '--'),
               color=plot_kwargs.get('color_ls_min_ic', 'grey'))

    ax.set_xlabel('Deviance', fontsize=plot_kwargs.get('fontsize', 14))
    ax.set_yticklabels(yticks_labels)
    ax.set_ylim(-1 + step, 0 - step)

    return ax
