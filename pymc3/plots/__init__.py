"""PyMC3 Plotting.

Plots are delegated to the ArviZ library, a general purpose library for
"exploratory analysis of Bayesian models." See https://arviz-devs.github.io/arviz/
for details on plots.
"""
import functools
import sys
import warnings
try:
    import arviz as az
except ImportError:  # arviz is optional, throw exception when used

    class _ImportWarner:
        __all__ = []

        def __init__(self, attr):
            self.attr = attr

        def __call__(self, *args, **kwargs):
            raise ImportError(
                "ArviZ is not installed. In order to use `{0.attr}`:\npip install arviz".format(self)
            )

    class _ArviZ:
        def __getattr__(self, attr):
            return _ImportWarner(attr)


    az = _ArviZ()

def map_args(func):
    swaps = [
        ('varnames', 'var_names')
    ]
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for (old, new) in swaps:
            if old in kwargs and new not in kwargs:
                warnings.warn('Keyword argument `{old}` renamed to `{new}`, and will be removed in pymc3 3.8'.format(old=old, new=new))
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)
    return wrapped

# pymc3 custom plots: override these names for custom behavior
autocorrplot = map_args(az.plot_autocorr)
forestplot = map_args(az.plot_forest)
kdeplot = map_args(az.plot_kde)
plot_posterior = map_args(az.plot_posterior)
traceplot = map_args(az.plot_trace)
energyplot = map_args(az.plot_energy)
densityplot = map_args(az.plot_density)
pairplot = map_args(az.plot_pair)

# addition arg mapping for compare plot
@functools.wraps(az.plot_compare)
def compareplot(*args, **kwargs):
    if 'comp_df' in kwargs:
        comp_df = kwargs['comp_df'].copy()
    else:
        args = list(args)
        comp_df = args[0].copy()
    if 'WAIC' in comp_df.columns:
        comp_df = comp_df.rename(index=str,
                                 columns={'WAIC': 'waic',
                                          'pWAIC': 'p_waic',
                                          'dWAIC': 'd_waic',
                                          'SE': 'se',
                                          'dSE': 'dse',
                                          'var_warn': 'warning'})
    elif 'LOO' in comp_df.columns:
        comp_df = comp_df.rename(index=str,
                                 columns={'LOO': 'loo',
                                          'pLOO': 'p_loo',
                                          'dLOO': 'd_loo',
                                          'SE': 'se',
                                          'dSE': 'dse',
                                          'shape_warn': 'warning'})
    if 'comp_df' in kwargs:
        kwargs['comp_df'] = comp_df
    else:
        args[0] = comp_df
    return az.plot_compare(*args, **kwargs)

from .posteriorplot import plot_posterior_predictive_glm


# Access to arviz plots: base plots provided by arviz
for plot in az.plots.__all__:
    setattr(sys.modules[__name__], plot, map_args(getattr(az.plots, plot)))

__all__ = tuple(az.plots.__all__) + (
    'autocorrplot',
    'compareplot',
    'forestplot',
    'kdeplot',
    'plot_posterior',
    'traceplot',
    'energyplot',
    'densityplot',
    'pairplot',
    'plot_posterior_predictive_glm',
)
