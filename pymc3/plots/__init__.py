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

autocorrplot = map_args(az.plot_autocorr)
compareplot = map_args(az.plot_compare)
forestplot = map_args(az.plot_forest)
kdeplot = map_args(az.plot_kde)
plot_posterior = map_args(az.plot_posterior)
traceplot = map_args(az.plot_trace)
energyplot = map_args(az.plot_energy)
densityplot = map_args(az.plot_density)
pairplot = map_args(az.plot_pair)

from .posteriorplot import plot_posterior_predictive_glm


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
