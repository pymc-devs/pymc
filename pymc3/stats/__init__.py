"""Statistical utility functions for PyMC

Diagnostics and auxiliary statistical functions are delegated to the ArviZ library, a general
purpose library for "exploratory analysis of Bayesian models." See
https://arviz-devs.github.io/arviz/ for details.
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
                "ArviZ is not installed. In order to use `{0.attr}`:\npip install arviz".format(
                    self
                )
            )

    class _ArviZ:
        def __getattr__(self, attr):
            return _ImportWarner(attr)

    az = _ArviZ()


def map_args(func):
    swaps = [("varnames", "var_names")]

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for (old, new) in swaps:
            if old in kwargs and new not in kwargs:
                warnings.warn(
                    "Keyword argument `{old}` renamed to `{new}`, and will be removed in "
                    "pymc3 3.8".format(old=old, new=new)
                )
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

    return wrapped


__all__ = [
    "bfmi",
    "compare",
    "ess",
    "geweke",
    "hpd",
    "loo",
    "mcse",
    "r2_score",
    "rhat",
    "summary",
    "waic",
]

bfmi = map_args(az.bfmi)
compare = map_args(az.compare)
ess = map_args(az.ess)
geweke = map_args(az.geweke)
hpd = map_args(az.hpd)
loo = map_args(az.loo)
mcse = map_args(az.mcse)
r2_score = map_args(az.r2_score)
rhat = map_args(az.rhat)
summary = map_args(az.summary)
waic = map_args(az.waic)


def gelman_rubin(*args, **kwargs):
    warnings.warn("gelman_rubin has been deprecated. In future, use rhat instead.")
    return rhat(*args, **kwargs)


def effective_n(*args, **kwargs):
    warnings.warn("effective_n has been deprecated. In future, use ess instead.")
    return ess(*args, **kwargs)


# Access to arviz stats: base stats provided by arviz
for stat in az.stats.__all__:
    setattr(sys.modules[__name__], stat, map_args(getattr(az.stats, stat)))

__all__ = tuple(az.stats.__all__) + (
    "compare",
    "hpd",
    "loo",
    "r2_score",
    "summary",
    "waic",
    "bfmi",
    "ess",
    "geweke",
    "mcse",
    "rhat",
    "gelman_rubin",
    "effective_n",
)
