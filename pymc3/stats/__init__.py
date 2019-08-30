"""Statistical utility functions for PyMC3

Diagnostics and auxiliary statistical functions are delegated to the ArviZ library, a general
purpose library for "exploratory analysis of Bayesian models." See
https://arviz-devs.github.io/arviz/ for details.
"""
import functools
import warnings

import arviz as az

def map_args(func):
    swaps = [("varnames", "var_names")]

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for (old, new) in swaps:
            if old in kwargs and new not in kwargs:
                warnings.warn(
                    "Keyword argument `{old}` renamed to `{new}`, and will be removed in "
                    "pymc3 3.9".format(old=old, new=new)
                )
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

    return wrapped


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
    warnings.warn("gelman_rubin has been deprecated. In the future, use rhat instead.")
    return rhat(*args, **kwargs)

gelman_rubin.__doc__ = rhat.__doc__


def effective_n(*args, **kwargs):
    warnings.warn("effective_n has been deprecated. In the future, use ess instead.")
    return ess(*args, **kwargs)

effective_n.__doc__ = ess.__doc__

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
    "gelman_rubin",  # deprecated, remove after 3.8
    "effective_n",  # deprecated, remove after 3.8
]
