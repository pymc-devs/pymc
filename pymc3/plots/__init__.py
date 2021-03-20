#   Copyright 2021 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Alias for the `plots` submodule from ArviZ.

Plots are delegated to the ArviZ library, a general purpose library for
"exploratory analysis of Bayesian models."
See https://arviz-devs.github.io/arviz/ for details on plots.
"""
import functools
import sys
import warnings

import arviz as az

# Makes this module as identical to arviz.plots as possible
for attr in az.plots.__all__:
    obj = getattr(az.plots, attr)
    if not attr.startswith("__"):
        setattr(sys.modules[__name__], attr, obj)


def map_args(func, alias: str):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if "varnames" in kwargs:
            raise DeprecationWarning(
                f"The `varnames` kwarg was renamed to `var_names`.", stacklevel=2
            )
        original = func.__name__
        warnings.warn(
            f"The function `{alias}` from PyMC3 is just an alias for `{original}` from ArviZ. "
            f"Please switch to `pymc3.{original}` or `arviz.{original}`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapped


# Always show the DeprecationWarnings
warnings.filterwarnings("once", category=DeprecationWarning, module="pymc3.plots")


# Aliases of ArviZ functions
autocorrplot = map_args(az.plot_autocorr, alias="autocorrplot")
forestplot = map_args(az.plot_forest, alias="forestplot")
kdeplot = map_args(az.plot_kde, alias="kdeplot")
energyplot = map_args(az.plot_energy, alias="energyplot")
densityplot = map_args(az.plot_density, alias="densityplot")
pairplot = map_args(az.plot_pair, alias="pairplot")
traceplot = map_args(az.plot_trace, alias="traceplot")


# Customized with kwarg reformatting
@functools.wraps(az.plot_compare)
def compareplot(*args, **kwargs):
    warnings.warn(
        f"The function `compareplot` from PyMC3 is an alias for `plot_compare` from ArviZ. "
        "It also applies some kwarg replacements. Nevertheless, please switch "
        f"to `pymc3.plot_compare` or `arviz.plot_compare`.",
        DeprecationWarning,
        stacklevel=2,
    )
    if "comp_df" in kwargs:
        comp_df = kwargs["comp_df"].copy()
    else:
        args = list(args)
        comp_df = args[0].copy()
    if "WAIC" in comp_df.columns:
        comp_df = comp_df.rename(
            index=str,
            columns={
                "WAIC": "waic",
                "pWAIC": "p_waic",
                "dWAIC": "d_waic",
                "SE": "se",
                "dSE": "dse",
                "var_warn": "warning",
            },
        )
    elif "LOO" in comp_df.columns:
        comp_df = comp_df.rename(
            index=str,
            columns={
                "LOO": "loo",
                "pLOO": "p_loo",
                "dLOO": "d_loo",
                "SE": "se",
                "dSE": "dse",
                "shape_warn": "warning",
            },
        )
    if "comp_df" in kwargs:
        kwargs["comp_df"] = comp_df
    else:
        args[0] = comp_df
    return az.plot_compare(*args, **kwargs)


from pymc3.plots.posteriorplot import plot_posterior_predictive_glm

__all__ = tuple(az.plots.__all__) + (
    "autocorrplot",
    "compareplot",
    "forestplot",
    "kdeplot",
    "traceplot",
    "energyplot",
    "densityplot",
    "pairplot",
    "plot_posterior_predictive_glm",
)
