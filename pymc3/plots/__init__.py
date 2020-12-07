#   Copyright 2020 The PyMC Developers
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

"""PyMC3 Plotting.

Plots are delegated to the ArviZ library, a general purpose library for
"exploratory analysis of Bayesian models." See https://arviz-devs.github.io/arviz/
for details on plots.
"""
import functools
import sys
import warnings

import arviz as az


def map_args(func):
    swaps = [("varnames", "var_names")]

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        for (old, new) in swaps:
            if old in kwargs and new not in kwargs:
                warnings.warn(
                    f"Keyword argument `{old}` renamed to `{new}`, and will be removed in pymc3 3.8"
                )
                kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

    return wrapped


# pymc3 custom plots: override these names for custom behavior
autocorrplot = map_args(az.plot_autocorr)
forestplot = map_args(az.plot_forest)
kdeplot = map_args(az.plot_kde)
plot_posterior = map_args(az.plot_posterior)
energyplot = map_args(az.plot_energy)
densityplot = map_args(az.plot_density)
pairplot = map_args(az.plot_pair)

# Use compact traceplot by default
@map_args
@functools.wraps(az.plot_trace)
def traceplot(*args, **kwargs):
    try:
        kwargs.setdefault("compact", True)
        return az.plot_trace(*args, **kwargs)
    except TypeError:
        kwargs.pop("compact")
        return az.plot_trace(*args, **kwargs)


# addition arg mapping for compare plot
@functools.wraps(az.plot_compare)
def compareplot(*args, **kwargs):
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


from .posteriorplot import plot_posterior_predictive_glm

# Access to arviz plots: base plots provided by arviz
for plot in az.plots.__all__:
    setattr(sys.modules[__name__], plot, map_args(getattr(az.plots, plot)))

__all__ = tuple(az.plots.__all__) + (
    "autocorrplot",
    "compareplot",
    "forestplot",
    "kdeplot",
    "plot_posterior",
    "traceplot",
    "energyplot",
    "densityplot",
    "pairplot",
    "plot_posterior_predictive_glm",
)
