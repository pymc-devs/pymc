#   Copyright 2023 The PyMC Developers
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


def alias_deprecation(func, alias: str):
    original = func.__name__

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        raise FutureWarning(
            f"The function `{alias}` from PyMC was an alias for `{original}` from ArviZ. "
            "It was removed in PyMC 4.0. "
            f"Switch to `pymc.{original}` or `arviz.{original}`."
        )

    return wrapped


# Aliases of ArviZ functions
autocorrplot = alias_deprecation(az.plot_autocorr, alias="autocorrplot")
forestplot = alias_deprecation(az.plot_forest, alias="forestplot")
kdeplot = alias_deprecation(az.plot_kde, alias="kdeplot")
energyplot = alias_deprecation(az.plot_energy, alias="energyplot")
densityplot = alias_deprecation(az.plot_density, alias="densityplot")
pairplot = alias_deprecation(az.plot_pair, alias="pairplot")
traceplot = alias_deprecation(az.plot_trace, alias="traceplot")
compareplot = alias_deprecation(az.plot_compare, alias="compareplot")


__all__ = tuple(az.plots.__all__) + (
    "autocorrplot",
    "compareplot",
    "forestplot",
    "kdeplot",
    "traceplot",
    "energyplot",
    "densityplot",
    "pairplot",
)
