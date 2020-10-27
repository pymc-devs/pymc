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
