#   Copyright 2024 - present The PyMC Developers
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

"""Alias for the `stats` submodule from ArviZ.

Diagnostics and auxiliary statistical functions are delegated to the ArviZ library, a general
purpose library for "exploratory analysis of Bayesian models."
See https://arviz-devs.github.io/arviz/ for details.
"""

import sys

from pymc.stats.log_density import compute_log_likelihood, compute_log_prior

__all__ = ("compute_log_likelihood", "compute_log_prior")


def __getattr__(name):
    import arviz_stats as azs

    try:
        value = getattr(azs, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    setattr(sys.modules[__name__], name, value)
    return value


def __dir__():
    import arviz_stats as azs

    own = ["compute_log_likelihood", "compute_log_prior"]
    return own + [attr for attr in dir(azs) if not attr.startswith("_")]
