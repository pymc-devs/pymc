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

import arviz as az

for attr in az.stats.__all__:
    obj = getattr(az.stats, attr)
    if not attr.startswith("__"):
        setattr(sys.modules[__name__], attr, obj)

from pymc.stats.log_density import compute_log_likelihood, compute_log_prior

__all__ = ("compute_log_likelihood", "compute_log_prior", *az.stats.__all__)
