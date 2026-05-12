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

"""Alias for the `plots` submodule from ArviZ.

Plots are delegated to the ArviZ library, a general purpose library for
"exploratory analysis of Bayesian models."
See https://arviz-devs.github.io/arviz/ for details on plots.
"""

import sys


def __getattr__(name):
    import arviz_plots as azp

    try:
        value = getattr(azp, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    setattr(sys.modules[__name__], name, value)
    return value


def __dir__():
    import arviz_plots as azp

    return [attr for attr in dir(azp) if not attr.startswith("_")]
