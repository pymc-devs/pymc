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

Plots are delegated to the `ArviZ <https://arviz-devs.github.io/arviz/>`_ library, a general purpose library for
exploratory analysis of Bayesian models. For more details, see https://arviz-devs.github.io/arviz/.

Only `plot_posterior_predictive_glm` is kept in the PyMC code base for now, but it will move to ArviZ once the latter adds features for regression plots.
"""
import functools
import sys
import warnings

import arviz as az

from pymc3.plots.posteriorplot import plot_posterior_predictive_glm

__all__ = ["plot_posterior_predictive_glm"]
