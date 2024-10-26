#   Copyright 2024 The PyMC Developers
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


"""PyMC: Bayesian Modeling and Probabilistic Programming in Python."""

import logging

_log = logging.getLogger(__name__)

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


def __set_compiler_flags():
    # Workarounds for PyTensor compiler problems on various platforms
    import pytensor

    current = pytensor.config.gcc__cxxflags
    augmented = f"{current} -Wno-c++11-narrowing"

    # Work around compiler bug in GCC < 8.4 related to structured exception
    # handling registers on Windows.
    # See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782 for details.
    # First disable C++ exception handling altogether since it's not needed
    # for the C extensions that we generate.
    augmented = f"{augmented} -fno-exceptions"
    # Now disable the generation of stack unwinding tables.
    augmented = f"{augmented} -fno-unwind-tables -fno-asynchronous-unwind-tables"

    pytensor.config.gcc__cxxflags = augmented


__set_compiler_flags()

from pymc import _version, gp, ode, sampling
from pymc.backends import *
from pymc.blocking import *
from pymc.data import *
from pymc.distributions import *
from pymc.exceptions import *
from pymc.func_utils import find_constrained_prior
from pymc.logprob import *
from pymc.math import (
    expand_packed_triangular,
    invlogit,
    invprobit,
    logaddexp,
    logit,
    logsumexp,
    probit,
)
from pymc.model.core import *
from pymc.model.transform.conditioning import do, observe
from pymc.model_graph import model_to_graphviz, model_to_networkx
from pymc.plots import *
from pymc.printing import *
from pymc.pytensorf import *
from pymc.sampling import *
from pymc.smc import *
from pymc.stats import *
from pymc.step_methods import *
from pymc.tuning import *
from pymc.util import drop_warning_stat
from pymc.variational import *
from pymc.vartypes import *

__version__ = _version.get_versions()["version"]
