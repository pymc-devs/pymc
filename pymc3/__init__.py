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

# pylint: disable=wildcard-import
__version__ = "3.11.0"

import logging
import multiprocessing as mp
import platform

_log = logging.getLogger("pymc3")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


def __set_compiler_flags():
    # Workarounds for Theano compiler problems on various platforms
    import theano

    current = theano.config.gcc__cxxflags
    theano.config.gcc__cxxflags = f"{current} -Wno-c++11-narrowing"


__set_compiler_flags()

from pymc3 import gp, ode, sampling
from pymc3.backends import load_trace, save_trace
from pymc3.backends.tracetab import *
from pymc3.blocking import *
from pymc3.data import *
from pymc3.distributions import *
from pymc3.distributions import transforms
from pymc3.exceptions import *
from pymc3.glm import *
from pymc3.math import (
    expand_packed_triangular,
    invlogit,
    invprobit,
    logaddexp,
    logit,
    logsumexp,
    probit,
)
from pymc3.model import *
from pymc3.model_graph import model_to_graphviz
from pymc3.plots import *
from pymc3.sampling import *
from pymc3.smc import *
from pymc3.step_methods import *
from pymc3.tests import test
from pymc3.theanof import *
from pymc3.tuning import *
from pymc3.variational import *
from pymc3.vartypes import *
