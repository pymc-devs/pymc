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
__version__ = "3.10.0"

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

    current = theano.config.gcc.cxxflags
    theano.config.gcc.cxxflags = f"{current} -Wno-c++11-narrowing"


__set_compiler_flags()

from . import gp, ode, sampling
from .backends import load_trace, save_trace
from .backends.tracetab import *
from .blocking import *
from .data import *
from .distributions import *
from .distributions import transforms
from .exceptions import *
from .glm import *
from .math import (
    expand_packed_triangular,
    invlogit,
    invprobit,
    logaddexp,
    logit,
    logsumexp,
    probit,
)
from .model import *
from .model_graph import model_to_graphviz
from .plots import *
from .sampling import *
from .smc import *
from .stats import *
from .step_methods import *
from .tests import test
from .theanof import *
from .tuning import *
from .variational import *
from .vartypes import *
