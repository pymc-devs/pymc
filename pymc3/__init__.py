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
__version__ = "3.9.0"

import logging
import multiprocessing as mp
import platform

_log = logging.getLogger("pymc3")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

# Set start method to forkserver for MacOS to enable multiprocessing
# Closes issue https://github.com/pymc-devs/pymc3/issues/3849
sys = platform.system()
if sys == "Darwin":
    new_context = mp.get_context("forkserver")


def __set_compiler_flags():
    # Workarounds for Theano compiler problems on various platforms
    import theano

    system = platform.system()
    if system == "Windows":
        theano.config.mode = "FAST_COMPILE"
    elif system == "Darwin":
        theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"


__set_compiler_flags()

from .blocking import *
from .distributions import *
from .distributions import transforms
from .glm import *
from . import gp
from .math import (
    logaddexp,
    logsumexp,
    logit,
    invlogit,
    expand_packed_triangular,
    probit,
    invprobit,
)
from .model import *
from .model_graph import model_to_graphviz
from . import ode
from .stats import *
from .sampling import *
from .step_methods import *
from .smc import *
from .theanof import *
from .tuning import *
from .variational import *
from .vartypes import *
from .exceptions import *
from . import sampling

from .backends.tracetab import *
from .backends import save_trace, load_trace, point_list_to_multitrace

from .plots import *
from .tests import test

from .data import *
