#   Copyright 2021 The PyMC Developers
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
__version__ = "4.0"

import logging
import multiprocessing as mp
import platform

_log = logging.getLogger("pymc3")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

_log.info(
    "You are running the v4 development version of PyMC3 which currently still lacks key features. You probably want to use the stable v3 instead which you can either install via conda or find on the v3 GitHub branch: https://github.com/pymc-devs/pymc3/tree/v3"
)


def __set_compiler_flags():
    # Workarounds for Aesara compiler problems on various platforms
    import aesara

    current = aesara.config.gcc__cxxflags
    aesara.config.gcc__cxxflags = f"{current} -Wno-c++11-narrowing"


__set_compiler_flags()

from pymc3 import gp, ode, sampling
from pymc3.aesaraf import *
from pymc3.backends import (
    load_trace,
    predictions_to_inference_data,
    save_trace,
    to_inference_data,
)
from pymc3.backends.tracetab import *
from pymc3.blocking import *
from pymc3.data import *
from pymc3.distributions import *
from pymc3.distributions import transforms
from pymc3.exceptions import *
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
from pymc3.printing import *
from pymc3.sampling import *
from pymc3.smc import *
from pymc3.stats import *
from pymc3.step_methods import *
from pymc3.tests import test
from pymc3.tuning import *
from pymc3.variational import *
from pymc3.vartypes import *
