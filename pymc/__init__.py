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
__version__ = "4.0.0b1"

import logging
import multiprocessing as mp
import platform

_log = logging.getLogger("pymc")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


def _check_install_compatibilitites():
    try:
        import theano

        _log.warning(
            "!" * 60
            + f"\nYour Python environment has Theano(-PyMC) {theano.__version__} installed, "
            + f"but you are importing PyMC {__version__} which uses Aesara as its backend."
            + f"\nFor PyMC {__version__} to work as expected you should uninstall Theano(-PyMC)."
            + "\nSee https://github.com/pymc-devs/pymc/wiki for update instructions.\n"
            + "!" * 60
        )
    except ImportError:
        pass

    try:
        import pymc3

        _log.warning(
            "!" * 60
            + f"\nYou are importing PyMC {__version__}, but your environment also has"
            + f" the legacy version PyMC3 {pymc3.__version__} installed."
            + f"\nFor PyMC {__version__} to work as expected you should uninstall PyMC3."
            + "\nSee https://github.com/pymc-devs/pymc/wiki for update instructions.\n"
            + "!" * 60
        )
    except ImportError:
        pass


_check_install_compatibilitites()


def __set_compiler_flags():
    # Workarounds for Aesara compiler problems on various platforms
    import aesara

    current = aesara.config.gcc__cxxflags
    aesara.config.gcc__cxxflags = f"{current} -Wno-c++11-narrowing"


__set_compiler_flags()

from pymc import gp, ode, sampling
from pymc.aesaraf import *
from pymc.backends import predictions_to_inference_data, to_inference_data
from pymc.backends.tracetab import *
from pymc.bart import *
from pymc.blocking import *
from pymc.data import *
from pymc.distributions import *
from pymc.distributions import transforms
from pymc.exceptions import *
from pymc.math import (
    expand_packed_triangular,
    invlogit,
    invprobit,
    logaddexp,
    logit,
    logsumexp,
    probit,
)
from pymc.model import *
from pymc.model_graph import model_to_graphviz
from pymc.plots import *
from pymc.printing import *
from pymc.sampling import *
from pymc.smc import *
from pymc.stats import *
from pymc.step_methods import *
from pymc.tests import test
from pymc.tuning import *
from pymc.variational import *
from pymc.vartypes import *
