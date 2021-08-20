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
__version__ = "3.11.4"

import logging
import multiprocessing as mp
import platform

import semver
import theano

_log = logging.getLogger("pymc3")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


def _check_backend_version():
    backend_paths = theano.__spec__.submodule_search_locations
    try:
        backend_version = theano.__version__
    except:
        print(
            "!" * 60
            + f"\nThe imported Theano(-PyMC) module is broken."
            + f"\nIt was imported from {backend_paths}"
            + "\nTry to uninstall/reinstall it after closing all active sessions/notebooks."
            + "\nAlso see https://github.com/pymc-devs/pymc3/wiki for installation instructions.\n"
            + "!" * 60
        )
        return
    if not semver.VersionInfo.parse(backend_version).match(">=1.1.2"):
        print(
            "!" * 60
            + f"\nThe installed Theano(-PyMC) version ({theano.__version__}) does not match the PyMC3 requirements."
            + f"\nIt was imported from {backend_paths}"
            + "\nFor PyMC3 to work, a compatible Theano-PyMC backend version must be installed."
            + "\nSee https://github.com/pymc-devs/pymc3/wiki for installation instructions.\n"
            + "!" * 60
        )


def __set_compiler_flags():
    # Workarounds for Theano compiler problems on various platforms
    current = theano.config.gcc__cxxflags
    theano.config.gcc__cxxflags = f"{current} -Wno-c++11-narrowing"


def _hotfix_theano_printing():
    """ This is a workaround for https://github.com/pymc-devs/aesara/issues/309 """
    try:
        import pydot
        import theano.printing

        if theano.printing.Node != pydot.Node:
            theano.printing.Node = pydot.Node
    except ImportError:
        # pydot is not installed
        pass


_check_backend_version()
__set_compiler_flags()
_hotfix_theano_printing()

from pymc3 import gp, ode, sampling
from pymc3.backends import load_trace, save_trace
from pymc3.backends.tracetab import *
from pymc3.blocking import *
from pymc3.data import *
from pymc3.distributions import *
from pymc3.distributions import transforms
from pymc3.exceptions import *
from pymc3.glm import *
from pymc3.gp.util import plot_gp_dist
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
from pymc3.stats import *
from pymc3.step_methods import *
from pymc3.tests import test
from pymc3.theanof import *
from pymc3.tuning import *
from pymc3.variational import *
from pymc3.vartypes import *
