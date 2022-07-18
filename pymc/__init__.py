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
__version__ = "4.1.3"

import logging
import sys

_log = logging.getLogger("pymc")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)


def __suppress_aesara_import_warnings():
    """This is a workaround to suppress nonlethal NumPy warnings.

    Some more printouts remain. See https://github.com/numpy/numpy/issues/21942

    Remove this once https://github.com/aesara-devs/aesara/pull/980 is merged.
    """
    # We need to catch warnings as in some cases NumPy prints
    # stuff that we don't want the user to see.
    import io
    import warnings

    from contextlib import redirect_stderr, redirect_stdout

    import numpy.distutils.system_info

    class NumpyCompatibleStdoutStringIO(io.StringIO):
        """Used as a temporary replacement of sys.stdout to capture Numpy's output.
        We want to simply use io.StringIO, but this doesn't work because
        Numpy expects the .encoding attribute to be a string. For io.StringIO,
        this attribute is set to None and cannot be modified, hence the need for
        this subclass.
        (See forward_bytes_to_stdout in numpy.distutils.exec_command.)
        """

        encoding = sys.stdout.encoding

    # Known executables which trigger false-positive warnings when not found.
    # Ref: <https://github.com/conda-forge/aesara-feedstock/issues/54>
    executables = ["g77", "f77", "ifort", "ifl", "f90", "DF", "efl"]

    # The Numpy function which defines blas_info emits false-positive warnings.
    # In what follows we capture these warnings and ignore them.
    with warnings.catch_warnings(record=True):
        # The warnings about missing executables don't use Python's "warnings"
        # mechanism, and thus are not filtered by the catch_warnings context
        # above. On Linux the warnings are printed to stderr, but on Windows
        # they are printed to stdout. Thus we capture and filter both stdout
        # and stderr.
        stdout_sio, stderr_sio = NumpyCompatibleStdoutStringIO(), io.StringIO()
        with redirect_stdout(stdout_sio), redirect_stderr(stderr_sio):
            numpy.distutils.system_info.get_info("blas_opt")

        # Print any unfiltered messages to stdout and stderr.
        # (We hope that, beyond what we filter out, there should have been
        # no messages printed to stdout or stderr. In case there were messages,
        # we want to print them for the user to see. In what follows, we print
        # the stdout messages followed by the stderr messages. This means that
        # messages will be printed in the wrong order in the case that
        # there is output to both stdout and stderr, and the stderr output
        # doesn't come at the end.)
        for captured_buffer, print_destination in (
            (stdout_sio, sys.stdout),
            (stderr_sio, sys.stderr),
        ):
            raw_lines = captured_buffer.getvalue().splitlines()
            filtered_lines = [
                line
                for line in raw_lines
                # Keep a line when none of the warnings are found within
                # (equiv. when all of the warnings are not found within).
                if all(f"Could not locate executable {exe}" not in line for exe in executables)
            ]
            for line in filtered_lines:
                print(line, file=print_destination)
    return


def __set_compiler_flags():
    # Workarounds for Aesara compiler problems on various platforms
    import aesara

    current = aesara.config.gcc__cxxflags
    augmented = f"{current} -Wno-c++11-narrowing"

    # Work around compiler bug in GCC < 8.4 related to structured exception
    # handling registers on Windows.
    # See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65782 for details.
    # First disable C++ exception handling altogether since it's not needed
    # for the C extensions that we generate.
    augmented = f"{augmented} -fno-exceptions"
    # Now disable the generation of stack unwinding tables.
    augmented = f"{augmented} -fno-unwind-tables -fno-asynchronous-unwind-tables"

    aesara.config.gcc__cxxflags = augmented


if sys.platform == "win32":
    __suppress_aesara_import_warnings()
__set_compiler_flags()

from pymc import gp, ode, sampling
from pymc.aesaraf import *
from pymc.backends import *
from pymc.blocking import *
from pymc.data import *
from pymc.distributions import *
from pymc.exceptions import *
from pymc.func_utils import find_constrained_prior
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
from pymc.tuning import *
from pymc.variational import *
from pymc.vartypes import *
