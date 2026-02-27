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
import warnings
import os
import subprocess
from pathlib import Path

import pytensor
import pytest

from numba.core.errors import NumbaPerformanceWarning, NumbaWarning


def _has_64bit_mingw(cxx_path: str) -> bool:
    cxx = Path(cxx_path.strip('"'))
    if not cxx.exists():
        return False

    try:
        proc = subprocess.run(
            [str(cxx), "-dumpmachine"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError:
        return False

    machine = (proc.stdout or "").strip().lower()
    return "x86_64" in machine or "amd64" in machine


@pytest.fixture(scope="session", autouse=True)
def windows_compiler_guard():
    cxx = (pytensor.config.cxx or "").strip()
    if not cxx:
        yield
        return

    # Some local Windows setups expose a 32-bit MinGW at C:\MinGW\bin\g++.EXE
    # which fails with "64-bit mode not compiled in". Fall back to pure-Python mode.
    if os.name == "nt" and "mingw" in cxx.lower() and not _has_64bit_mingw(cxx):
        with pytensor.config.change_flags(cxx=""):
            yield
        return

    yield


@pytest.fixture(scope="function", autouse=True)
def pytensor_config():
    config = pytensor.config.change_flags(on_opt_error="raise")
    with config:
        yield


@pytest.fixture(scope="function", autouse=True)
def exception_verbosity():
    config = pytensor.config.change_flags(exception_verbosity="high")
    with config:
        yield


@pytest.fixture(scope="function", autouse=False)
def strict_float32():
    if pytensor.config.floatX == "float32":
        config = pytensor.config.change_flags(warn_float64="raise")
        with config:
            yield
    else:
        yield


@pytest.fixture
def fail_on_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
        warnings.filterwarnings("ignore", ".*Cannot cache.*", NumbaWarning)
        yield
