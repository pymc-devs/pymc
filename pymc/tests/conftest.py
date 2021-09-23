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

import aesara
import numpy as np
import pytest

import pymc as pm


@pytest.fixture(scope="function", autouse=True)
def aesara_config():
    config = aesara.config.change_flags(on_opt_error="raise")
    with config:
        yield


@pytest.fixture(scope="function", autouse=True)
def exception_verbosity():
    config = aesara.config.change_flags(exception_verbosity="high")
    with config:
        yield


@pytest.fixture(scope="function", autouse=False)
def strict_float32():
    if aesara.config.floatX == "float32":
        config = aesara.config.change_flags(warn_float64="raise")
        with config:
            yield
    else:
        yield


@pytest.fixture(scope="function", autouse=False)
def seeded_test():
    # TODO: use this instead of SeededTest
    np.random.seed(42)
    pm.set_at_rng(42)
