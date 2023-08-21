#   Copyright 2023 The PyMC Developers
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

import numpy as np
import pytensor
import pytest


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


@pytest.fixture(scope="function", autouse=False)
def seeded_test():
    np.random.seed(20160911)


@pytest.fixture
def fail_on_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
