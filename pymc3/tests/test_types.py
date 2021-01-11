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

from copy import copy

import numpy as np
import theano

from pymc3.distributions import Normal
from pymc3.model import Model
from pymc3.sampling import sample
from pymc3.step_methods import MLDA, NUTS, HamiltonianMC, Metropolis, Slice


class TestType:
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS)

    def setup_method(self):
        # save theano config object
        self.theano_config = copy(theano.config)

    def teardown_method(self):
        # restore theano config
        theano.config = self.theano_config

    @theano.config.change_flags({"floatX": "float64", "warn_float64": "ignore"})
    def test_float64(self):
        with Model() as model:
            x = Normal("x", testval=np.array(1.0, dtype="float64"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=np.random.randn(5))

        assert x.dtype == "float64"
        assert obs.dtype == "float64"

        for sampler in self.samplers:
            with model:
                sample(10, sampler())

    @theano.config.change_flags({"floatX": "float32", "warn_float64": "warn"})
    def test_float32(self):
        with Model() as model:
            x = Normal("x", testval=np.array(1.0, dtype="float32"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=np.random.randn(5).astype("float32"))

        assert x.dtype == "float32"
        assert obs.dtype == "float32"

        for sampler in self.samplers:
            with model:
                sample(10, sampler())

    @theano.config.change_flags({"floatX": "float64", "warn_float64": "ignore"})
    def test_float64_MLDA(self):
        data = np.random.randn(5)

        with Model() as coarse_model:
            x = Normal("x", testval=np.array(1.0, dtype="float64"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=data + 0.5)

        with Model() as model:
            x = Normal("x", testval=np.array(1.0, dtype="float64"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=data)

        assert x.dtype == "float64"
        assert obs.dtype == "float64"

        with model:
            sample(10, MLDA(coarse_models=[coarse_model]))

    @theano.config.change_flags({"floatX": "float32", "warn_float64": "warn"})
    def test_float32_MLDA(self):
        data = np.random.randn(5).astype("float32")

        with Model() as coarse_model:
            x = Normal("x", testval=np.array(1.0, dtype="float32"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=data + 0.5)

        with Model() as model:
            x = Normal("x", testval=np.array(1.0, dtype="float32"))
            obs = Normal("obs", mu=x, sigma=1.0, observed=data)

        assert x.dtype == "float32"
        assert obs.dtype == "float32"

        with model:
            sample(10, MLDA(coarse_models=[coarse_model]))
