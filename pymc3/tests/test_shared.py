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

import numpy as np
import theano

import pymc3 as pm

from pymc3.tests.helpers import SeededTest


class TestShared(SeededTest):
    def test_deterministic(self):
        with pm.Model() as model:
            data_values = np.array([0.5, 0.4, 5, 2])
            X = theano.shared(np.asarray(data_values, dtype=theano.config.floatX), borrow=True)
            pm.Normal("y", 0, 1, observed=X)
            model.logp(model.test_point)

    def test_sample(self):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200)

        x_shared = theano.shared(x)

        with pm.Model() as model:
            b = pm.Normal("b", 0.0, 10.0)
            pm.Normal("obs", b * x_shared, np.sqrt(1e-2), observed=y)
            prior_trace0 = pm.sample_prior_predictive(1000)

            trace = pm.sample(1000, init=None, tune=1000, chains=1)
            pp_trace0 = pm.sample_posterior_predictive(trace, 1000)
            pp_trace01 = pm.fast_sample_posterior_predictive(trace, 1000)

            x_shared.set_value(x_pred)
            prior_trace1 = pm.sample_prior_predictive(1000)
            pp_trace1 = pm.sample_posterior_predictive(trace, 1000)
            pp_trace11 = pm.fast_sample_posterior_predictive(trace, 1000)

        assert prior_trace0["b"].shape == (1000,)
        assert prior_trace0["obs"].shape == (1000, 100)
        np.testing.assert_allclose(x, pp_trace0["obs"].mean(axis=0), atol=1e-1)
        np.testing.assert_allclose(x, pp_trace01["obs"].mean(axis=0), atol=1e-1)

        assert prior_trace1["b"].shape == (1000,)
        assert prior_trace1["obs"].shape == (1000, 200)
        np.testing.assert_allclose(x_pred, pp_trace1["obs"].mean(axis=0), atol=1e-1)
        np.testing.assert_allclose(x_pred, pp_trace11["obs"].mean(axis=0), atol=1e-1)
