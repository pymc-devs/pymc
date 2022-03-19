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
import scipy.stats as st

import pymc as pm

from pymc.tests.helpers import SeededTest


class TestShared(SeededTest):
    def test_deterministic(self):
        with pm.Model() as model:
            data_values = np.array([0.5, 0.4, 5, 2])
            X = aesara.shared(np.asarray(data_values, dtype=aesara.config.floatX), borrow=True)
            pm.Normal("y", 0, 1, observed=X)
            assert np.all(
                np.isclose(model.compile_logp(sum=False)({}), st.norm().logpdf(data_values))
            )

    def test_sample(self):
        x = np.random.normal(size=100)
        y = x + np.random.normal(scale=1e-2, size=100)

        x_pred = np.linspace(-3, 3, 200)

        x_shared = aesara.shared(x)

        with pm.Model() as model:
            b = pm.Normal("b", 0.0, 10.0)
            pm.Normal("obs", b * x_shared, np.sqrt(1e-2), observed=y)
            prior_trace0 = pm.sample_prior_predictive(1000)

            idata = pm.sample(1000, tune=1000, chains=1)
            pp_trace0 = pm.sample_posterior_predictive(idata)

            x_shared.set_value(x_pred)
            prior_trace1 = pm.sample_prior_predictive(1000)
            pp_trace1 = pm.sample_posterior_predictive(idata)

        assert prior_trace0.prior["b"].shape == (1, 1000)
        assert prior_trace0.prior_predictive["obs"].shape == (1, 1000, 100)
        np.testing.assert_allclose(
            x, pp_trace0.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )

        assert prior_trace1.prior["b"].shape == (1, 1000)
        assert prior_trace1.prior_predictive["obs"].shape == (1, 1000, 200)
        np.testing.assert_allclose(
            x_pred, pp_trace1.posterior_predictive["obs"].mean(("chain", "draw")), atol=1e-1
        )
