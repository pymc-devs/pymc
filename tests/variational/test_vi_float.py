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

import numpy as np
import pytensor
import pytest

import pymc as pm


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_vi_float_precision(dtype_str):
    """Test that Variational Inference (ADVI) works under float32 and float64 precision."""
    rng = np.random.default_rng(123)

    # Temporarily override floatX only within this test scope
    with pytensor.config.change_flags(floatX=dtype_str):
        data = rng.normal(loc=1.0, scale=1.0, size=50).astype(dtype_str)

        with pm.Model():
            mu = pm.Normal(
                "mu",
                mu=np.array(0.0, dtype=dtype_str),
                sigma=np.array(1.0, dtype=dtype_str),
            )
            sigma = pm.HalfNormal(
                "sigma",
                sigma=np.array(1.0, dtype=dtype_str),
            )
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

            approx = pm.fit(
                n=100,
                method="advi",
                progressbar=False,
                random_seed=123,
            )

        assert approx is not None

        # Request MultiTrace so dtype checks are straightforward
        trace = approx.sample(
            draws=10,
            return_inferencedata=False,
            random_seed=123,
        )

        assert trace.get_values("mu").dtype == np.dtype(dtype_str)
        assert trace.get_values("sigma").dtype == np.dtype(dtype_str)
