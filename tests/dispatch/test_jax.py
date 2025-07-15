#   Copyright 2024 The PyMC Developers
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
import pytest

from pytensor import function

import pymc as pm

from pymc.dispatch import dispatch_jax  # noqa: F401

jax = pytest.importorskip("jax")


@pytest.mark.parametrize("sigma", [0.02, 5])
def test_jax_TruncatedNormal(sigma):
    with pm.Model() as m:
        lower = 5
        upper = 8
        mu = 6

        a = pm.TruncatedNormal(
            "a", mu, sigma, lower=lower, upper=upper, rng=np.random.default_rng(seed=123)
        )

        f_jax = function(
            [],
            [
                pm.TruncatedNormal(
                    "b",
                    mu,
                    sigma,
                    lower=lower,
                    upper=upper,
                    rng=np.random.default_rng(seed=123),
                )
            ],
            mode="JAX",
        )
        res = f_jax()

        draws = pm.draw(a, draws=100, mode="JAX")

    assert jax.numpy.all((draws >= lower) & (draws <= upper))
    assert jax.numpy.all((res[0] >= lower) & (res[0] <= upper))
