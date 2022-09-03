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
import pytest
import scipy.stats as st

import pymc as pm

from pymc.initial_point import make_initial_point_fn


class TestMoments:
    @pytest.mark.parametrize("mu", [0, np.arange(3)], ids=str)
    @pytest.mark.parametrize("sigma", [1, np.array([1, 2, 5])], ids=str)
    @pytest.mark.parametrize("size", [None, 3, (5, 3)], ids=str)
    def test_simulator_moment(self, mu, sigma, size):
        def normal_sim(rng, mu, sigma, size):
            return rng.normal(mu, sigma, size=size)

        with pm.Model() as model:
            x = pm.Simulator("x", normal_sim, mu, sigma, size=size)

        fn = make_initial_point_fn(
            model=model,
            return_transformed=False,
            default_strategy="moment",
        )

        random_draw = model["x"].eval()
        result = fn(0)["x"]
        assert result.shape == random_draw.shape

        # We perform a z-test between the moment and expected mean from a sample of 10 draws
        # This test fails if the number of samples averaged in moment(Simulator)
        # is much smaller than 10, but would not catch the case where the number of samples
        # is higher than the expected 10

        n = 10  # samples
        expected_sample_mean = mu
        expected_sample_mean_std = np.sqrt(sigma**2 / n)

        # Multiple test adjustment for z-test to maintain alpha=0.01
        alpha = 0.01
        alpha /= 2 * 2 * 3  # Correct for number of test permutations
        alpha /= random_draw.size  # Correct for distribution size
        cutoff = st.norm().ppf(1 - (alpha / 2))

        assert np.all(np.abs((result - expected_sample_mean) / expected_sample_mean_std) < cutoff)
