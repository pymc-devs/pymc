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
import scipy.stats as st

from arviz import InferenceData, dict_to_dataset

from pymc.distributions import Normal
from pymc.distributions.transforms import log
from pymc.model import Model
from pymc.stats.log_prior import compute_log_prior


class TestComputeLogPrior:
    @pytest.mark.parametrize("transform", (False, True))
    def test_basic(self, transform):
        transform = log if transform else None
        with Model() as m:
            x = Normal("x", transform=transform)
            x_value_var = m.rvs_to_values[x]
            Normal("y", x, observed=[0, 1, 2])

            idata = InferenceData(posterior=dict_to_dataset({"x": np.arange(100).reshape(4, 25)}))
            res = compute_log_prior(idata)

        # Check we didn't erase the original mappings
        assert m.rvs_to_values[x] is x_value_var
        assert m.rvs_to_transforms[x] is transform

        assert res is idata
        assert res.log_prior.dims == {"chain": 4, "draw": 25}

        np.testing.assert_allclose(
            res.log_prior["x"].values,
            st.norm(0, 1).logpdf(idata.posterior["x"].values),
        )
