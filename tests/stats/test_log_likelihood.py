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
import numpy as np
import pytest
import scipy.stats as st

from arviz import InferenceData, dict_to_dataset, from_dict

from pymc.distributions import Dirichlet, Normal
from pymc.distributions.transforms import log
from pymc.model import Model
from pymc.stats.log_likelihood import compute_log_likelihood
from tests.distributions.test_multivariate import dirichlet_logpdf


class TestComputeLogLikelihood:
    @pytest.mark.parametrize("transform", (False, True))
    def test_basic(self, transform):
        transform = log if transform else None
        with Model(coords={"test_dim": range(3)}) as m:
            x = Normal("x", transform=transform)
            x_value_var = m.rvs_to_values[x]
            y = Normal("y", x, observed=[0, 1, 2], dims=("test_dim",))

            idata = InferenceData(posterior=dict_to_dataset({"x": np.arange(100).reshape(4, 25)}))
            res = compute_log_likelihood(idata)

        # Check we didn't erase the original mappings
        assert m.rvs_to_values[x] is x_value_var
        assert m.rvs_to_transforms[x] is transform

        assert res is idata
        assert res.log_likelihood.dims == {"chain": 4, "draw": 25, "test_dim": 3}

        np.testing.assert_allclose(
            res.log_likelihood["y"].values,
            st.norm.logpdf([0, 1, 2], np.arange(100)[:, None]).reshape(4, 25, 3),
        )

    def test_multivariate(self):
        rng = np.random.default_rng(39)

        p_draws = rng.normal(size=(4, 25, 3))
        y_draws = st.dirichlet(np.ones(3)).rvs(10, random_state=rng)
        with Model(coords={"test_event_dim": range(10), "test_support_dim": range(3)}) as m:
            p = Normal("p", dims=("test_support_dim",))
            y = Dirichlet(
                "y", a=p.exp(), observed=y_draws, dims=("test_event_dim", "test_support_dim")
            )

            idata = InferenceData(posterior=dict_to_dataset({"p": p_draws}))
            res = compute_log_likelihood(idata)

        assert res.log_likelihood.dims == {"chain": 4, "draw": 25, "test_event_dim": 10}

        np.testing.assert_allclose(
            res.log_likelihood["y"].values,
            dirichlet_logpdf(y_draws, np.exp(p_draws)[..., None, :]),
        )

    def test_var_names(self):
        with Model() as m:
            x = Normal("x")
            y1 = Normal("y1", x, observed=[0, 1, 2])
            y2 = Normal("y2", x, observed=[3, 4])

        idata = InferenceData(posterior=dict_to_dataset({"x": np.arange(100).reshape(4, 25)}))

        res_y1 = compute_log_likelihood(
            idata, var_names=["y1"], extend_inferencedata=False, model=m, progressbar=False
        )
        assert res_y1 is not idata
        assert set(res_y1.data_vars) == {"y1"}
        np.testing.assert_allclose(
            res_y1["y1"].values,
            st.norm.logpdf([0, 1, 2], np.arange(100)[:, None]).reshape(4, 25, 3),
        )

        res_y2 = compute_log_likelihood(
            idata, var_names=["y2"], extend_inferencedata=False, model=m, progressbar=False
        )
        assert res_y2 is not idata
        assert set(res_y2.data_vars) == {"y2"}
        np.testing.assert_allclose(
            res_y2["y2"].values,
            st.norm.logpdf([3, 4], np.arange(100)[:, None]).reshape(4, 25, 2),
        )

        res_both = compute_log_likelihood(idata, model=m, progressbar=False)
        assert res_both is idata
        assert set(res_both.log_likelihood.data_vars) == {"y1", "y2"}
        np.testing.assert_allclose(
            res_y1["y1"].values,
            res_both.log_likelihood["y1"].values,
        )
        np.testing.assert_allclose(
            res_y2["y2"].values,
            res_both.log_likelihood["y2"].values,
        )

    def test_invalid_var_names(self):
        with Model() as m:
            x = Normal("x")
            y = Normal("y", x, observed=[0, 1, 2])

            idata = InferenceData(posterior=dict_to_dataset({"x": np.arange(100).reshape(4, 25)}))
            with pytest.raises(ValueError, match="var_names must refer to observed_RVs"):
                compute_log_likelihood(idata, var_names=["x"])

    def test_dims_without_coords(self):
        # Issues #6820
        with Model() as m:
            x = Normal("x")
            y = Normal("y", x, observed=[0, 0, 0], shape=(3,), dims="obs")

            trace = from_dict({"x": [[0, 1]]})
            llike = compute_log_likelihood(trace)

        assert len(llike.log_likelihood["obs"]) == 3
        np.testing.assert_allclose(
            llike.log_likelihood["y"].values,
            st.norm.logpdf([[[0, 0, 0], [1, 1, 1]]]),
        )
