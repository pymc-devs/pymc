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
import pandas as pd
import pytest

from numpy import array, ma

from pymc3.distributions.continuous import Gamma, Normal, Uniform
from pymc3.distributions.transforms import Interval
from pymc3.exceptions import ImputationWarning
from pymc3.model import Model
from pymc3.sampling import sample, sample_posterior_predictive, sample_prior_predictive


@pytest.mark.parametrize(
    "data",
    [ma.masked_values([1, 2, -1, 4, -1], value=-1), pd.DataFrame([1, 2, np.nan, 4, np.nan])],
)
def test_missing(data):

    with Model() as model:
        x = Normal("x", 1, 1)
        with pytest.warns(ImputationWarning):
            y = Normal("y", x, 1, observed=data)

    assert "y_missing" in model.named_vars

    test_point = model.initial_point
    assert not np.isnan(model.logp(test_point))

    with model:
        prior_trace = sample_prior_predictive()
    assert {"x", "y"} <= set(prior_trace.keys())


def test_missing_with_predictors():
    predictors = array([0.5, 1, 0.5, 2, 0.3])
    data = ma.masked_values([1, 2, -1, 4, -1], value=-1)
    with Model() as model:
        x = Normal("x", 1, 1)
        with pytest.warns(ImputationWarning):
            y = Normal("y", x * predictors, 1, observed=data)

    assert "y_missing" in model.named_vars

    test_point = model.initial_point
    assert not np.isnan(model.logp(test_point))

    with model:
        prior_trace = sample_prior_predictive()
    assert {"x", "y"} <= set(prior_trace.keys())


def test_missing_dual_observations():
    with Model() as model:
        obs1 = ma.masked_values([1, 2, -1, 4, -1], value=-1)
        obs2 = ma.masked_values([-1, -1, 6, -1, 8], value=-1)
        beta1 = Normal("beta1", 1, 1)
        beta2 = Normal("beta2", 2, 1)
        latent = Normal("theta", size=5)
        with pytest.warns(ImputationWarning):
            ovar1 = Normal("o1", mu=beta1 * latent, observed=obs1)
        with pytest.warns(ImputationWarning):
            ovar2 = Normal("o2", mu=beta2 * latent, observed=obs2)

        prior_trace = sample_prior_predictive()
        assert {"beta1", "beta2", "theta", "o1", "o2"} <= set(prior_trace.keys())
        # TODO: Assert something
        trace = sample(chains=1, draws=50)


def test_interval_missing_observations():
    with Model() as model:
        obs1 = ma.masked_values([1, 2, -1, 4, -1], value=-1)
        obs2 = ma.masked_values([-1, -1, 6, -1, 8], value=-1)

        rng = aesara.shared(np.random.RandomState(2323), borrow=True)

        with pytest.warns(ImputationWarning):
            theta1 = Uniform("theta1", 0, 5, observed=obs1, rng=rng)
        with pytest.warns(ImputationWarning):
            theta2 = Normal("theta2", mu=theta1, observed=obs2, rng=rng)

        assert "theta1_observed_interval__" in model.named_vars
        assert "theta1_missing_interval__" in model.named_vars
        assert isinstance(
            model.rvs_to_values[model.named_vars["theta1_observed"]].tag.transform, Interval
        )

        prior_trace = sample_prior_predictive()

        # Make sure the observed + missing combined deterministics have the
        # same shape as the original observations vectors
        assert prior_trace["theta1"].shape[-1] == obs1.shape[0]
        assert prior_trace["theta2"].shape[-1] == obs2.shape[0]

        # Make sure that the observed values are newly generated samples
        assert np.all(np.var(prior_trace["theta1_observed"], 0) > 0.0)
        assert np.all(np.var(prior_trace["theta2_observed"], 0) > 0.0)

        # Make sure the missing parts of the combined deterministic matches the
        # sampled missing and observed variable values
        assert np.mean(prior_trace["theta1"][:, obs1.mask] - prior_trace["theta1_missing"]) == 0.0
        assert np.mean(prior_trace["theta1"][:, ~obs1.mask] - prior_trace["theta1_observed"]) == 0.0
        assert np.mean(prior_trace["theta2"][:, obs2.mask] - prior_trace["theta2_missing"]) == 0.0
        assert np.mean(prior_trace["theta2"][:, ~obs2.mask] - prior_trace["theta2_observed"]) == 0.0

        assert {"theta1", "theta2"} <= set(prior_trace.keys())

        trace = sample(chains=1, draws=50, compute_convergence_checks=False)

        assert np.all(0 < trace["theta1_missing"].mean(0))
        assert np.all(0 < trace["theta2_missing"].mean(0))
        assert "theta1" not in trace.varnames
        assert "theta2" not in trace.varnames

        # Make sure that the observed values are newly generated samples and that
        # the observed and deterministic matche
        pp_trace = sample_posterior_predictive(trace)
        assert np.all(np.var(pp_trace["theta1"], 0) > 0.0)
        assert np.all(np.var(pp_trace["theta2"], 0) > 0.0)
        assert np.mean(pp_trace["theta1"][:, ~obs1.mask] - pp_trace["theta1_observed"]) == 0.0
        assert np.mean(pp_trace["theta2"][:, ~obs2.mask] - pp_trace["theta2_observed"]) == 0.0


def test_double_counting():
    with Model(check_bounds=False) as m1:
        x = Gamma("x", 1, 1, size=4)

    logp_val = m1.logp({"x_log__": np.array([0, 0, 0, 0])})
    assert logp_val == -4.0

    with Model(check_bounds=False) as m2:
        x = Gamma("x", 1, 1, observed=[1, 1, 1, np.nan])

    logp_val = m2.logp({"x_missing_log__": np.array([0])})
    assert logp_val == -4.0


def test_missing_logp():
    with Model() as m:
        theta1 = Normal("theta1", 0, 5, observed=[0, 1, 2, 3, 4])
        theta2 = Normal("theta2", mu=theta1, observed=[0, 1, 2, 3, 4])
    m_logp = m.logp()

    with Model() as m_missing:
        theta1 = Normal("theta1", 0, 5, observed=np.array([0, 1, np.nan, 3, np.nan]))
        theta2 = Normal("theta2", mu=theta1, observed=np.array([np.nan, np.nan, 2, np.nan, 4]))
    m_missing_logp = m_missing.logp({"theta1_missing": [2, 4], "theta2_missing": [0, 1, 3]})

    assert m_logp == m_missing_logp
