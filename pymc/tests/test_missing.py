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
import pytest
import scipy.stats

from aesara.graph import graph_inputs
from numpy import array, ma

from pymc import joint_logpt
from pymc.distributions import Dirichlet, Gamma, Normal, Uniform
from pymc.exceptions import ImputationWarning
from pymc.model import Model
from pymc.sampling import sample, sample_posterior_predictive, sample_prior_predictive


@pytest.fixture(params=["masked", "pandas"])
def missing_data(request):
    if request.param == "masked":
        return ma.masked_values([1, 2, -1, 4, -1], value=-1)
    else:
        # request.param == "pandas"
        pd = pytest.importorskip("pandas")
        return pd.DataFrame([1, 2, np.nan, 4, np.nan])


def test_missing(missing_data):

    with Model() as model:
        x = Normal("x", 1, 1)
        with pytest.warns(ImputationWarning):
            _ = Normal("y", x, 1, observed=missing_data)

    assert "y_missing" in model.named_vars

    test_point = model.compute_initial_point()
    assert not np.isnan(model.compile_logp()(test_point))

    with model:
        prior_trace = sample_prior_predictive(return_inferencedata=False)
    assert {"x", "y"} <= set(prior_trace.keys())


def test_missing_with_predictors():
    predictors = array([0.5, 1, 0.5, 2, 0.3])
    data = ma.masked_values([1, 2, -1, 4, -1], value=-1)
    with Model() as model:
        x = Normal("x", 1, 1)
        with pytest.warns(ImputationWarning):
            y = Normal("y", x * predictors, 1, observed=data)

    assert "y_missing" in model.named_vars

    test_point = model.compute_initial_point()
    assert not np.isnan(model.compile_logp()(test_point))

    with model:
        prior_trace = sample_prior_predictive(return_inferencedata=False)
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

        prior_trace = sample_prior_predictive(return_inferencedata=False)
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

        assert "theta1_observed" in model.named_vars
        assert "theta1_missing_interval__" in model.named_vars
        assert not hasattr(
            model.rvs_to_values[model.named_vars["theta1_observed"]].tag, "transform"
        )

        prior_trace = sample_prior_predictive(return_inferencedata=False)

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

        trace = sample(
            chains=1, draws=50, compute_convergence_checks=False, return_inferencedata=False
        )

        assert np.all(0 < trace["theta1_missing"].mean(0))
        assert np.all(0 < trace["theta2_missing"].mean(0))
        assert "theta1" not in trace.varnames
        assert "theta2" not in trace.varnames

        # Make sure that the observed values are newly generated samples and that
        # the observed and deterministic matche
        pp_trace = sample_posterior_predictive(trace, return_inferencedata=False, keep_size=False)
        assert np.all(np.var(pp_trace["theta1"], 0) > 0.0)
        assert np.all(np.var(pp_trace["theta2"], 0) > 0.0)
        assert np.mean(pp_trace["theta1"][:, ~obs1.mask] - pp_trace["theta1_observed"]) == 0.0
        assert np.mean(pp_trace["theta2"][:, ~obs2.mask] - pp_trace["theta2_observed"]) == 0.0


def test_double_counting():
    with Model(check_bounds=False) as m1:
        x = Gamma("x", 1, 1, size=4)

    logp_val = m1.compile_logp()({"x_log__": np.array([0, 0, 0, 0])})
    assert logp_val == -4.0

    with Model(check_bounds=False) as m2:
        x = Gamma("x", 1, 1, observed=[1, 1, 1, np.nan])

    logp_val = m2.compile_logp()({"x_missing_log__": np.array([0])})
    assert logp_val == -4.0


def test_missing_logp():
    with Model() as m:
        theta1 = Normal("theta1", 0, 5, observed=[0, 1, 2, 3, 4])
        theta2 = Normal("theta2", mu=theta1, observed=[0, 1, 2, 3, 4])
    m_logp = m.compile_logp()({})

    with Model() as m_missing:
        theta1 = Normal("theta1", 0, 5, observed=np.array([0, 1, np.nan, 3, np.nan]))
        theta2 = Normal("theta2", mu=theta1, observed=np.array([np.nan, np.nan, 2, np.nan, 4]))
    m_missing_logp = m_missing.compile_logp()(
        {"theta1_missing": [2, 4], "theta2_missing": [0, 1, 3]}
    )

    assert m_logp == m_missing_logp


def test_missing_multivariate():
    """Test model with missing variables whose transform changes base shape still works"""

    with Model() as m_miss:
        with pytest.raises(
            NotImplementedError,
            match="Automatic inputation is only supported for univariate RandomVariables",
        ):
            x = Dirichlet(
                "x", a=[1, 2, 3], observed=np.array([[0.3, 0.3, 0.4], [np.nan, np.nan, np.nan]])
            )

    # TODO: Test can be used when local_subtensor_rv_lift supports multivariate distributions
    # from pymc.distributions.transforms import simplex
    #
    # with Model() as m_unobs:
    #     x = Dirichlet("x", a=[1, 2, 3])
    #
    # inp_vals = simplex.forward(np.array([0.3, 0.3, 0.4])).eval()
    # assert np.isclose(
    #     m_miss.compile_logp()({"x_missing_simplex__": inp_vals}),
    #     m_unobs.compile_logp(jacobian=False)({"x_simplex__": inp_vals}) * 2,
    # )


def test_missing_vector_parameter():
    with Model() as m:
        x = Normal(
            "x",
            np.array([-10, 10]),
            0.1,
            observed=np.array([[np.nan, 10], [-10, np.nan], [np.nan, np.nan]]),
        )
    x_draws = x.eval()
    assert x_draws.shape == (3, 2)
    assert np.all(x_draws[:, 0] < 0)
    assert np.all(x_draws[:, 1] > 0)
    assert np.isclose(
        m.compile_logp()({"x_missing": np.array([-10, 10, -10, 10])}),
        scipy.stats.norm(scale=0.1).logpdf(0) * 6,
    )


def test_missing_symmetric():
    """Check that logpt works when partially observed variable have equal observed and
    unobserved dimensions.

    This would fail in a previous implementation because the two variables would be
    equivalent and one of them would be discarded during MergeOptimization while
    buling the logpt graph
    """
    with Model() as m:
        x = Gamma("x", alpha=3, beta=10, observed=np.array([1, np.nan]))

    x_obs_rv = m["x_observed"]
    x_obs_vv = m.rvs_to_values[x_obs_rv]

    x_unobs_rv = m["x_missing"]
    x_unobs_vv = m.rvs_to_values[x_unobs_rv]

    logp = joint_logpt([x_obs_rv, x_unobs_rv], {x_obs_rv: x_obs_vv, x_unobs_rv: x_unobs_vv})
    logp_inputs = list(graph_inputs([logp]))
    assert x_obs_vv in logp_inputs
    assert x_unobs_vv in logp_inputs
