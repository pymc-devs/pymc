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
import pytest
import scipy as sp

import pymc as pm

from pymc import logcdf, logp
from pymc.distributions.shape_utils import change_dist_size


class TestCensored:
    @pytest.mark.parametrize("censored", (False, True))
    def test_censored_workflow(self, censored):
        # Based on pymc-examples/censored_data
        rng = np.random.default_rng(1234)
        size = 500
        true_mu = 13.0
        true_sigma = 5.0

        # Set censoring limits
        low = 3.0
        high = 16.0

        # Draw censored samples
        data = rng.normal(true_mu, true_sigma, size)
        data[data <= low] = low
        data[data >= high] = high

        rng = 17092021
        with pm.Model() as m:
            mu = pm.Normal(
                "mu",
                mu=((high - low) / 2) + low,
                sigma=(high - low) / 2.0,
                initval="support_point",
            )
            sigma = pm.HalfNormal("sigma", sigma=(high - low) / 2.0, initval="support_point")
            observed = pm.Censored(
                "observed",
                pm.Normal.dist(mu=mu, sigma=sigma),
                lower=low if censored else None,
                upper=high if censored else None,
                observed=data,
            )

            prior_pred = pm.sample_prior_predictive(random_seed=rng)
            # posterior = pm.sample(tune=250, draws=250, random_seed=rng)
            posterior = pm.sample(
                tune=240, draws=270, discard_tuned_samples=True, random_seed=rng, max_treedepth=10
            )
            posterior_pred = pm.sample_posterior_predictive(posterior, random_seed=rng)

        expected = True if censored else False
        assert (9 < prior_pred.prior_predictive.mean() < 10) == expected
        assert (13 < posterior.posterior["mu"].mean() < 14) == expected
        assert (4.5 < posterior.posterior["sigma"].mean() < 5.5) == expected
        assert (12 < posterior_pred.posterior_predictive.mean() < 13) == expected

    def test_censored_invalid_dist(self):
        with pm.Model():
            invalid_dist = pm.Normal
            with pytest.raises(
                ValueError,
                match=r"Censoring dist must be a distribution created via the",
            ):
                x = pm.Censored("x", invalid_dist, lower=None, upper=None)

        with pm.Model():
            mv_dist = pm.Dirichlet.dist(a=[1, 1, 1])
            with pytest.raises(
                NotImplementedError,
                match="Censoring of multivariate distributions has not been implemented yet",
            ):
                x = pm.Censored("x", mv_dist, lower=None, upper=None)

        with pm.Model():
            registered_dist = pm.Normal("dist")
            with pytest.raises(
                ValueError,
                match="The dist dist was already registered in the current model",
            ):
                x = pm.Censored("x", registered_dist, lower=None, upper=None)

    def test_change_dist_size(self):
        base_dist = pm.Censored.dist(pm.Normal.dist(), -1, 1, size=(3, 2))

        new_dist = change_dist_size(base_dist, (4, 1))
        assert new_dist.eval().shape == (4, 1)

        new_dist = change_dist_size(base_dist, (4,), expand=True)
        assert new_dist.eval().shape == (4, 3, 2)

    def test_dist_broadcasted_by_lower_upper(self):
        x = pm.Censored.dist(pm.Normal.dist(), lower=np.zeros((2,)), upper=None)
        assert tuple(x.owner.inputs[0].shape.eval()) == (2,)

        x = pm.Censored.dist(pm.Normal.dist(), lower=np.zeros((2,)), upper=np.zeros((4, 2)))
        assert tuple(x.owner.inputs[0].shape.eval()) == (4, 2)

        x = pm.Censored.dist(
            pm.Normal.dist(size=(3, 4, 2)), lower=np.zeros((2,)), upper=np.zeros((4, 2))
        )
        assert tuple(x.owner.inputs[0].shape.eval()) == (3, 4, 2)

    def test_censored_categorical(self):
        cat = pm.Categorical.dist([0.1, 0.2, 0.2, 0.3, 0.2], shape=(5,))

        np.testing.assert_allclose(
            logp(cat, [-1, 0, 1, 2, 3, 4, 5]).exp().eval(),
            [0, 0.1, 0.2, 0.2, 0.3, 0.2, 0],
        )

        censored_cat = pm.Censored.dist(cat, lower=1, upper=3, shape=(5,))

        np.testing.assert_allclose(
            logp(censored_cat, [-1, 0, 1, 2, 3, 4, 5]).exp().eval(),
            [0, 0, 0.3, 0.2, 0.5, 0, 0],
        )

    def test_censored_logcdf_continuous(self):
        norm = pm.Normal.dist(0, 1)
        eval_points = np.array([-np.inf, -2, -1, 0, 1, 2, np.inf])
        expected_logcdf_uncensored = sp.stats.norm.logcdf(eval_points)

        match_str = "divide by zero encountered in log|invalid value encountered in subtract"

        # No censoring
        censored_norm = pm.Censored.dist(norm, lower=None, upper=None)
        censored_eval = logcdf(censored_norm, eval_points).eval()
        np.testing.assert_allclose(censored_eval, expected_logcdf_uncensored)

        # Left censoring
        censored_norm = pm.Censored.dist(norm, lower=-1, upper=None)
        expected_left = np.where(eval_points < -1, -np.inf, expected_logcdf_uncensored)
        censored_eval = logcdf(censored_norm, eval_points).eval()
        np.testing.assert_allclose(
            censored_eval,
            expected_left,
            rtol=1e-6,
        )

        # Right censoring
        censored_norm = pm.Censored.dist(norm, lower=None, upper=1)
        expected_right = np.where(eval_points >= 1, 0.0, expected_logcdf_uncensored)
        censored_eval = logcdf(censored_norm, eval_points).eval()
        np.testing.assert_allclose(
            censored_eval,
            expected_right,
            rtol=1e-6,
        )

        # Interval censoring
        censored_norm = pm.Censored.dist(norm, lower=-1, upper=1)
        expected_interval = np.where(eval_points < -1, -np.inf, expected_logcdf_uncensored)
        expected_interval = np.where(eval_points >= 1, 0.0, expected_interval)
        censored_eval = logcdf(censored_norm, eval_points).eval()
        np.testing.assert_allclose(
            censored_eval,
            expected_interval,
            rtol=1e-6,
        )

    def test_censored_logcdf_discrete(self):
        probs = [0.1, 0.2, 0.2, 0.3, 0.2]
        cat = pm.Categorical.dist(probs)
        eval_points = np.array([-1, 0, 1, 2, 3, 4, 5])

        cdf = np.cumsum(probs)
        log_cdf_base = np.log(cdf)
        expected_logcdf_uncensored = np.full_like(eval_points, -np.inf, dtype=float)
        expected_logcdf_uncensored[1:6] = log_cdf_base
        expected_logcdf_uncensored[6] = 0.0

        # No censoring
        censored_cat = pm.Censored.dist(cat, lower=None, upper=None)
        np.testing.assert_allclose(
            logcdf(censored_cat, eval_points).eval(),
            expected_logcdf_uncensored,
        )

        # Left censoring
        censored_cat = pm.Censored.dist(cat, lower=1, upper=None)
        expected_left = np.where(eval_points < 1, -np.inf, expected_logcdf_uncensored)
        np.testing.assert_allclose(
            logcdf(censored_cat, eval_points).eval(),
            expected_left,
        )

        # Right censoring
        censored_cat = pm.Censored.dist(cat, lower=None, upper=3)
        expected_right = np.where(eval_points >= 3, 0.0, expected_logcdf_uncensored)
        np.testing.assert_allclose(
            logcdf(censored_cat, eval_points).eval(),
            expected_right,
        )

        # Interval censoring
        censored_cat = pm.Censored.dist(cat, lower=1, upper=3)
        expected_interval = np.where(eval_points < 1, -np.inf, expected_logcdf_uncensored)
        expected_interval = np.where(eval_points >= 3, 0.0, expected_interval)
        np.testing.assert_allclose(
            logcdf(censored_cat, eval_points).eval(),
            expected_interval,
        )
