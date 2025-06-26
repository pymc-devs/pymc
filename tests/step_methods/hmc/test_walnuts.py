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

import warnings

import numpy as np
import numpy.testing as npt
import pytest

import pymc as pm

from pymc.exceptions import SamplingError
from pymc.step_methods.hmc import WALNUTS
from tests import sampler_fixtures as sf
from tests.helpers import RVsAssignmentStepsTester, StepMethodTester


class WalnutsFixture(sf.BaseSampler):
    @classmethod
    def make_step(cls):
        args = {}
        if hasattr(cls, "step_args"):
            args.update(cls.step_args)
        if "scaling" not in args:
            _, step = pm.sampling.mcmc.init_nuts(n_init=10000, **args)
            # Replace the NUTS step with WALNUTS but keep the same mass matrix
            step = pm.WALNUTS(potential=step.potential, target_accept=step.target_accept, **args)
        else:
            step = pm.WALNUTS(**args)
        return step

    def test_target_accept(self):
        accept = self.trace[self.burn :]["mean_tree_accept"]
        npt.assert_allclose(accept.mean(), self.step.target_accept, 1)


# Basic distribution tests - these are relevant for WALNUTS since it's a general HMC sampler
class TestWALNUTSUniform(WalnutsFixture, sf.UniformFixture):
    n_samples = 5000  # Reduced for faster testing
    tune = 500
    burn = 500
    chains = 2
    min_n_eff = 2000
    rtol = 0.1
    atol = 0.05
    step_args = {"random_seed": 202010}


class TestWALNUTSNormal(WalnutsFixture, sf.NormalFixture):
    n_samples = 5000  # Reduced for faster testing
    tune = 500
    burn = 0
    chains = 2
    min_n_eff = 4000
    rtol = 0.1
    atol = 0.05
    step_args = {"random_seed": 123456}


# WALNUTS-specific functionality tests
class TestWalnutsSpecific:
    def test_walnuts_specific_stats(self):
        """Test that WALNUTS produces its specific statistics."""
        with pm.Model():
            pm.Normal("x", mu=0, sigma=1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                trace = pm.sample(
                    draws=10, tune=5, chains=1, return_inferencedata=False, step=pm.WALNUTS()
                )

        # Check WALNUTS-specific stats are present
        walnuts_stats = ["n_steps_total", "avg_steps_per_proposal"]
        for stat in walnuts_stats:
            assert stat in trace.stat_names, f"WALNUTS-specific stat '{stat}' missing"
            stats_values = trace.get_sampler_stats(stat)
            assert stats_values.shape == (10,), f"Wrong shape for {stat}"
            assert np.all(stats_values >= 0), f"{stat} should be non-negative"

        # Check that n_steps_total makes sense relative to tree_size
        n_steps = trace.get_sampler_stats("n_steps_total")
        tree_size = trace.get_sampler_stats("tree_size")
        # n_steps_total should generally be >= tree_size (adaptive steps might use more steps)
        assert np.all(n_steps >= tree_size), "n_steps_total should be >= tree_size"

    def test_walnuts_parameters(self):
        """Test WALNUTS-specific parameters."""
        with pm.Model():
            pm.Normal("x", mu=0, sigma=1)

            # Test custom max_error parameter
            step = pm.WALNUTS(max_error=0.5, max_treedepth=8)
            assert step.max_error == 0.5
            assert step.max_treedepth == 8

            # Test early_max_treedepth
            assert hasattr(step, "early_max_treedepth")

    def test_bad_init_handling(self):
        """Test that WALNUTS handles bad initialization properly."""
        with pm.Model():
            pm.HalfNormal("a", sigma=1, initval=-1, default_transform=None)
            with pytest.raises(SamplingError) as error:
                pm.sample(chains=1, random_seed=1, step=pm.WALNUTS())
            error.match("Bad initial energy")

    def test_competence_method(self):
        """Test WALNUTS competence for different variable types."""
        from pymc.step_methods.compound import Competence

        # Mock continuous variable with gradient
        class MockVar:
            dtype = "float64"  # continuous_types contains strings, not dtype objects

        var = MockVar()
        assert WALNUTS.competence(var, has_grad=True) == Competence.COMPATIBLE
        assert WALNUTS.competence(var, has_grad=False) == Competence.INCOMPATIBLE

    def test_required_attributes(self):
        """Test that WALNUTS has all required attributes."""
        with pm.Model():
            pm.Normal("x", mu=0, sigma=1)
            step = pm.WALNUTS()

            # Check required attributes
            assert hasattr(step, "name")
            assert step.name == "walnuts"
            assert hasattr(step, "default_blocked")
            assert step.default_blocked is True
            assert hasattr(step, "stats_dtypes_shapes")

            # Check WALNUTS-specific stats are defined
            required_stats = ["n_steps_total", "avg_steps_per_proposal"]
            for stat in required_stats:
                assert stat in step.stats_dtypes_shapes


# Test step method functionality
class TestStepWALNUTS(StepMethodTester):
    @pytest.mark.parametrize(
        "step_fn, draws",
        [
            (lambda C, _: WALNUTS(scaling=C, is_cov=True, blocked=False), 1000),
            (lambda C, _: WALNUTS(scaling=C, is_cov=True), 1000),
        ],
    )
    def test_step_continuous(self, step_fn, draws):
        self.step_continuous(step_fn, draws)


class TestRVsAssignmentWALNUTS(RVsAssignmentStepsTester):
    @pytest.mark.parametrize("step, step_kwargs", [(WALNUTS, {})])
    def test_continuous_steps(self, step, step_kwargs):
        self.continuous_steps(step, step_kwargs)


def test_walnuts_step_legacy_value_grad_function():
    """Test WALNUTS with legacy value grad function (compatibility test)."""
    with pm.Model() as m:
        x = pm.Normal("x", shape=(2,))
        y = pm.Normal("y", x, shape=(3, 2))

    legacy_value_grad_fn = m.logp_dlogp_function(ravel_inputs=False, mode="FAST_COMPILE")
    legacy_value_grad_fn.set_extra_values({})
    walnuts = WALNUTS(model=m, logp_dlogp_func=legacy_value_grad_fn)

    # Confirm it is a function of multiple variables
    logp, dlogp = walnuts._logp_dlogp_func([np.zeros((2,)), np.zeros((3, 2))])
    np.testing.assert_allclose(dlogp, np.zeros(8))

    # Confirm we can perform a WALNUTS step
    ip = m.initial_point()
    new_ip, _ = walnuts.step(ip)
    assert np.all(new_ip["x"] != ip["x"])
    assert np.all(new_ip["y"] != ip["y"])
