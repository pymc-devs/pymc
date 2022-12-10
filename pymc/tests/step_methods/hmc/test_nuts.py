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

import logging
import sys
import warnings

import numpy as np
import pytensor.tensor as at
import pytest

import pymc as pm

from pymc.exceptions import SamplingError
from pymc.pytensorf import floatX
from pymc.step_methods.hmc import NUTS
from pymc.tests import sampler_fixtures as sf
from pymc.tests.helpers import RVsAssignmentStepsTester, StepMethodTester


class TestNUTSUniform(sf.NutsFixture, sf.UniformFixture):
    n_samples = 10000
    tune = 1000
    burn = 1000
    chains = 4
    min_n_eff = 9000
    rtol = 0.1
    atol = 0.05


class TestNUTSUniform2(TestNUTSUniform):
    step_args = {"target_accept": 0.95}


class TestNUTSUniform3(TestNUTSUniform):
    step_args = {"target_accept": 0.80}


class TestNUTSNormal(sf.NutsFixture, sf.NormalFixture):
    n_samples = 10000
    tune = 1000
    burn = 0
    chains = 2
    min_n_eff = 10000
    rtol = 0.1
    atol = 0.05


class TestNUTSBetaBinomial(sf.NutsFixture, sf.BetaBinomialFixture):
    n_samples = 2000
    ks_thin = 5
    tune = 1000
    burn = 0
    chains = 2
    min_n_eff = 400


class TestNUTSStudentT(sf.NutsFixture, sf.StudentTFixture):
    n_samples = 10000
    tune = 1000
    burn = 0
    chains = 2
    min_n_eff = 1000
    rtol = 0.1
    atol = 0.05


@pytest.mark.skip("Takes too long to run")
class TestNUTSNormalLong(sf.NutsFixture, sf.NormalFixture):
    n_samples = 500000
    tune = 5000
    burn = 0
    chains = 2
    min_n_eff = 300000
    rtol = 0.01
    atol = 0.001


class TestNUTSLKJCholeskyCov(sf.NutsFixture, sf.LKJCholeskyCovFixture):
    n_samples = 2000
    tune = 1000
    burn = 0
    chains = 2
    min_n_eff = 200


class TestNutsCheckTrace:
    def test_multiple_samplers(self, caplog):
        with pm.Model():
            prob = pm.Beta("prob", alpha=5.0, beta=3.0)
            pm.Binomial("outcome", n=1, p=prob)
            caplog.clear()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                pm.sample(3, tune=2, discard_tuned_samples=False, n_init=None, chains=1)
            messages = [msg.msg for msg in caplog.records]
            assert all("boolean index did not" not in msg for msg in messages)

    def test_bad_init_nonparallel(self):
        with pm.Model():
            pm.HalfNormal("a", sigma=1, initval=-1, transform=None)
            with pytest.raises(SamplingError) as error:
                pm.sample(chains=1, random_seed=1)
            error.match("Initial evaluation")

    @pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
    def test_bad_init_parallel(self):
        with pm.Model():
            pm.HalfNormal("a", sigma=1, initval=-1, transform=None)
            with pytest.raises(SamplingError) as error:
                pm.sample(cores=2, random_seed=1)
            error.match("Initial evaluation")

    def test_emits_energy_warnings(self, caplog):
        with pm.Model():
            a = pm.Normal("a", size=2, initval=floatX(np.zeros(2)))
            a = at.switch(a > 0, np.inf, a)
            b = at.slinalg.solve(floatX(np.eye(2)), a, check_finite=False)
            pm.Normal("c", mu=b, size=2, initval=floatX(np.r_[0.0, 0.0]))
            caplog.clear()
            # The logger name must be specified for DEBUG level capturing to work
            with caplog.at_level(logging.DEBUG, logger="pymc"):
                idata = pm.sample(20, tune=5, chains=2, random_seed=526)
            assert any("Energy change" in w.msg for w in caplog.records)

    def test_sampler_stats(self):
        with pm.Model() as model:
            pm.Normal("x", mu=0, sigma=1)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
                trace = pm.sample(draws=10, tune=1, chains=1, return_inferencedata=False)

        # Assert stats exist and have the correct shape.
        expected_stat_names = {
            "depth",
            "diverging",
            "energy",
            "energy_error",
            "model_logp",
            "max_energy_error",
            "mean_tree_accept",
            "step_size",
            "step_size_bar",
            "tree_size",
            "tune",
            "perf_counter_diff",
            "perf_counter_start",
            "process_time_diff",
            "reached_max_treedepth",
            "index_in_trajectory",
            "largest_eigval",
            "smallest_eigval",
            "warning",
        }
        assert trace.stat_names == expected_stat_names
        for varname in trace.stat_names:
            if varname == "warning":
                # Warnings don't squeeze reliably.
                # But once we stop squeezing alltogether that's going to be OK.
                # See https://github.com/pymc-devs/pymc/issues/6207
                continue
            assert trace.get_sampler_stats(varname).shape == (10,)

        # Assert model logp is computed correctly: computing post-sampling
        # and tracking while sampling should give same results.
        model_logp_fn = model.compile_logp()
        model_logp_ = np.array(
            [
                model_logp_fn(trace.point(i, chain=c))
                for c in trace.chains
                for i in range(len(trace))
            ]
        )
        assert (trace.model_logp == model_logp_).all()


class TestStepNUTS(StepMethodTester):
    @pytest.mark.parametrize(
        "step_fn, draws",
        [
            (lambda C, _: NUTS(scaling=C, is_cov=True, blocked=False), 1000),
            (lambda C, _: NUTS(scaling=C, is_cov=True), 1000),
        ],
    )
    def test_step_continuous(self, step_fn, draws):
        self.step_continuous(step_fn, draws)


class TestRVsAssignmentNUTS(RVsAssignmentStepsTester):
    @pytest.mark.parametrize("step, step_kwargs", [(NUTS, {})])
    def test_continuous_steps(self, step, step_kwargs):
        self.continuous_steps(step, step_kwargs)
