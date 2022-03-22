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

import shutil
import sys
import tempfile

import aesara
import aesara.tensor as at
import arviz as az
import numpy as np
import numpy.testing as npt
import pytest

from aesara.compile.ops import as_op
from aesara.graph.op import Op

import pymc as pm

from pymc.aesaraf import floatX
from pymc.data import Data
from pymc.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    HalfNormal,
    MvNormal,
    Normal,
)
from pymc.exceptions import SamplingError
from pymc.model import Model, Potential, set_data
from pymc.sampling import assign_step_methods, sample
from pymc.step_methods import (
    MLDA,
    NUTS,
    BinaryGibbsMetropolis,
    CategoricalGibbsMetropolis,
    CompoundStep,
    DEMetropolis,
    DEMetropolisZ,
    EllipticalSlice,
    HamiltonianMC,
    Metropolis,
    MultivariateNormalProposal,
    NormalProposal,
    RecursiveDAProposal,
    Slice,
    UniformProposal,
)
from pymc.step_methods.mlda import extract_Q_estimate
from pymc.tests.checks import close_to
from pymc.tests.models import (
    mv_prior_simple,
    mv_simple,
    mv_simple_coarse,
    mv_simple_discrete,
    mv_simple_very_coarse,
    simple_2model_continuous,
    simple_categorical,
)


class TestStepMethods:
    def setup_class(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_class(self):
        shutil.rmtree(self.temp_dir)

    def check_stat(self, check, idata, name):
        if hasattr(idata, "warmup_posterior"):
            group = idata.warmup_posterior
        else:
            group = idata.posterior
        for (var, stat, value, bound) in check:
            s = stat(group[var].sel(chain=0, draw=slice(2000, None)), axis=0)
            close_to(s, value, bound)

    def test_step_continuous(self):
        start, model, (mu, C) = mv_simple()
        unc = np.diag(C) ** 0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, unc, unc / 10.0))
        _, model_coarse, _ = mv_simple_coarse()
        with model:
            steps = (
                Slice(),
                HamiltonianMC(scaling=C, is_cov=True, blocked=False),
                NUTS(scaling=C, is_cov=True, blocked=False),
                Metropolis(S=C, proposal_dist=MultivariateNormalProposal, blocked=True),
                Slice(blocked=True),
                HamiltonianMC(scaling=C, is_cov=True),
                NUTS(scaling=C, is_cov=True),
                CompoundStep(
                    [
                        HamiltonianMC(scaling=C, is_cov=True),
                        HamiltonianMC(scaling=C, is_cov=True, blocked=False),
                    ]
                ),
                MLDA(
                    coarse_models=[model_coarse],
                    base_S=C,
                    base_proposal_dist=MultivariateNormalProposal,
                ),
            )
        for step in steps:
            idata = sample(
                0,
                tune=8000,
                chains=1,
                discard_tuned_samples=False,
                step=step,
                start=start,
                model=model,
                random_seed=1,
            )
            self.check_stat(check, idata, step.__class__.__name__)

    def test_step_discrete(self):
        if aesara.config.floatX == "float32":
            return  # Cannot use @skip because it only skips one iteration of the yield
        start, model, (mu, C) = mv_simple_discrete()
        unc = np.diag(C) ** 0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, unc, unc / 10.0))
        with model:
            steps = (Metropolis(S=C, proposal_dist=MultivariateNormalProposal),)
        for step in steps:
            idata = sample(
                20000, tune=0, step=step, start=start, model=model, random_seed=1, chains=1
            )
            self.check_stat(check, idata, step.__class__.__name__)

    def test_step_categorical(self):
        start, model, (mu, C) = simple_categorical()
        unc = C**0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, unc, unc / 10.0))
        with model:
            steps = (
                CategoricalGibbsMetropolis([model.x], proposal="uniform"),
                CategoricalGibbsMetropolis([model.x], proposal="proportional"),
            )
        for step in steps:
            idata = sample(8000, tune=0, step=step, start=start, model=model, random_seed=1)
            self.check_stat(check, idata, step.__class__.__name__)

    @pytest.mark.xfail(reason="EllipticalSlice not refactored for v4")
    def test_step_elliptical_slice(self):
        start, model, (K, L, mu, std, noise) = mv_prior_simple()
        unc = noise**0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, std, unc / 10.0))
        with model:
            steps = (EllipticalSlice(prior_cov=K), EllipticalSlice(prior_chol=L))
        for step in steps:
            idata = sample(
                5000, tune=0, step=step, start=start, model=model, random_seed=1, chains=1
            )
            self.check_stat(check, idata, step.__class__.__name__)


class TestMetropolisProposal:
    def test_proposal_choice(self):
        _, model, _ = mv_simple()
        with model:
            initial_point = model.compute_initial_point()
            initial_point_size = sum(initial_point[n.name].size for n in model.value_vars)

            s = np.ones(initial_point_size)
            sampler = Metropolis(S=s)
            assert isinstance(sampler.proposal_dist, NormalProposal)
            s = np.diag(s)
            sampler = Metropolis(S=s)
            assert isinstance(sampler.proposal_dist, MultivariateNormalProposal)
            s[0, 0] = -s[0, 0]
            with pytest.raises(np.linalg.LinAlgError):
                sampler = Metropolis(S=s)

    def test_mv_proposal(self):
        np.random.seed(42)
        cov = np.random.randn(5, 5)
        cov = cov.dot(cov.T)
        prop = MultivariateNormalProposal(cov)
        samples = np.array([prop() for _ in range(10000)])
        npt.assert_allclose(np.cov(samples.T), cov, rtol=0.2)


class TestCompoundStep:
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS, DEMetropolis)

    @pytest.mark.skipif(
        aesara.config.floatX == "float32", reason="Test fails on 32 bit due to linalg issues"
    )
    def test_non_blocked(self):
        """Test that samplers correctly create non-blocked compound steps."""
        _, model = simple_2model_continuous()
        with model:
            for sampler in self.samplers:
                assert isinstance(sampler(blocked=False), CompoundStep)

    @pytest.mark.skipif(
        aesara.config.floatX == "float32", reason="Test fails on 32 bit due to linalg issues"
    )
    def test_blocked(self):
        _, model = simple_2model_continuous()
        with model:
            for sampler in self.samplers:
                sampler_instance = sampler(blocked=True)
                assert not isinstance(sampler_instance, CompoundStep)
                assert isinstance(sampler_instance, sampler)


class TestAssignStepMethods:
    def test_bernoulli(self):
        """Test bernoulli distribution is assigned binary gibbs metropolis method"""
        with Model() as model:
            Bernoulli("x", 0.5)
            steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)

    def test_normal(self):
        """Test normal distribution is assigned NUTS method"""
        with Model() as model:
            Normal("x", 0, 1)
            steps = assign_step_methods(model, [])
        assert isinstance(steps, NUTS)

    def test_categorical(self):
        """Test categorical distribution is assigned categorical gibbs metropolis method"""
        with Model() as model:
            Categorical("x", np.array([0.25, 0.75]))
            steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)
        with Model() as model:
            Categorical("y", np.array([0.25, 0.70, 0.05]))
            steps = assign_step_methods(model, [])
        assert isinstance(steps, CategoricalGibbsMetropolis)

    def test_binomial(self):
        """Test binomial distribution is assigned metropolis method."""
        with Model() as model:
            Binomial("x", 10, 0.5)
            steps = assign_step_methods(model, [])
        assert isinstance(steps, Metropolis)

    def test_normal_nograd_op(self):
        """Test normal distribution without an implemented gradient is assigned slice method"""
        with Model() as model:
            x = Normal("x", 0, 1)

            # a custom Aesara Op that does not have a grad:
            is_64 = aesara.config.floatX == "float64"
            itypes = [at.dscalar] if is_64 else [at.fscalar]
            otypes = [at.dscalar] if is_64 else [at.fscalar]

            @as_op(itypes, otypes)
            def kill_grad(x):
                return x

            data = np.random.normal(size=(100,))
            Normal("y", mu=kill_grad(x), sigma=1, observed=data.astype(aesara.config.floatX))

            steps = assign_step_methods(model, [])
        assert isinstance(steps, Slice)

    def test_modify_step_methods(self):
        """Test step methods can be changed"""
        # remove nuts from step_methods
        step_methods = list(pm.STEP_METHODS)
        step_methods.remove(NUTS)
        pm.STEP_METHODS = step_methods

        with Model() as model:
            Normal("x", 0, 1)
            steps = assign_step_methods(model, [])
        assert not isinstance(steps, NUTS)

        # add back nuts
        pm.STEP_METHODS = step_methods + [NUTS]

        with Model() as model:
            Normal("x", 0, 1)
            steps = assign_step_methods(model, [])
        assert isinstance(steps, NUTS)


class TestPopulationSamplers:

    steppers = [DEMetropolis]

    def test_checks_population_size(self):
        """Test that population samplers check the population size."""
        with Model() as model:
            n = Normal("n", mu=0, sigma=1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                with pytest.raises(ValueError):
                    sample(draws=10, tune=10, chains=1, cores=1, step=step)
                # don't parallelize to make test faster
                sample(draws=10, tune=10, chains=4, cores=1, step=step)

    def test_demcmc_warning_on_small_populations(self):
        """Test that a warning is raised when n_chains <= n_dims"""
        with Model() as model:
            Normal("n", mu=0, sigma=1, size=(2, 3))
            with pytest.warns(UserWarning) as record:
                sample(
                    draws=5,
                    tune=5,
                    chains=6,
                    step=DEMetropolis(),
                    # make tests faster by not parallelizing; disable convergence warning
                    cores=1,
                    compute_convergence_checks=False,
                )

    def test_demcmc_tune_parameter(self):
        """Tests that validity of the tune setting is checked"""
        with Model() as model:
            Normal("n", mu=0, sigma=1, size=(2, 3))

            step = DEMetropolis()
            assert step.tune is None

            step = DEMetropolis(tune="scaling")
            assert step.tune == "scaling"

            step = DEMetropolis(tune="lambda")
            assert step.tune == "lambda"

            with pytest.raises(ValueError):
                DEMetropolis(tune="foo")

    def test_nonparallelized_chains_are_random(self):
        with Model() as model:
            x = Normal("x", 0, 1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                idata = sample(chains=4, cores=1, draws=20, tune=0, step=DEMetropolis())
                samples = idata.posterior["x"].values[:, 5]

                assert len(set(samples)) == 4, f"Parallelized {stepper} chains are identical."

    def test_parallelized_chains_are_random(self):
        with Model() as model:
            x = Normal("x", 0, 1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                idata = sample(chains=4, cores=4, draws=20, tune=0, step=DEMetropolis())
                samples = idata.posterior["x"].values[:, 5]

                assert len(set(samples)) == 4, f"Parallelized {stepper} chains are identical."


class TestMetropolis:
    def test_tuning_reset(self):
        """Re-use of the step method instance with cores=1 must not leak tuning information between chains."""
        with Model() as pmodel:
            D = 3
            Normal("n", 0, 2, size=(D,))
            idata = sample(
                tune=600,
                draws=500,
                step=Metropolis(tune=True, scaling=0.1),
                cores=1,
                chains=3,
                discard_tuned_samples=False,
            )
        for c in idata.posterior.chain:
            # check that the tuned settings changed and were reset
            assert idata.warmup_sample_stats["scaling"].sel(chain=c).values[0] == 0.1
            tuned = idata.warmup_sample_stats["scaling"].sel(chain=c).values[-1]
            assert tuned != 0.1
            np.testing.assert_array_equal(idata.sample_stats["scaling"].sel(chain=c).values, tuned)


class TestDEMetropolisZ:
    def test_tuning_lambda_sequential(self):
        with Model() as pmodel:
            Normal("n", 0, 2, size=(3,))
            idata = sample(
                tune=1000,
                draws=500,
                step=DEMetropolisZ(tune="lambda", lamb=0.92),
                cores=1,
                chains=3,
                discard_tuned_samples=False,
            )
        for c in idata.posterior.chain:
            # check that the tuned settings changed and were reset
            assert idata.warmup_sample_stats["lambda"].sel(chain=c).values[0] == 0.92
            tuned = idata.warmup_sample_stats["lambda"].sel(chain=c).values[-1]
            assert tuned != 0.92
            np.testing.assert_array_equal(idata.sample_stats["lambda"].sel(chain=c).values, tuned)

    def test_tuning_epsilon_parallel(self):
        with Model() as pmodel:
            Normal("n", 0, 2, size=(3,))
            idata = sample(
                tune=1000,
                draws=500,
                step=DEMetropolisZ(tune="scaling", scaling=0.002),
                cores=2,
                chains=2,
                discard_tuned_samples=False,
            )
        for c in idata.posterior.chain:
            # check that the tuned settings changed and were reset
            assert idata.warmup_sample_stats["scaling"].sel(chain=c).values[0] == 0.002
            tuned = idata.warmup_sample_stats["scaling"].sel(chain=c).values[-1]
            assert tuned != 0.002
            np.testing.assert_array_equal(idata.sample_stats["scaling"].sel(chain=c).values, tuned)

    def test_tuning_none(self):
        with Model() as pmodel:
            Normal("n", 0, 2, size=(3,))
            idata = sample(
                tune=1000,
                draws=500,
                step=DEMetropolisZ(tune=None),
                cores=1,
                chains=2,
                discard_tuned_samples=False,
            )
        for c in idata.posterior.chain:
            # check that all tunable parameters remained constant
            assert len(set(idata.warmup_sample_stats["lambda"].sel(chain=c).values)) == 1
            assert len(set(idata.warmup_sample_stats["scaling"].sel(chain=c).values)) == 1

    def test_tuning_reset(self):
        """Re-use of the step method instance with cores=1 must not leak tuning information between chains."""
        with Model() as pmodel:
            D = 3
            Normal("n", 0, 2, size=(D,))
            idata = sample(
                tune=1000,
                draws=500,
                step=DEMetropolisZ(tune="scaling", scaling=0.002),
                cores=1,
                chains=3,
                discard_tuned_samples=False,
            )
        for c in idata.posterior.chain:
            # check that the tuned settings changed and were reset
            warmup = idata.warmup_sample_stats["scaling"].sel(chain=c).values
            assert warmup[0] == 0.002
            assert warmup[-1] != 0.002
            # check that the variance of the first 50 iterations is much lower than the last 100
            samples = idata.warmup_posterior["n"].sel(chain=c).values
            for d in range(D):
                var_start = np.var(samples[:50, d])
                var_end = np.var(samples[-100:, d])
                assert var_start < 0.1 * var_end

    def test_tune_drop_fraction(self):
        tune = 300
        tune_drop_fraction = 0.85
        draws = 200
        with Model() as pmodel:
            Normal("n", 0, 2, size=(3,))
            step = DEMetropolisZ(tune_drop_fraction=tune_drop_fraction)
            idata = sample(
                tune=tune, draws=draws, step=step, cores=1, chains=1, discard_tuned_samples=False
            )
            assert len(idata.warmup_posterior.draw) == tune
            assert len(idata.posterior.draw) == draws
            assert len(step._history) == (tune - tune * tune_drop_fraction) + draws

    @pytest.mark.parametrize(
        "variable,has_grad,outcome",
        [("n", True, 1), ("n", False, 1), ("b", True, 0), ("b", False, 0)],
    )
    def test_competence(self, variable, has_grad, outcome):
        with Model() as pmodel:
            Normal("n", 0, 2, size=(3,))
            Binomial("b", n=2, p=0.3)
        assert DEMetropolisZ.competence(pmodel[variable], has_grad=has_grad) == outcome

    @pytest.mark.parametrize("tune_setting", ["foo", True, False])
    def test_invalid_tune(self, tune_setting):
        with Model() as pmodel:
            Normal("n", 0, 2, size=(3,))
            with pytest.raises(ValueError):
                DEMetropolisZ(tune=tune_setting)

    def test_custom_proposal_dist(self):
        with Model() as pmodel:
            D = 3
            Normal("n", 0, 2, size=(D,))
            trace = sample(
                tune=100,
                draws=50,
                step=DEMetropolisZ(proposal_dist=NormalProposal),
                cores=1,
                chains=3,
                discard_tuned_samples=False,
            )


class TestNutsCheckTrace:
    def test_multiple_samplers(self, caplog):
        with Model():
            prob = Beta("prob", alpha=5.0, beta=3.0)
            Binomial("outcome", n=1, p=prob)
            caplog.clear()
            sample(3, tune=2, discard_tuned_samples=False, n_init=None, chains=1)
            messages = [msg.msg for msg in caplog.records]
            assert all("boolean index did not" not in msg for msg in messages)

    def test_bad_init_nonparallel(self):
        with Model():
            HalfNormal("a", sigma=1, initval=-1, transform=None)
            with pytest.raises(SamplingError) as error:
                sample(chains=1, random_seed=1)
            error.match("Initial evaluation")

    @pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
    def test_bad_init_parallel(self):
        with Model():
            HalfNormal("a", sigma=1, initval=-1, transform=None)
            with pytest.raises(SamplingError) as error:
                sample(cores=2, random_seed=1)
            error.match("Initial evaluation")

    def test_linalg(self, caplog):
        with Model():
            a = Normal("a", size=2, initval=floatX(np.zeros(2)))
            a = at.switch(a > 0, np.inf, a)
            b = at.slinalg.solve(floatX(np.eye(2)), a, check_finite=False)
            Normal("c", mu=b, size=2, initval=floatX(np.r_[0.0, 0.0]))
            caplog.clear()
            trace = sample(20, tune=5, chains=2, return_inferencedata=False)
            warns = [msg.msg for msg in caplog.records]
            assert np.any(trace["diverging"])
            assert (
                any("divergence after tuning" in warn for warn in warns)
                or any("divergences after tuning" in warn for warn in warns)
                or any("only diverging samples" in warn for warn in warns)
            )

            with pytest.raises(ValueError) as error:
                trace.report.raise_ok()
            error.match("issues during sampling")

            assert not trace.report.ok

    def test_sampler_stats(self):
        with Model() as model:
            Normal("x", mu=0, sigma=1)
            trace = sample(draws=10, tune=1, chains=1, return_inferencedata=False)

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
        }
        assert trace.stat_names == expected_stat_names
        for varname in trace.stat_names:
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


class TestMLDA:
    steppers = [MLDA]

    def test_proposal_and_base_proposal_choice(self):
        """Test that proposal_dist and base_proposal_dist are set as
        expected by MLDA"""
        _, model, _ = mv_simple()
        _, model_coarse, _ = mv_simple_coarse()
        with model:
            sampler = MLDA(coarse_models=[model_coarse], base_sampler="Metropolis")
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, NormalProposal)

            sampler = MLDA(coarse_models=[model_coarse])
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, UniformProposal)

            initial_point = model.compute_initial_point()
            initial_point_size = sum(initial_point[n.name].size for n in model.value_vars)
            s = np.ones(initial_point_size)
            sampler = MLDA(coarse_models=[model_coarse], base_sampler="Metropolis", base_S=s)
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, NormalProposal)

            sampler = MLDA(coarse_models=[model_coarse], base_S=s)
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, UniformProposal)

            s = np.diag(s)
            sampler = MLDA(coarse_models=[model_coarse], base_sampler="Metropolis", base_S=s)
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, MultivariateNormalProposal)

            sampler = MLDA(coarse_models=[model_coarse], base_S=s)
            assert isinstance(sampler.proposal_dist, RecursiveDAProposal)
            assert sampler.base_proposal_dist is None
            assert isinstance(sampler.step_method_below.proposal_dist, UniformProposal)

            s[0, 0] = -s[0, 0]
            with pytest.raises(np.linalg.LinAlgError):
                MLDA(coarse_models=[model_coarse], base_sampler="Metropolis", base_S=s)

    def test_step_methods_in_each_level(self):
        """Test that MLDA creates the correct hierarchy of step methods when no
        coarse models are passed and when two coarse models are passed."""
        _, model, _ = mv_simple()
        _, model_coarse, _ = mv_simple_coarse()
        _, model_very_coarse, _ = mv_simple_very_coarse()
        with model:
            initial_point = model.compute_initial_point()
            initial_point_size = sum(initial_point[n.name].size for n in model.value_vars)
            s = np.ones(initial_point_size) + 2.0
            sampler = MLDA(
                coarse_models=[model_very_coarse, model_coarse],
                base_S=s,
                base_sampler="Metropolis",
            )
            assert isinstance(sampler.step_method_below, MLDA)
            assert isinstance(sampler.step_method_below.step_method_below, Metropolis)
            assert np.all(sampler.step_method_below.step_method_below.proposal_dist.s == s)

            sampler = MLDA(coarse_models=[model_very_coarse, model_coarse], base_S=s)
            assert isinstance(sampler.step_method_below, MLDA)
            assert isinstance(sampler.step_method_below.step_method_below, DEMetropolisZ)
            assert np.all(sampler.step_method_below.step_method_below.proposal_dist.s == s)

    def test_exceptions_coarse_models(self):
        """Test that MLDA generates the expected exceptions when no coarse_models arg
        is passed, an empty list is passed or when coarse_models is not a list"""
        with pytest.raises(TypeError):
            _, model, _ = mv_simple()
            with model:
                MLDA()

        with pytest.raises(ValueError):
            _, model, _ = mv_simple()
            with model:
                MLDA(coarse_models=[])

        with pytest.raises(ValueError):
            _, model, _ = mv_simple()
            with model:
                MLDA(coarse_models=(model, model))

    def test_nonparallelized_chains_are_random(self):
        """Test that parallel chain are not identical when no parallelisation
        is applied"""
        with Model() as coarse_model:
            Normal("x", 0.3, 1)

        with Model():
            Normal("x", 0, 1)
            for stepper in TestMLDA.steppers:
                step = stepper(coarse_models=[coarse_model])
                idata = sample(chains=2, cores=1, draws=20, tune=0, step=step, random_seed=1)
                samples = idata.posterior["x"].values[:, 5]
                assert len(set(samples)) == 2, f"Non parallelized {stepper} chains are identical."

    def test_parallelized_chains_are_random(self):
        """Test that parallel chain are
        not identical when parallelisation
        is applied"""
        with Model() as coarse_model:
            Normal("x", 0.3, 1)

        with Model():
            Normal("x", 0, 1)
            for stepper in TestMLDA.steppers:
                step = stepper(coarse_models=[coarse_model])
                idata = sample(chains=2, cores=2, draws=20, tune=0, step=step, random_seed=1)
                samples = idata.posterior["x"].values[:, 5]
                assert len(set(samples)) == 2, f"Parallelized {stepper} chains are identical."

    def test_acceptance_rate_against_coarseness(self):
        """Test that the acceptance rate increases
        when the coarse model is closer to
        the fine model."""
        with Model() as coarse_model_0:
            Normal("x", 5.0, 1.0)

        with Model() as coarse_model_1:
            Normal("x", 6.0, 2.0)

        with Model() as coarse_model_2:
            Normal("x", 20.0, 5.0)

        possible_coarse_models = [coarse_model_0, coarse_model_1, coarse_model_2]
        acc = []

        with Model():
            Normal("x", 5.0, 1.0)
            for coarse_model in possible_coarse_models:
                step = MLDA(coarse_models=[coarse_model], subsampling_rates=3)
                idata = sample(chains=1, draws=500, tune=100, step=step, random_seed=1)
                acc.append(idata.sample_stats["accepted"].mean())
            assert acc[0] > acc[1] > acc[2], (
                "Acceptance rate is not "
                "strictly increasing when"
                "coarse model is closer to "
                "fine model. Acceptance rates"
                "were: {}".format(acc)
            )

    def test_mlda_non_blocked(self):
        """Test that MLDA correctly creates non-blocked
        compound steps in level 0 when using a Metropolis
        base sampler."""
        _, model = simple_2model_continuous()
        _, model_coarse = simple_2model_continuous()
        with model:
            for stepper in self.steppers:
                assert isinstance(
                    stepper(
                        coarse_models=[model_coarse],
                        base_sampler="Metropolis",
                        base_blocked=False,
                    ).step_method_below,
                    CompoundStep,
                )

    def test_mlda_blocked(self):
        """Test the type of base sampler instantiated
        when switching base_blocked flag while
        the base sampler is Metropolis and when
        the base sampler is DEMetropolisZ."""
        _, model = simple_2model_continuous()
        _, model_coarse = simple_2model_continuous()
        with model:
            for stepper in self.steppers:
                assert not isinstance(
                    stepper(
                        coarse_models=[model_coarse],
                        base_sampler="Metropolis",
                        base_blocked=True,
                    ).step_method_below,
                    CompoundStep,
                )
                assert isinstance(
                    stepper(
                        coarse_models=[model_coarse],
                        base_sampler="Metropolis",
                        base_blocked=True,
                    ).step_method_below,
                    Metropolis,
                )
                assert isinstance(
                    stepper(coarse_models=[model_coarse]).step_method_below,
                    DEMetropolisZ,
                )

    def test_tuning_and_scaling_on(self):
        """Test that tune and base_scaling change as expected when
        tuning is on."""
        np.random.seed(1234)
        ts = 100
        _, model = simple_2model_continuous()
        _, model_coarse = simple_2model_continuous()
        with model:
            trace_0 = sample(
                tune=ts,
                draws=20,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_sampler="Metropolis",
                    base_tune_interval=50,
                    base_scaling=100.0,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=1234,
                return_inferencedata=False,
            )

            trace_1 = sample(
                tune=ts,
                draws=20,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_tune_target="scaling",
                    base_tune_interval=50,
                    base_scaling=100.0,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=1234,
                return_inferencedata=False,
            )

            trace_2 = sample(
                tune=ts,
                draws=20,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_tune_interval=50,
                    base_scaling=10,
                    base_lamb=100.0,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=1234,
                return_inferencedata=False,
            )

        assert trace_0.get_sampler_stats("tune", chains=0)[0]
        assert trace_0.get_sampler_stats("tune", chains=0)[ts - 1]
        assert not trace_0.get_sampler_stats("tune", chains=0)[ts]
        assert not trace_0.get_sampler_stats("tune", chains=0)[-1]
        assert trace_0.get_sampler_stats("base_scaling", chains=0)[0, 0] == 100.0
        assert trace_0.get_sampler_stats("base_scaling", chains=0)[0, 1] == 100.0
        assert trace_0.get_sampler_stats("base_scaling", chains=0)[-1, 0] < 100.0
        assert trace_0.get_sampler_stats("base_scaling", chains=0)[-1, 1] < 100.0

        assert trace_1.get_sampler_stats("tune", chains=0)[0]
        assert trace_1.get_sampler_stats("tune", chains=0)[ts - 1]
        assert not trace_1.get_sampler_stats("tune", chains=0)[ts]
        assert not trace_1.get_sampler_stats("tune", chains=0)[-1]
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[0] == 100.0
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[-1] < 100.0

        assert trace_2.get_sampler_stats("tune", chains=0)[0]
        assert trace_2.get_sampler_stats("tune", chains=0)[ts - 1]
        assert not trace_2.get_sampler_stats("tune", chains=0)[ts]
        assert not trace_2.get_sampler_stats("tune", chains=0)[-1]
        assert trace_2.get_sampler_stats("base_lambda", chains=0)[0] == 100.0
        assert trace_2.get_sampler_stats("base_lambda", chains=0)[-1] < 100.0

    def test_tuning_and_scaling_off(self):
        """Test that tuning is deactivated when sample()'s tune=0 and that
        MLDA's tune=False is overridden by sample()'s tune."""
        np.random.seed(12345)
        _, model = simple_2model_continuous()
        _, model_coarse = simple_2model_continuous()

        ts_0 = 0
        with model:
            trace_0 = sample(
                tune=ts_0,
                draws=100,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_sampler="Metropolis",
                    base_tune_interval=50,
                    base_scaling=100.0,
                    tune=False,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=12345,
                return_inferencedata=False,
            )

        ts_1 = 100
        with model:
            trace_1 = sample(
                tune=ts_1,
                draws=20,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_sampler="Metropolis",
                    base_tune_interval=50,
                    base_scaling=100.0,
                    tune=False,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=12345,
                return_inferencedata=False,
            )

        assert not trace_0.get_sampler_stats("tune", chains=0)[0]
        assert not trace_0.get_sampler_stats("tune", chains=0)[-1]
        assert (
            trace_0.get_sampler_stats("base_scaling", chains=0)[0, 0]
            == trace_0.get_sampler_stats("base_scaling", chains=0)[-1, 0]
            == trace_0.get_sampler_stats("base_scaling", chains=0)[0, 1]
            == trace_0.get_sampler_stats("base_scaling", chains=0)[-1, 1]
            == 100.0
        )

        assert trace_1.get_sampler_stats("tune", chains=0)[0]
        assert trace_1.get_sampler_stats("tune", chains=0)[ts_1 - 1]
        assert not trace_1.get_sampler_stats("tune", chains=0)[ts_1]
        assert not trace_1.get_sampler_stats("tune", chains=0)[-1]
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[0, 0] == 100.0
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[0, 1] == 100.0
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[-1, 0] < 100.0
        assert trace_1.get_sampler_stats("base_scaling", chains=0)[-1, 1] < 100.0

        ts_2 = 0
        with model:
            trace_2 = sample(
                tune=ts_2,
                draws=100,
                step=MLDA(
                    coarse_models=[model_coarse],
                    base_tune_interval=50,
                    base_lamb=100.0,
                    base_tune_target=None,
                ),
                chains=1,
                discard_tuned_samples=False,
                random_seed=12345,
                return_inferencedata=False,
            )

        assert not trace_2.get_sampler_stats("tune", chains=0)[0]
        assert not trace_2.get_sampler_stats("tune", chains=0)[-1]
        assert (
            trace_2.get_sampler_stats("base_lambda", chains=0)[0]
            == trace_2.get_sampler_stats("base_lambda", chains=0)[-1]
            == trace_2.get_sampler_stats("base_lambda", chains=0)[0]
            == trace_2.get_sampler_stats("base_lambda", chains=0)[-1]
            == 100.0
        )

    def test_trace_length(self):
        """Check if trace length is as expected."""
        tune = 100
        draws = 50
        with Model() as coarse_model:
            Normal("n", 0, 2.2, size=(3,))
        with Model():
            Normal("n", 0, 2, size=(3,))
            step = MLDA(coarse_models=[coarse_model])
            idata = sample(tune=tune, draws=draws, step=step, chains=1, discard_tuned_samples=False)
            assert len(idata.warmup_posterior.draw) == tune
            assert len(idata.posterior.draw) == draws

    @pytest.mark.parametrize(
        "variable,has_grad,outcome",
        [("n", True, 1), ("n", False, 1), ("b", True, 0), ("b", False, 0)],
    )
    def test_competence(self, variable, has_grad, outcome):
        """Test if competence function returns expected
        results for different models"""
        with Model() as pmodel:
            Normal("n", 0, 2, size=(3,))
            Binomial("b", n=2, p=0.3)
        assert MLDA.competence(pmodel[variable], has_grad=has_grad) == outcome

    def test_multiple_subsampling_rates(self):
        """Test that when you give a single integer it is applied to all levels and
        when you give a list the list is applied correctly."""
        with Model() as coarse_model_0:
            Normal("n", 0, 2.2, size=(3,))
        with Model() as coarse_model_1:
            Normal("n", 0, 2.1, size=(3,))
        with Model():
            Normal("n", 0, 2.0, size=(3,))

            step_1 = MLDA(coarse_models=[coarse_model_0, coarse_model_1], subsampling_rates=3)
            assert len(step_1.subsampling_rates) == 2
            assert step_1.subsampling_rates[0] == step_1.subsampling_rates[1] == 3

            step_2 = MLDA(coarse_models=[coarse_model_0, coarse_model_1], subsampling_rates=[3, 4])
            assert step_2.subsampling_rates[0] == 3
            assert step_2.subsampling_rates[1] == 4

            with pytest.raises(ValueError):
                step_3 = MLDA(
                    coarse_models=[coarse_model_0, coarse_model_1],
                    subsampling_rates=[3, 4, 10],
                )

    def test_aem_mu_sigma(self):
        """Test that AEM estimates mu_B and Sigma_B in
        the coarse models of a 3-level LR example correctly"""
        # create data for linear regression
        if aesara.config.floatX == "float32":
            p = "float32"
        else:
            p = "float64"
        np.random.seed(123456)
        size = 200
        true_intercept = 1
        true_slope = 2
        sigma = 1
        x = np.linspace(0, 1, size, dtype=p)
        # y = a + b*x
        true_regression_line = true_intercept + true_slope * x
        # add noise
        y = true_regression_line + np.random.normal(0, sigma**2, size)
        s = np.identity(y.shape[0], dtype=p)
        np.fill_diagonal(s, sigma**2)

        # forward model Op - here, just the regression equation
        class ForwardModel(Op):
            if aesara.config.floatX == "float32":
                itypes = [at.fvector]
                otypes = [at.fvector]
            else:
                itypes = [at.dvector]
                otypes = [at.dvector]

            def __init__(self, x, pymc_model):
                self.x = x
                self.pymc_model = pymc_model

            def perform(self, node, inputs, outputs):
                intercept = inputs[0][0]
                x_coeff = inputs[0][1]

                temp = intercept + x_coeff * x + self.pymc_model.bias.get_value()
                with self.pymc_model:
                    set_data({"model_output": temp})
                outputs[0][0] = np.array(temp)

        # create the coarse models with separate biases
        mout = []
        coarse_models = []

        with Model() as coarse_model_0:
            bias = Data("bias", 3.5 * np.ones(y.shape, dtype=p))
            mu_B = Data("mu_B", -1.3 * np.ones(y.shape, dtype=p))
            Sigma_B = Data("Sigma_B", np.zeros((y.shape[0], y.shape[0]), dtype=p))
            model_output = Data("model_output", np.zeros(y.shape, dtype=p))
            Sigma_e = Data("Sigma_e", s)

            # Define priors
            intercept = Normal("Intercept", 0, sigma=20)
            x_coeff = Normal("x", 0, sigma=20)

            theta = at.as_tensor_variable([intercept, x_coeff])

            mout.append(ForwardModel(x, coarse_model_0))

            # Define likelihood
            likelihood = MvNormal("y", mu=mout[0](theta) + mu_B, cov=Sigma_e, observed=y)

            coarse_models.append(coarse_model_0)

        with Model() as coarse_model_1:
            bias = Data("bias", 2.2 * np.ones(y.shape, dtype=p))
            mu_B = Data("mu_B", -2.2 * np.ones(y.shape, dtype=p))
            Sigma_B = Data("Sigma_B", np.zeros((y.shape[0], y.shape[0]), dtype=p))
            model_output = Data("model_output", np.zeros(y.shape, dtype=p))
            Sigma_e = Data("Sigma_e", s)

            # Define priors
            intercept = Normal("Intercept", 0, sigma=20)
            x_coeff = Normal("x", 0, sigma=20)

            theta = at.as_tensor_variable([intercept, x_coeff])

            mout.append(ForwardModel(x, coarse_model_1))

            # Define likelihood
            likelihood = MvNormal("y", mu=mout[1](theta) + mu_B, cov=Sigma_e, observed=y)

            coarse_models.append(coarse_model_1)

        # fine model and inference
        with Model() as model:
            bias = Data("bias", np.zeros(y.shape, dtype=p))
            model_output = Data("model_output", np.zeros(y.shape, dtype=p))
            Sigma_e = Data("Sigma_e", s)

            # Define priors
            intercept = Normal("Intercept", 0, sigma=20)
            x_coeff = Normal("x", 0, sigma=20)

            theta = at.as_tensor_variable([intercept, x_coeff])

            mout.append(ForwardModel(x, model))

            # Define likelihood
            likelihood = MvNormal("y", mu=mout[-1](theta), cov=Sigma_e, observed=y)

            step_mlda = MLDA(coarse_models=coarse_models, adaptive_error_model=True)

            trace_mlda = sample(
                draws=100,
                step=step_mlda,
                chains=1,
                tune=200,
                discard_tuned_samples=True,
                random_seed=84759238,
            )

            m0 = step_mlda.step_method_below.model_below.mu_B.get_value()
            s0 = step_mlda.step_method_below.model_below.Sigma_B.get_value()
            m1 = step_mlda.model_below.mu_B.get_value()
            s1 = step_mlda.model_below.Sigma_B.get_value()

            assert np.allclose(m0, -3.5)
            assert np.allclose(m1, -2.2)
            assert np.allclose(s0, 0, atol=1e-3)
            assert np.allclose(s1, 0, atol=1e-3)

    def test_variance_reduction(self):
        """
        Test if the right stats are outputed when variance reduction is used in MLDA,
        if the output estimates are close (VR estimate vs. standard estimate from
        the first chain) and if the variance of VR is lower. Uses a linear regression
        model with multiple levels where approximate levels have fewer data.

        """
        # arithmetic precision
        if aesara.config.floatX == "float32":
            p = "float32"
        else:
            p = "float64"

        # set up the model and data
        seed = 12345
        np.random.seed(seed)
        size = 100
        true_intercept = 1
        true_slope = 2
        sigma = 0.1
        x = np.linspace(0, 1, size, dtype=p)
        # y = a + b*x
        true_regression_line = true_intercept + true_slope * x
        # add noise
        y = true_regression_line + np.random.normal(0, sigma**2, size)
        s = sigma

        x_coarse_0 = x[::3]
        y_coarse_0 = y[::3]
        x_coarse_1 = x[::2]
        y_coarse_1 = y[::2]

        # MCMC parameters
        ndraws = 200
        ntune = 100
        nsub = 3
        nchains = 1

        # define likelihoods with different Q
        class Likelihood(Op):
            if aesara.config.floatX == "float32":
                itypes = [at.fvector]
                otypes = [at.fscalar]
            else:
                itypes = [at.dvector]
                otypes = [at.dscalar]

            def __init__(self, x, y, pymc_model):
                self.x = x
                self.y = y
                self.pymc_model = pymc_model

            def perform(self, node, inputs, outputs):
                intercept = inputs[0][0]
                x_coeff = inputs[0][1]

                temp = np.array(intercept + x_coeff * self.x, dtype=p)
                with self.pymc_model:
                    set_data({"Q": np.array(x_coeff, dtype=p)})
                outputs[0][0] = np.array(
                    -(0.5 / s**2) * np.sum((temp - self.y) ** 2, dtype=p), dtype=p
                )

        # run four MLDA steppers for all combinations of
        # base_sampler and forward model
        for stepper in ["Metropolis", "DEMetropolisZ"]:
            mout = []
            coarse_models = []

            rng = np.random.RandomState(seed)

            with Model(rng_seeder=rng) as coarse_model_0:
                if aesara.config.floatX == "float32":
                    Q = Data("Q", np.float32(0.0))
                else:
                    Q = Data("Q", np.float64(0.0))

                # Define priors
                intercept = Normal("Intercept", true_intercept, sigma=1)
                x_coeff = Normal("x", true_slope, sigma=1)

                theta = at.as_tensor_variable([intercept, x_coeff])

                mout.append(Likelihood(x_coarse_0, y_coarse_0, coarse_model_0))
                Potential("likelihood", mout[0](theta))

                coarse_models.append(coarse_model_0)

            rng = np.random.RandomState(seed)

            with Model(rng_seeder=rng) as coarse_model_1:
                if aesara.config.floatX == "float32":
                    Q = Data("Q", np.float32(0.0))
                else:
                    Q = Data("Q", np.float64(0.0))

                # Define priors
                intercept = Normal("Intercept", true_intercept, sigma=1)
                x_coeff = Normal("x", true_slope, sigma=1)

                theta = at.as_tensor_variable([intercept, x_coeff])

                mout.append(Likelihood(x_coarse_1, y_coarse_1, coarse_model_1))
                Potential("likelihood", mout[1](theta))

                coarse_models.append(coarse_model_1)

            rng = np.random.RandomState(seed)

            with Model(rng_seeder=rng) as model:
                if aesara.config.floatX == "float32":
                    Q = Data("Q", np.float32(0.0))
                else:
                    Q = Data("Q", np.float64(0.0))

                # Define priors
                intercept = Normal("Intercept", true_intercept, sigma=1)
                x_coeff = Normal("x", true_slope, sigma=1)

                theta = at.as_tensor_variable([intercept, x_coeff])

                mout.append(Likelihood(x, y, model))
                Potential("likelihood", mout[-1](theta))

                step = MLDA(
                    coarse_models=coarse_models,
                    base_sampler=stepper,
                    subsampling_rates=nsub,
                    variance_reduction=True,
                    store_Q_fine=True,
                )

                trace = sample(
                    draws=ndraws,
                    step=step,
                    chains=nchains,
                    tune=ntune,
                    cores=1,
                    discard_tuned_samples=True,
                    random_seed=seed,
                    return_inferencedata=False,
                )

                # get fine level stats (standard method)
                Q_2 = trace.get_sampler_stats("Q_2").reshape((nchains, ndraws))
                Q_mean_standard = Q_2.mean(axis=1).mean()
                Q_se_standard = np.sqrt(Q_2.var() / az.ess(np.array(Q_2, np.float64)))

                # get VR stats
                Q_mean_vr, Q_se_vr = extract_Q_estimate(trace, 3)

                # check that returned values are floats and finite.
                assert isinstance(Q_mean_standard, np.floating)
                assert np.isfinite(Q_mean_standard)
                assert isinstance(Q_mean_vr, np.floating)
                assert np.isfinite(Q_mean_vr)
                assert isinstance(Q_se_standard, np.floating)
                assert np.isfinite(Q_se_standard)
                assert isinstance(Q_se_vr, np.floating)
                assert np.isfinite(Q_se_vr)

                # check consistency of QoI across levels.
                Q_1_0 = np.concatenate(trace.get_sampler_stats("Q_1_0")).reshape(
                    (nchains, ndraws * nsub)
                )
                Q_2_1 = np.concatenate(trace.get_sampler_stats("Q_2_1")).reshape((nchains, ndraws))
                assert Q_1_0.mean(axis=1) == 0.0
                assert Q_2_1.mean(axis=1) == 0.0


class TestRVsAssignmentSteps:
    """
    Test that step methods convert input RVs to respective value vars
    Step methods are tested with one and two variables to cover compound
    the special branches in `BlockedStep.__new__`
    """

    @pytest.mark.parametrize(
        "step, step_kwargs",
        [
            (NUTS, {}),
            (HamiltonianMC, {}),
            (Metropolis, {}),
            (Slice, {}),
            (EllipticalSlice, {"prior_cov": np.eye(1)}),
            (DEMetropolis, {}),
            (DEMetropolisZ, {}),
            # (MLDA, {}),  # TODO
        ],
    )
    def test_continuous_steps(self, step, step_kwargs):
        with Model() as m:
            c1 = HalfNormal("c1")
            c2 = HalfNormal("c2")

            assert [m.rvs_to_values[c1]] == step([c1], **step_kwargs).vars
            assert {m.rvs_to_values[c1], m.rvs_to_values[c2]} == set(
                step([c1, c2], **step_kwargs).vars
            )

    @pytest.mark.parametrize(
        "step, step_kwargs",
        [
            (BinaryGibbsMetropolis, {}),
            (CategoricalGibbsMetropolis, {}),
        ],
    )
    def test_discrete_steps(self, step, step_kwargs):
        with Model() as m:
            d1 = Bernoulli("d1", p=0.5)
            d2 = Bernoulli("d2", p=0.5)

            assert [m.rvs_to_values[d1]] == step([d1], **step_kwargs).vars
            assert {m.rvs_to_values[d1], m.rvs_to_values[d2]} == set(
                step([d1, d2], **step_kwargs).vars
            )

    def test_compound_step(self):
        with Model() as m:
            c1 = HalfNormal("c1")
            c2 = HalfNormal("c2")

            step1 = NUTS([c1])
            step2 = NUTS([c2])
            step = CompoundStep([step1, step2])
            assert {m.rvs_to_values[c1], m.rvs_to_values[c2]} == set(step.vars)
