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

import pymc as pm

from pymc.aesaraf import floatX
from pymc.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    HalfNormal,
    Multinomial,
    Normal,
)
from pymc.exceptions import SamplingError
from pymc.model import Model
from pymc.sampling import assign_step_methods, sample
from pymc.step_methods import (
    NUTS,
    BinaryGibbsMetropolis,
    CategoricalGibbsMetropolis,
    CompoundStep,
    DEMetropolis,
    DEMetropolisZ,
    HamiltonianMC,
    Metropolis,
    MultivariateNormalProposal,
    NormalProposal,
    Slice,
)
from pymc.tests.checks import close_to
from pymc.tests.helpers import fast_unstable_sampling_mode
from pymc.tests.models import (
    mv_simple,
    mv_simple_coarse,
    mv_simple_discrete,
    simple_2model_continuous,
    simple_categorical,
)


class TestStepMethods:
    def setup_class(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_class(self):
        shutil.rmtree(self.temp_dir)

    def check_stat(self, check, idata, name):
        group = idata.posterior
        for (var, stat, value, bound) in check:
            s = stat(group[var].sel(chain=0), axis=0)
            close_to(s, value, bound, name)

    def check_stat_dtype(self, step, idata):
        # TODO: This check does not confirm the announced dtypes are correct as the
        #  sampling machinery will convert them automatically.
        for stats_dtypes in getattr(step, "stats_dtypes", []):
            for stat, dtype in stats_dtypes.items():
                if stat == "tune":
                    continue
                assert idata.sample_stats[stat].dtype == np.dtype(dtype)

    @pytest.mark.parametrize(
        "step_fn, draws",
        [
            (lambda C, _: HamiltonianMC(scaling=C, is_cov=True, blocked=False), 1000),
            (lambda C, _: HamiltonianMC(scaling=C, is_cov=True), 1000),
            (lambda C, _: NUTS(scaling=C, is_cov=True, blocked=False), 1000),
            (lambda C, _: NUTS(scaling=C, is_cov=True), 1000),
            (
                lambda C, _: CompoundStep(
                    [
                        HamiltonianMC(scaling=C, is_cov=True),
                        HamiltonianMC(scaling=C, is_cov=True, blocked=False),
                    ]
                ),
                1000,
            ),
            (lambda *_: Slice(), 2000),
            (lambda *_: Slice(blocked=True), 2000),
            (
                lambda C, _: Metropolis(
                    S=C, proposal_dist=MultivariateNormalProposal, blocked=True
                ),
                4000,
            ),
        ],
        ids=str,
    )
    def test_step_continuous(self, step_fn, draws):
        start, model, (mu, C) = mv_simple()
        unc = np.diag(C) ** 0.5
        check = (("x", np.mean, mu, unc / 10), ("x", np.std, unc, unc / 10))
        _, model_coarse, _ = mv_simple_coarse()
        with model:
            step = step_fn(C, model_coarse)
            idata = sample(
                tune=1000,
                draws=draws,
                chains=1,
                step=step,
                start=start,
                model=model,
                random_seed=1,
            )
            self.check_stat(check, idata, step.__class__.__name__)
            self.check_stat_dtype(idata, step)

    def test_step_discrete(self):
        start, model, (mu, C) = mv_simple_discrete()
        unc = np.diag(C) ** 0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, unc, unc / 10.0))
        with model:
            step = Metropolis(S=C, proposal_dist=MultivariateNormalProposal)
            idata = sample(
                tune=1000,
                draws=2000,
                chains=1,
                step=step,
                start=start,
                model=model,
                random_seed=1,
            )
            self.check_stat(check, idata, step.__class__.__name__)
            self.check_stat_dtype(idata, step)

    @pytest.mark.parametrize("proposal", ["uniform", "proportional"])
    def test_step_categorical(self, proposal):
        start, model, (mu, C) = simple_categorical()
        unc = C**0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, unc, unc / 10.0))
        with model:
            step = CategoricalGibbsMetropolis([model.x], proposal=proposal)
            idata = sample(
                tune=1000,
                draws=2000,
                chains=1,
                step=step,
                start=start,
                model=model,
                random_seed=1,
            )
            self.check_stat(check, idata, step.__class__.__name__)
            self.check_stat_dtype(idata, step)


class TestCompoundStep:
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS, DEMetropolis)

    def test_non_blocked(self):
        """Test that samplers correctly create non-blocked compound steps."""
        with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
            _, model = simple_2model_continuous()
            with model:
                for sampler in self.samplers:
                    assert isinstance(sampler(blocked=False), CompoundStep)

    def test_blocked(self):
        with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
            _, model = simple_2model_continuous()
            with model:
                for sampler in self.samplers:
                    sampler_instance = sampler(blocked=True)
                    assert not isinstance(sampler_instance, CompoundStep)
                    assert isinstance(sampler_instance, sampler)

    def test_name(self):
        with Model() as m:
            c1 = HalfNormal("c1")
            c2 = HalfNormal("c2")

            step1 = NUTS([c1])
            step2 = Slice([c2])
            step = CompoundStep([step1, step2])
        assert step.name == "Compound[nuts, slice]"


class TestAssignStepMethods:
    def test_bernoulli(self):
        """Test bernoulli distribution is assigned binary gibbs metropolis method"""
        with Model() as model:
            Bernoulli("x", 0.5)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)

    def test_normal(self):
        """Test normal distribution is assigned NUTS method"""
        with Model() as model:
            Normal("x", 0, 1)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, NUTS)

    def test_categorical(self):
        """Test categorical distribution is assigned categorical gibbs metropolis method"""
        with Model() as model:
            Categorical("x", np.array([0.25, 0.75]))
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)
        with Model() as model:
            Categorical("y", np.array([0.25, 0.70, 0.05]))
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert isinstance(steps, CategoricalGibbsMetropolis)

    def test_binomial(self):
        """Test binomial distribution is assigned metropolis method."""
        with Model() as model:
            Binomial("x", 10, 0.5)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
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

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
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
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                steps = assign_step_methods(model, [])
        assert not isinstance(steps, NUTS)

        # add back nuts
        pm.STEP_METHODS = step_methods + [NUTS]

        with Model() as model:
            Normal("x", 0, 1)
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
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
    def test_proposal_choice(self):
        with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
            _, model, _ = mv_simple()
            with model:
                initial_point = model.initial_point()
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

    @pytest.mark.parametrize(
        "batched_dist",
        (
            Binomial.dist(n=5, p=0.9),  # scalar case
            Binomial.dist(n=np.arange(40) + 1, p=np.linspace(0.1, 0.9, 40), shape=(40,)),
            Binomial.dist(
                n=(np.arange(20) + 1)[::-1],
                p=np.linspace(0.1, 0.9, 20),
                shape=(
                    2,
                    20,
                ),
            ),
            Dirichlet.dist(a=np.ones(3) * (np.arange(40) + 1)[:, None], shape=(40, 3)),
            Dirichlet.dist(a=np.ones(3) * (np.arange(20) + 1)[:, None], shape=(2, 20, 3)),
        ),
    )
    def test_elemwise_update(self, batched_dist):
        with Model() as m:
            m.register_rv(batched_dist, name="batched_dist")
            step = pm.Metropolis([batched_dist])
            assert step.elemwise_update == (batched_dist.ndim > 0)
            trace = pm.sample(draws=1000, chains=2, step=step, random_seed=428)

        assert az.rhat(trace).max()["batched_dist"].values < 1.1
        assert az.ess(trace).min()["batched_dist"].values > 50

    def test_multinomial_no_elemwise_update(self):
        with Model() as m:
            batched_dist = Multinomial("batched_dist", n=5, p=np.ones(4) / 4, shape=(10, 4))
            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                step = pm.Metropolis([batched_dist])
                assert not step.elemwise_update


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
            trace = sample(20, tune=5, chains=2, return_inferencedata=False, random_seed=526)
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
            "index_in_trajectory",
            "largest_eigval",
            "smallest_eigval",
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
            (DEMetropolis, {}),
            (DEMetropolisZ, {}),
        ],
    )
    def test_continuous_steps(self, step, step_kwargs):
        with Model() as m:
            c1 = HalfNormal("c1")
            c2 = HalfNormal("c2")

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
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

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                assert [m.rvs_to_values[d1]] == step([d1], **step_kwargs).vars
            assert {m.rvs_to_values[d1], m.rvs_to_values[d2]} == set(
                step([d1, d2], **step_kwargs).vars
            )

    def test_compound_step(self):
        with Model() as m:
            c1 = HalfNormal("c1")
            c2 = HalfNormal("c2")

            with aesara.config.change_flags(mode=fast_unstable_sampling_mode):
                step1 = NUTS([c1])
                step2 = NUTS([c2])
                step = CompoundStep([step1, step2])
            assert {m.rvs_to_values[c1], m.rvs_to_values[c2]} == set(step.vars)
