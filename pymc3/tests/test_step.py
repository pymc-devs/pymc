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
import tempfile
import sys

from .checks import close_to
from .models import (
    simple_categorical,
    mv_simple,
    mv_simple_discrete,
    mv_prior_simple,
    simple_2model_continuous,
)
from pymc3.sampling import assign_step_methods, sample
from pymc3.parallel_sampling import ParallelSamplingError
from pymc3.exceptions import SamplingError
from pymc3.model import Model
from pymc3.step_methods import (
    NUTS,
    BinaryGibbsMetropolis,
    CategoricalGibbsMetropolis,
    Metropolis,
    Slice,
    CompoundStep,
    NormalProposal,
    MultivariateNormalProposal,
    HamiltonianMC,
    EllipticalSlice,
    DEMetropolis,
    DEMetropolisZ,
)
from pymc3.theanof import floatX
from pymc3.distributions import Binomial, Normal, Bernoulli, Categorical, Beta, HalfNormal

from numpy.testing import assert_array_almost_equal
import numpy as np
import numpy.testing as npt
import pytest
import theano
import theano.tensor as tt
from .helpers import select_by_precision


class TestStepMethods:  # yield test doesn't work subclassing object
    master_samples = {
        Slice: np.array(
            [
                0.10233528,
                0.40458486,
                0.17329217,
                0.46281232,
                0.22556278,
                1.52632836,
                -0.27823807,
                0.02539625,
                1.02711735,
                0.03686346,
                -0.62841281,
                -0.27125083,
                0.31989505,
                0.84031155,
                -0.18949138,
                1.60550262,
                1.01375291,
                -0.29742941,
                0.35312738,
                0.43363622,
                1.18898078,
                0.80063888,
                0.38445644,
                0.90184395,
                1.69150017,
                2.05452171,
                -0.13334755,
                1.61265408,
                1.36579345,
                1.3216292,
                -0.59487037,
                -0.34648927,
                1.05107285,
                0.42870305,
                0.61552257,
                0.55239884,
                0.13929271,
                0.26213809,
                -0.2316028,
                0.19711046,
                1.42832629,
                1.93641434,
                -0.81142379,
                -0.31059485,
                -0.3189694,
                1.43542534,
                0.40311093,
                1.63103768,
                0.24034874,
                0.33924866,
                0.94951616,
                0.71700185,
                0.79273056,
                -0.44569146,
                1.91974783,
                0.84673795,
                1.12411833,
                -0.83123811,
                -0.54310095,
                -0.00721347,
                0.9925055,
                1.04015058,
                -0.34958074,
                -0.14926302,
                -0.47990225,
                -0.75629446,
                -0.95942067,
                1.68179204,
                1.20598073,
                1.39675733,
                1.22755935,
                0.06728757,
                1.05184231,
                1.01126791,
                -0.67327093,
                0.21429651,
                1.33730461,
                -1.56174184,
                -0.64348764,
                0.98050636,
                0.25923049,
                0.58622631,
                0.46589069,
                1.44367347,
                -0.43141573,
                1.08293374,
                -0.5563204,
                1.46287904,
                1.26019815,
                0.52972104,
                1.08792687,
                1.10064358,
                1.84881549,
                0.91179647,
                0.69316592,
                -0.47657064,
                2.22747063,
                0.83388935,
                0.84680716,
                -0.10556406,
            ]
        ),
        HamiltonianMC: np.array(
            [
                1.43583525,
                1.43583525,
                1.43583525,
                -0.57415005,
                0.91472062,
                0.91472062,
                0.36282799,
                0.80991631,
                0.84457253,
                0.84457253,
                -0.12651784,
                -0.12651784,
                0.39027088,
                -0.22998424,
                0.64337475,
                0.64337475,
                0.03504003,
                1.2667789,
                1.2667789,
                0.34770874,
                0.224319,
                0.224319,
                1.00416894,
                0.46161403,
                0.28217305,
                0.28217305,
                0.50327811,
                0.50327811,
                0.50327811,
                0.50327811,
                0.42335724,
                0.42335724,
                0.20336198,
                0.20336198,
                0.20336198,
                0.16330229,
                0.16330229,
                -0.7332075,
                1.04924226,
                1.04924226,
                0.39630439,
                0.16481719,
                0.16481719,
                0.84146061,
                0.83146709,
                0.83146709,
                0.32748059,
                1.00918804,
                1.00918804,
                0.91034823,
                1.31278027,
                1.38222654,
                1.38222654,
                -0.32268814,
                -0.32268814,
                2.1866116,
                1.21679252,
                -0.15916878,
                -0.15916878,
                0.38958249,
                0.38958249,
                0.54971928,
                0.05591406,
                0.87712017,
                0.87712017,
                0.19409043,
                0.19409043,
                0.19409043,
                0.40718849,
                0.63399349,
                0.35510353,
                0.35510353,
                0.47860847,
                0.47860847,
                0.69805772,
                0.16686305,
                0.16686305,
                0.16686305,
                0.04971251,
                0.04971251,
                -0.90052793,
                -0.73203754,
                1.02258958,
                1.02258958,
                -0.14144856,
                -0.14144856,
                1.43017486,
                1.23202605,
                1.23202605,
                0.24442885,
                0.78300516,
                0.30494261,
                0.30494261,
                0.30494261,
                -0.00596443,
                1.31695235,
                0.81375848,
                0.81375848,
                0.81375848,
                1.91238675,
            ]
        ),
        Metropolis: np.array(
            [
                1.62434536,
                1.01258895,
                0.4844172,
                0.4844172,
                0.4844172,
                0.4844172,
                0.4844172,
                0.4844172,
                0.4844172,
                0.4844172,
                0.31198899,
                0.31198899,
                0.31198899,
                0.31198899,
                1.21284494,
                0.52911708,
                0.261229,
                0.79158447,
                0.10441177,
                -0.74079387,
                -0.74079387,
                -0.50637818,
                -0.50637818,
                -0.50637818,
                -0.45557042,
                -0.45557042,
                -0.33541147,
                0.28179164,
                0.58196196,
                0.22971211,
                0.02081788,
                0.60744107,
                0.8930284,
                0.8930284,
                1.40595822,
                1.10786538,
                1.10786538,
                1.10786538,
                1.10786538,
                -0.28863095,
                -0.12859388,
                0.74757504,
                0.74757504,
                0.74757504,
                0.97766977,
                0.97766977,
                0.75534163,
                0.55458356,
                0.75288328,
                0.87189193,
                0.9937132,
                0.9937132,
                0.61842825,
                0.61842825,
                0.27457457,
                0.31817143,
                0.31817143,
                0.31817143,
                -0.77674042,
                -0.60735798,
                0.13319847,
                -0.82050213,
                -0.82050213,
                -0.50534274,
                -0.15479676,
                -0.15479676,
                -0.19349227,
                -0.19349227,
                -0.21810923,
                -0.21810923,
                -0.21810923,
                1.0180548,
                -0.18121323,
                0.68213209,
                0.68213209,
                1.23266958,
                1.23266958,
                0.60913885,
                1.41099989,
                1.45756718,
                1.45756718,
                1.45756718,
                1.45756718,
                1.59526839,
                1.82776295,
                1.82776295,
                1.82776295,
                1.82776295,
                2.2691274,
                2.16897216,
                2.18638157,
                1.06436284,
                0.54726838,
                0.54726838,
                1.04247971,
                0.86777655,
                0.86777655,
                0.86777655,
                0.86777655,
                0.61914177,
            ]
        ),
        NUTS: np.array(
            [
                0.550575,
                0.550575,
                0.80031201,
                0.91580544,
                1.34622953,
                1.34622953,
                -0.63861533,
                -0.62101385,
                -0.62101385,
                -0.60250375,
                -1.04753424,
                -0.34850626,
                0.35882649,
                -0.20339408,
                -0.18077466,
                -0.18077466,
                0.1242007,
                -0.48708213,
                0.01216292,
                0.01216292,
                -0.15991487,
                0.0118306,
                0.0118306,
                0.02512962,
                -0.06002705,
                0.61278464,
                -0.45991609,
                -0.45991609,
                -0.45991609,
                -0.3067988,
                -0.3067988,
                -0.30830273,
                -0.62877494,
                -0.5896293,
                0.32740518,
                0.32740518,
                0.55321326,
                0.34885231,
                0.34885231,
                0.35304997,
                1.20016133,
                1.20016133,
                1.26432486,
                1.22481613,
                1.46040499,
                1.2251786,
                0.29954482,
                0.29954482,
                0.5713582,
                0.5755183,
                0.26968846,
                0.68253483,
                0.68253483,
                0.69418724,
                1.4172782,
                1.4172782,
                0.85063608,
                0.23409974,
                -0.65012501,
                1.16211157,
                -0.04844954,
                1.34390994,
                -0.44058335,
                -0.44058335,
                0.85096033,
                0.98734074,
                1.31200906,
                1.2751574,
                1.2751574,
                0.04377635,
                0.08244824,
                0.6342471,
                -0.31243596,
                1.0165907,
                -0.19025897,
                -0.19025897,
                0.02133041,
                -0.02335463,
                0.43923434,
                -0.45033488,
                0.05985518,
                -0.10019701,
                1.34229104,
                1.28571862,
                0.59557205,
                0.63730268,
                0.63730268,
                0.54269992,
                0.54269992,
                -0.48334519,
                1.02199273,
                -0.17367903,
                -0.17367903,
                0.8470911,
                -0.12868214,
                1.8986946,
                1.55412619,
                1.55412619,
                0.90228003,
                1.3328478
            ]
        ),
    }

    def setup_class(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_class(self):
        shutil.rmtree(self.temp_dir)

    @pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
    def test_sample_exact(self):
        for step_method in self.master_samples:
            self.check_trace(step_method)

    def check_trace(self, step_method):
        """Tests whether the trace for step methods is exactly the same as on master.

        Code changes that effect how random numbers are drawn may change this, and require
        `master_samples` to be updated, but such changes should be noted and justified in the
        commit.

        This method may also be used to benchmark step methods across commits, by running, for
        example

        ```
        BENCHMARK=100000 ./scripts/test.sh -s pymc3/tests/test_step.py:TestStepMethods
        ```

        on multiple commits.
        """
        n_steps = 100
        with Model() as model:
            x = Normal("x", mu=0, sigma=1)
            y = Normal("y", mu=x, sigma=1, observed=1)
            if step_method.__name__ == "NUTS":
                step = step_method(scaling=model.test_point)
                trace = sample(
                    0, tune=n_steps, discard_tuned_samples=False, step=step, random_seed=1, chains=1
                )
            else:
                trace = sample(
                    0,
                    tune=n_steps,
                    discard_tuned_samples=False,
                    step=step_method(),
                    random_seed=1,
                    chains=1,
                )

        assert_array_almost_equal(
            trace["x"],
            self.master_samples[step_method],
            decimal=select_by_precision(float64=6, float32=4),
        )

    def check_stat(self, check, trace, name):
        for (var, stat, value, bound) in check:
            s = stat(trace[var][2000:], axis=0)
            close_to(s, value, bound)

    def test_step_continuous(self):
        start, model, (mu, C) = mv_simple()
        unc = np.diag(C) ** 0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, unc, unc / 10.0))
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
            )
        for step in steps:
            trace = sample(
                0,
                tune=8000,
                chains=1,
                discard_tuned_samples=False,
                step=step,
                start=start,
                model=model,
                random_seed=1,
            )
            self.check_stat(check, trace, step.__class__.__name__)

    def test_step_discrete(self):
        if theano.config.floatX == "float32":
            return  # Cannot use @skip because it only skips one iteration of the yield
        start, model, (mu, C) = mv_simple_discrete()
        unc = np.diag(C) ** 0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, unc, unc / 10.0))
        with model:
            steps = (Metropolis(S=C, proposal_dist=MultivariateNormalProposal),)
        for step in steps:
            trace = sample(
                20000, tune=0, step=step, start=start, model=model, random_seed=1, chains=1
            )
            self.check_stat(check, trace, step.__class__.__name__)

    def test_step_categorical(self):
        start, model, (mu, C) = simple_categorical()
        unc = C ** 0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, unc, unc / 10.0))
        with model:
            steps = (
                CategoricalGibbsMetropolis(model.x, proposal="uniform"),
                CategoricalGibbsMetropolis(model.x, proposal="proportional"),
            )
        for step in steps:
            trace = sample(8000, tune=0, step=step, start=start, model=model, random_seed=1)
            self.check_stat(check, trace, step.__class__.__name__)

    def test_step_elliptical_slice(self):
        start, model, (K, L, mu, std, noise) = mv_prior_simple()
        unc = noise ** 0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, std, unc / 10.0))
        with model:
            steps = (EllipticalSlice(prior_cov=K), EllipticalSlice(prior_chol=L))
        for step in steps:
            trace = sample(
                5000, tune=0, step=step, start=start, model=model, random_seed=1, chains=1
            )
            self.check_stat(check, trace, step.__class__.__name__)


class TestMetropolisProposal:
    def test_proposal_choice(self):
        _, model, _ = mv_simple()
        with model:
            s = np.ones(model.ndim)
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
        theano.config.floatX == "float32", reason="Test fails on 32 bit due to linalg issues"
    )
    def test_non_blocked(self):
        """Test that samplers correctly create non-blocked compound steps."""
        _, model = simple_2model_continuous()
        with model:
            for sampler in self.samplers:
                assert isinstance(sampler(blocked=False), CompoundStep)

    @pytest.mark.skipif(
        theano.config.floatX == "float32", reason="Test fails on 32 bit due to linalg issues"
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

            # a custom Theano Op that does not have a grad:
            is_64 = theano.config.floatX == "float64"
            itypes = [tt.dscalar] if is_64 else [tt.fscalar]
            otypes = [tt.dscalar] if is_64 else [tt.fscalar]

            @theano.as_op(itypes, otypes)
            def kill_grad(x):
                return x

            data = np.random.normal(size=(100,))
            Normal("y", mu=kill_grad(x), sigma=1, observed=data.astype(theano.config.floatX))

            steps = assign_step_methods(model, [])
        assert isinstance(steps, Slice)


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
        pass

    def test_demcmc_warning_on_small_populations(self):
        """Test that a warning is raised when n_chains <= n_dims"""
        with Model() as model:
            Normal("n", mu=0, sigma=1, shape=(2,3))
            with pytest.warns(UserWarning) as record:
                sample(
                    draws=5, tune=5, chains=6, step=DEMetropolis(),
                    # make tests faster by not parallelizing; disable convergence warning
                    cores=1, compute_convergence_checks=False
                )
        pass

    def test_demcmc_tune_parameter(self):
        """Tests that validity of the tune setting is checked"""
        with Model() as model:
            Normal("n", mu=0, sigma=1, shape=(2,3))
            
            step = DEMetropolis()
            assert step.tune is None

            step = DEMetropolis(tune='scaling')
            assert step.tune == 'scaling'

            step = DEMetropolis(tune='lambda')
            assert step.tune == 'lambda'

            with pytest.raises(ValueError):
                DEMetropolis(tune='foo')
        pass

    def test_nonparallelized_chains_are_random(self):
        with Model() as model:
            x = Normal("x", 0, 1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                trace = sample(chains=4, cores=1, draws=20, tune=0, step=DEMetropolis())
                samples = np.array(trace.get_values("x", combine=False))[:, 5]

                assert len(set(samples)) == 4, "Parallelized {} " "chains are identical.".format(
                    stepper
                )
        pass

    def test_parallelized_chains_are_random(self):
        with Model() as model:
            x = Normal("x", 0, 1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                trace = sample(chains=4, cores=4, draws=20, tune=0, step=DEMetropolis())
                samples = np.array(trace.get_values("x", combine=False))[:, 5]

                assert len(set(samples)) == 4, "Parallelized {} " "chains are identical.".format(
                    stepper
                )
        pass


class TestMetropolis:
    def test_tuning_reset(self):
        """Re-use of the step method instance with cores=1 must not leak tuning information between chains."""
        with Model() as pmodel:
            D = 3
            Normal('n', 0, 2, shape=(D,))
            trace = sample(
                tune=600,
                draws=500,
                step=Metropolis(tune=True, scaling=0.1),
                cores=1,
                chains=3,
                discard_tuned_samples=False
            )
        for c in range(trace.nchains):
            # check that the tuned settings changed and were reset
            assert trace.get_sampler_stats('scaling', chains=c)[0] == 0.1
            assert trace.get_sampler_stats('scaling', chains=c)[-1] != 0.1
        pass


class TestDEMetropolisZ:
    def test_tuning_lambda_sequential(self):
        with Model() as pmodel:
            Normal('n', 0, 2, shape=(3,))
            trace = sample(
                tune=1000,
                draws=500,
                step=DEMetropolisZ(tune='lambda', lamb=0.92),
                cores=1,
                chains=3,
                discard_tuned_samples=False
            )
        for c in range(trace.nchains):
            # check that the tuned settings changed and were reset
            assert trace.get_sampler_stats('lambda', chains=c)[0] == 0.92
            assert trace.get_sampler_stats('lambda', chains=c)[-1] != 0.92
            assert set(trace.get_sampler_stats('tune', chains=c)) == {True, False}
        pass

    def test_tuning_epsilon_parallel(self):
        with Model() as pmodel:
            Normal('n', 0, 2, shape=(3,))
            trace = sample(
                tune=1000,
                draws=500,
                step=DEMetropolisZ(tune='scaling', scaling=0.002),
                cores=2,
                chains=2,
                discard_tuned_samples=False
            )
        for c in range(trace.nchains):
            # check that the tuned settings changed and were reset
            assert trace.get_sampler_stats('scaling', chains=c)[0] == 0.002
            assert trace.get_sampler_stats('scaling', chains=c)[-1] != 0.002
            assert set(trace.get_sampler_stats('tune', chains=c)) == {True, False}
        pass

    def test_tuning_none(self):
        with Model() as pmodel:
            Normal('n', 0, 2, shape=(3,))
            trace = sample(
                tune=1000,
                draws=500,
                step=DEMetropolisZ(tune=None),
                cores=1,
                chains=2,
                discard_tuned_samples=False
            )
        for c in range(trace.nchains):
            # check that all tunable parameters remained constant
            assert len(set(trace.get_sampler_stats('lambda', chains=c))) == 1
            assert len(set(trace.get_sampler_stats('scaling', chains=c))) == 1
            assert set(trace.get_sampler_stats('tune', chains=c)) == {True, False}
        pass

    def test_tuning_reset(self):
        """Re-use of the step method instance with cores=1 must not leak tuning information between chains."""
        with Model() as pmodel:
            D = 3
            Normal('n', 0, 2, shape=(D,))
            trace = sample(
                tune=1000,
                draws=500,
                step=DEMetropolisZ(tune='scaling', scaling=0.002),
                cores=1,
                chains=3,
                discard_tuned_samples=False
            )
        for c in range(trace.nchains):
            # check that the tuned settings changed and were reset
            assert trace.get_sampler_stats('scaling', chains=c)[0] == 0.002
            assert trace.get_sampler_stats('scaling', chains=c)[-1] != 0.002
            # check that the variance of the first 50 iterations is much lower than the last 100
            for d in range(D):
                var_start = np.var(trace.get_values('n', chains=c)[:50,d])
                var_end = np.var(trace.get_values('n', chains=c)[-100:,d])
                assert var_start < 0.1 * var_end
        pass

    def test_tune_drop_fraction(self):
        tune = 300
        tune_drop_fraction = 0.85
        draws = 200
        with Model() as pmodel:
            Normal('n', 0, 2, shape=(3,))
            step = DEMetropolisZ(tune_drop_fraction=tune_drop_fraction)
            trace = sample(
                tune=tune,
                draws=draws,
                step=step,
                cores=1,
                chains=1,
                discard_tuned_samples=False
            )
            assert len(trace) == tune + draws
            assert len(step._history) == (tune - tune * tune_drop_fraction) + draws
        pass

    @pytest.mark.parametrize('variable,has_grad,outcome', [('n', True, 1),('n', False, 1),('b', True, 0),('b', False, 0)])
    def test_competence(self, variable, has_grad, outcome):
        with Model() as pmodel:
            Normal('n', 0, 2, shape=(3,))
            Binomial('b', n=2, p=0.3)
        assert DEMetropolisZ.competence(pmodel[variable], has_grad=has_grad) == outcome
        pass

    @pytest.mark.parametrize('tune_setting', ['foo', True, False])
    def test_invalid_tune(self, tune_setting):
        with Model() as pmodel:
            Normal('n', 0, 2, shape=(3,))
            with pytest.raises(ValueError):
                DEMetropolisZ(tune=tune_setting)
        pass

    def test_custom_proposal_dist(self):
        with Model() as pmodel:
            D = 3
            Normal('n', 0, 2, shape=(D,))
            trace = sample(
                tune=100,
                draws=50,
                step=DEMetropolisZ(proposal_dist=NormalProposal),
                cores=1,
                chains=3,
                discard_tuned_samples=False
            )
        pass


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
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
            HalfNormal("a", sigma=1, testval=-1, transform=None)
            with pytest.raises(SamplingError) as error:
                sample(init=None, chains=1, random_seed=1)
            error.match("Bad initial")

    @pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
    def test_bad_init_parallel(self):
        with Model():
            HalfNormal("a", sigma=1, testval=-1, transform=None)
            with pytest.raises(ParallelSamplingError) as error:
                sample(init=None, cores=2, random_seed=1)
            error.match("Bad initial")

    def test_linalg(self, caplog):
        with Model():
            a = Normal("a", shape=2)
            a = tt.switch(a > 0, np.inf, a)
            b = tt.slinalg.solve(floatX(np.eye(2)), a)
            Normal("c", mu=b, shape=2)
            caplog.clear()
            trace = sample(20, init=None, tune=5, chains=2)
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
            x = Normal("x", mu=0, sigma=1)
            trace = sample(draws=10, tune=1, chains=1)

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
        }
        assert trace.stat_names == expected_stat_names
        for varname in trace.stat_names:
            assert trace.get_sampler_stats(varname).shape == (10,)

        # Assert model logp is computed correctly: computing post-sampling
        # and tracking while sampling should give same results.
        model_logp_ = np.array(
            [model.logp(trace.point(i, chain=c)) for c in trace.chains for i in range(len(trace))]
        )
        assert (trace.model_logp == model_logp_).all()
