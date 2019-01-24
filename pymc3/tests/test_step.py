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
    SMC,
    DEMetropolis,
)
from pymc3.theanof import floatX
from pymc3.distributions import (
    Binomial,
    Normal,
    Bernoulli,
    Categorical,
    Beta,
    HalfNormal,
)

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
                0.43733634,
                0.43733634,
                0.15955614,
                -0.44355329,
                0.21465731,
                0.30148244,
                0.45527282,
                0.45527282,
                0.41753005,
                -0.03480236,
                1.16599611,
                0.565306,
                0.565306,
                0.0077143,
                -0.18291321,
                -0.14577946,
                -0.00703353,
                -0.00703353,
                0.14345194,
                -0.12345058,
                0.76875516,
                0.76875516,
                0.84289506,
                0.24596225,
                0.95287087,
                1.3799335,
                1.1493899,
                1.1493899,
                2.0255982,
                -0.77850273,
                0.11604115,
                0.11604115,
                0.39296557,
                0.34826491,
                0.5951183,
                0.63097341,
                0.57938784,
                0.57938784,
                0.76570029,
                0.63516046,
                0.23667784,
                2.0151377,
                1.92064966,
                1.09125654,
                -0.43716787,
                0.61939595,
                0.30566853,
                0.30566853,
                0.3690641,
                0.3690641,
                0.3690641,
                1.26497542,
                0.90890334,
                0.01482818,
                0.01482818,
                -0.15542473,
                0.26475651,
                0.32687263,
                1.21902207,
                0.6708017,
                -0.18867695,
                -0.18867695,
                -0.07141329,
                -0.04631175,
                -0.16855462,
                -0.16855462,
                1.05455573,
                0.47371825,
                0.47371825,
                0.86307077,
                0.86307077,
                0.51484125,
                1.0022533,
                1.0022533,
                1.02370316,
                0.71331829,
                0.71331829,
                0.71331829,
                0.40758664,
                0.81307434,
                -0.46269741,
                -0.60284666,
                0.06710527,
                0.06710527,
                -0.35055053,
                0.36727629,
                0.36727629,
                0.69350367,
                0.11268647,
                0.37681301,
                1.10168386,
                0.49559472,
                0.49559472,
                0.06193658,
                -0.07947103,
                0.01969434,
                1.28470893,
                -0.13536813,
                -0.13536813,
                0.6575966,
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
                0.80046332,
                0.91590059,
                1.34621916,
                1.34621916,
                -0.63917773,
                -0.65770809,
                -0.65770809,
                -0.64512868,
                -1.05448153,
                -0.5225666,
                0.14335153,
                -0.0034499,
                -0.0034499,
                0.05309212,
                -0.53186371,
                0.29325825,
                0.43210854,
                0.56284837,
                0.56284837,
                0.38041767,
                0.47322034,
                0.49937368,
                0.49937368,
                0.44424258,
                0.44424258,
                -0.02790848,
                -0.40470145,
                -0.35725567,
                -0.43744228,
                0.41955432,
                0.31099421,
                0.31099421,
                0.65811717,
                0.66649398,
                0.38493786,
                0.54114658,
                0.54114658,
                0.68222408,
                0.66404942,
                1.44143108,
                1.15638799,
                -0.06775775,
                -0.06775775,
                0.30418561,
                0.23543403,
                0.57934404,
                -0.5435111,
                -0.47938915,
                -0.23816662,
                0.36793792,
                0.36793792,
                0.64980016,
                0.52150456,
                0.64643321,
                0.26130179,
                1.10569077,
                1.10569077,
                1.23662797,
                -0.36928735,
                -0.14303069,
                0.85298904,
                0.85298904,
                0.31422085,
                0.32113762,
                0.32113762,
                1.0692238,
                1.0692238,
                1.60127576,
                1.49249738,
                1.09065107,
                0.84264371,
                0.84264371,
                -0.08832343,
                0.04868027,
                -0.02679449,
                -0.02679449,
                0.91989101,
                0.65754478,
                -0.39220625,
                0.08379492,
                1.03055634,
                1.03055634,
                1.71071332,
                1.58740483,
                1.67905741,
                0.77744868,
                0.15050587,
                0.15050587,
                0.73979127,
                0.15445515,
                0.13134717,
                0.85068974,
                0.85068974,
                0.6974799,
                0.16170472,
                0.86405959,
                0.86405959,
                -0.22032854,
            ]
        ),
        SMC: np.array(
            [
                0.96155575,
                0.12354058,
                1.37843394,
                -0.64490769,
                0.56153344,
                0.56949613,
                1.48805567,
                0.86943208,
                1.28672225,
                1.11818376,
                0.99202034,
                0.76951474,
                -0.80478442,
                0.25773324,
                1.71676489,
                0.47726659,
                1.41458346,
                0.91086253,
                0.36146025,
                -0.05769771,
                0.59441151,
                1.86157896,
                0.84378345,
                -0.08420864,
                0.96264496,
                1.18466616,
                -0.08301892,
                0.69627504,
                0.52721449,
                0.69209438,
                1.22462443,
                1.3527388,
                -0.96055421,
                0.95149676,
                0.64798478,
                0.74435388,
                0.67474009,
                0.16073648,
                0.6660611,
                0.62209787,
                0.72155144,
                0.93515369,
                0.62462479,
                0.93689047,
                0.66827631,
                1.22635052,
                1.40935767,
                -0.12069125,
                0.29298952,
                0.32982792,
                0.34051724,
                -0.24309767,
                -0.29414865,
                0.37741563,
                -0.29052578,
                -0.36946086,
                0.70680747,
                0.282181,
                0.91349846,
                1.61566788,
                1.35593426,
                0.22115063,
                -0.1904221,
                -0.53213195,
                0.25832089,
                1.272686,
                -0.34608613,
                0.7405656,
                0.86422801,
                0.4133873,
                0.02002896,
                1.52716732,
                1.11700266,
                0.01264456,
                1.28187506,
                -0.14362652,
                0.02715474,
                2.23628253,
                1.04628416,
                1.02788034,
                -0.16128468,
                0.5390423,
                1.34396764,
                1.53622121,
                0.9573948,
                2.14416466,
                0.71949634,
                0.78759446,
                1.39579606,
                -0.03169538,
                1.47822573,
                0.01943775,
                1.31197471,
                0.47475589,
                1.35694485,
                1.10879123,
                1.68316079,
                -0.3677566,
                0.01876864,
                1.10056079,
                -0.73678253,
                0.41594072,
                1.11037177,
                0.89196759,
                0.06043841,
                0.11064186,
                1.03143181,
                0.64732913,
                0.12394593,
                -0.26254331,
                0.36509784,
                0.7893701,
                -0.40537559,
                -0.78701105,
                1.54832725,
                0.41225113,
                -0.72722386,
                1.51618746,
                -0.34325617,
                -0.37255259,
                0.3190562,
                0.30245186,
                0.61122402,
                0.416849,
                0.56464343,
                1.19103522,
                0.74743207,
                0.8641633,
                1.71882087,
                1.07100044,
                1.33798974,
                1.48935406,
                1.16031688,
                0.37036675,
                -0.55450346,
                -0.70492211,
                -0.03610812,
                0.19853608,
                0.08956581,
                -0.2381033,
                0.83095776,
                0.76563511,
                -0.07106511,
                0.56329078,
                1.15931015,
                0.04533864,
                -0.21308172,
                0.274915,
                1.57040134,
                0.78145705,
                1.92064424,
                0.28787777,
                1.46101294,
                1.08120648,
                0.18144586,
                1.34311923,
                1.46516965,
                1.51600453,
                -0.09295297,
                -0.85488172,
                1.40737347,
                0.73784988,
                0.06846084,
                -0.1996966,
                0.3227957,
                1.20572832,
                -0.31104901,
                -0.01546281,
                0.0172552,
                0.22116851,
                0.0931431,
                -0.22007601,
                0.03757681,
                1.5345242,
                0.88473831,
                1.22550042,
                0.10627061,
                0.23637588,
                -0.11944451,
                -0.21066274,
                1.33649839,
                -1.0085795,
                0.48370976,
                1.38720187,
                -0.20363393,
                -0.01723264,
                0.33616712,
                0.57274987,
                -0.70025566,
                -0.05105602,
                -0.37324572,
                2.01686044,
                0.15311289,
                -0.32750163,
                0.56208031,
                0.90689661,
                0.43908833,
                0.35270695,
                0.37425973,
                -0.80283989,
            ]
        ),
    }

    def setup_class(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_class(self):
        shutil.rmtree(self.temp_dir)

    @pytest.mark.xfail(
        condition=(theano.config.floatX == "float32"), reason="Fails on float32"
    )
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
            if step_method.__name__ == "SMC":
                trace = sample(
                    draws=200, random_seed=1, progressbar=False, step=step_method()
                )
            elif step_method.__name__ == "NUTS":
                step = step_method(scaling=model.test_point)
                trace = sample(
                    0,
                    tune=n_steps,
                    discard_tuned_samples=False,
                    step=step,
                    random_seed=1,
                    chains=1,
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
                20000,
                tune=0,
                step=step,
                start=start,
                model=model,
                random_seed=1,
                chains=1,
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
            trace = sample(
                8000, tune=0, step=step, start=start, model=model, random_seed=1
            )
            self.check_stat(check, trace, step.__class__.__name__)

    def test_step_elliptical_slice(self):
        start, model, (K, L, mu, std, noise) = mv_prior_simple()
        unc = noise ** 0.5
        check = (("x", np.mean, mu, unc / 10.0), ("x", np.std, std, unc / 10.0))
        with model:
            steps = (EllipticalSlice(prior_cov=K), EllipticalSlice(prior_chol=L))
        for step in steps:
            trace = sample(
                5000,
                tune=0,
                step=step,
                start=start,
                model=model,
                random_seed=1,
                chains=1,
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
        theano.config.floatX == "float32",
        reason="Test fails on 32 bit due to linalg issues",
    )
    def test_non_blocked(self):
        """Test that samplers correctly create non-blocked compound steps."""
        _, model = simple_2model_continuous()
        with model:
            for sampler in self.samplers:
                assert isinstance(sampler(blocked=False), CompoundStep)

    @pytest.mark.skipif(
        theano.config.floatX == "float32",
        reason="Test fails on 32 bit due to linalg issues",
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
            Normal(
                "y", mu=kill_grad(x), sigma=1, observed=data.astype(theano.config.floatX)
            )

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
                    trace = sample(draws=100, chains=1, step=step)
                trace = sample(draws=100, chains=4, step=step)
        pass

    def test_parallelized_chains_are_random(self):
        with Model() as model:
            x = Normal("x", 0, 1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                trace = sample(
                    chains=4, draws=20, tune=0, step=DEMetropolis()
                )
                samples = np.array(trace.get_values("x", combine=False))[:, 5]

                assert (
                    len(set(samples)) == 4
                ), "Parallelized {} " "chains are identical.".format(stepper)
        pass


@pytest.mark.xfail(
    condition=(theano.config.floatX == "float32"), reason="Fails on float32"
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
            HalfNormal("a", sigma=1, testval=-1, transform=None)
            with pytest.raises(SamplingError) as error:
                sample(init=None, chains=1, random_seed=1)
            error.match("Bad initial")

    @pytest.mark.skipif(sys.version_info < (3,6),
                    reason="requires python3.6 or higher")
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
            [
                model.logp(trace.point(i, chain=c))
                for c in trace.chains
                for i in range(len(trace))
            ]
        )
        assert (trace.model_logp == model_logp_).all()
