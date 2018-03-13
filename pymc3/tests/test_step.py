import shutil
import tempfile

from .checks import close_to
from .models import (simple_categorical, mv_simple, mv_simple_discrete,
                     mv_prior_simple, simple_2model_continuous)
from pymc3.sampling import assign_step_methods, sample
from pymc3.model import Model
from pymc3.step_methods import (NUTS, BinaryGibbsMetropolis, CategoricalGibbsMetropolis,
                                Metropolis, Slice, CompoundStep, NormalProposal,
                                MultivariateNormalProposal, HamiltonianMC,
                                EllipticalSlice, smc, DEMetropolis)
from pymc3.theanof import floatX
from pymc3.distributions import (
    Binomial, Normal, Bernoulli, Categorical, Beta, HalfNormal)

from numpy.testing import assert_array_almost_equal
import numpy as np
import numpy.testing as npt
import pytest
import theano
import theano.tensor as tt
from .helpers import select_by_precision


class TestStepMethods(object):  # yield test doesn't work subclassing object
    master_samples = {
        Slice: np.array([
            -5.95252353e-01, -1.81894861e-01, -4.98211488e-01,
            -1.02262800e-01, -4.26726030e-01, 1.75446860e+00,
            -1.30022548e+00, 8.35658004e-01, 8.95879638e-01,
            -8.85214481e-01, -6.63530918e-01, -8.39303080e-01,
            9.42792225e-01, 9.03554344e-01, 8.45254684e-01,
            -1.43299803e+00, 9.04897201e-01, -1.74303131e-01,
            -6.38611581e-01, 1.50013968e+00, 1.06864438e+00,
            -4.80484421e-01, -7.52199709e-01, 1.95067495e+00,
            -3.67960104e+00, 2.49291588e+00, -2.11039152e+00,
            1.61674758e-01, -1.59564182e-01, 2.19089873e-01,
            1.88643940e+00, 4.04098154e-01, -4.59352326e-01,
            -9.06370675e-01, 5.42817654e-01, 6.99040611e-03,
            1.66396391e-01, -4.74549281e-01, 8.19064437e-02,
            1.69689952e+00, -1.62667304e+00, 1.61295808e+00,
            1.30099144e+00, -5.46722750e-01, -7.87745494e-01,
            7.91027521e-01, -2.35706976e-02, 1.68824376e+00,
            7.10566880e-01, -7.23551374e-01, 8.85613069e-01,
            -1.27300146e+00, 1.80274430e+00, 9.34266276e-01,
            2.40427061e+00, -1.85132552e-01, 4.47234196e-01,
            -9.81894859e-01, -2.83399706e-01, 1.84717533e+00,
            -1.58593284e+00, 3.18027270e-02, 1.40566006e+00,
            -9.45758714e-01, 1.18813188e-01, -1.19938604e+00,
            -8.26038466e-01, 5.03469984e-01, -4.72742758e-01,
            2.27820946e-01, -1.02608915e-03, -6.02507158e-01,
            7.72739682e-01, 7.16064505e-01, -1.63693490e+00,
            -3.97161966e-01, 1.17147944e+00, -2.87796982e+00,
            -1.59533297e+00, 6.73096114e-01, -3.34397247e-01,
            1.22357427e-01, -4.57299104e-02, 1.32005771e+00,
            -1.29910645e+00, 8.16168850e-01, -1.47357594e+00,
            1.34688446e+00, 1.06377551e+00, 4.34296696e-02,
            8.23143354e-01, 8.40906324e-01, 1.88596864e+00,
            5.77120694e-01, 2.71732927e-01, -1.36217979e+00,
            2.41488213e+00, 4.68298379e-01, 4.86342250e-01,
            -8.43949966e-01]),
        HamiltonianMC: np.array([
             0.40608634,  0.40608634,  0.04610354, -0.78588609,  0.03773683,
            -0.49373368,  0.21708042,  0.21708042, -0.14413517, -0.68284611,
             0.76299659,  0.24128663,  0.24128663, -0.54835464, -0.84408365,
            -0.82762589, -0.67429432, -0.67429432, -0.57900517, -0.97138029,
            -0.37809745, -0.37809745, -0.19333181, -0.40329098,  0.54999765,
             1.171515  ,  0.90279792,  0.90279792,  1.63830503, -0.90436674,
            -0.02516293, -0.02516293, -0.22177082, -0.28061216, -0.10158021,
             0.0807234 ,  0.16994063,  0.16994063,  0.4141503 ,  0.38505666,
            -0.25936504,  2.12074192,  2.24467132,  0.9628703 , -1.37044749,
             0.32983336, -0.55317525, -0.55317525, -0.40295662, -0.40295662,
            -0.40295662,  0.49076931,  0.04234407, -1.0002905 ,  0.99823615,
             0.99823615,  0.24915904, -0.00965061,  0.48519377,  0.21959942,
            -0.93094702, -0.93094702, -0.76812553, -0.73699981, -0.91834134,
            -0.91834134,  0.79522886, -0.04267669, -0.04267669,  0.51368761,
             0.51368761,  0.02255577,  0.70823409,  0.70823409,  0.73921198,
             0.30295007,  0.30295007,  0.30295007, -0.1300897 ,  0.44310964,
            -1.35839961, -1.55398633, -0.57323153, -0.57323153, -1.15435458,
            -0.17697793, -0.17697793,  0.2925856 , -0.56119025, -0.15360141,
             0.83715916, -0.02340449, -0.02340449, -0.63074456, -0.82745942,
            -0.67626237,  1.13814805, -0.81857813, -0.81857813,  0.26367166]),
        Metropolis: np.array([
            1.62434536, 1.01258895, 0.4844172, -0.58855142, 1.15626034, 0.39505344, 1.85716138,
            -0.20297933, -0.20297933, -0.20297933, -0.20297933, -1.08083775, -1.08083775,
            0.06388596, 0.96474191, 0.28101405, 0.01312597, 0.54348144, -0.14369126, -0.98889691,
            -0.98889691, -0.75448121, -0.94631676, -0.94631676, -0.89550901, -0.89550901,
            -0.77535005, -0.15814694, 0.14202338, -0.21022647, -0.4191207, 0.16750249, 0.45308981,
            1.33823098, 1.8511608, 1.55306796, 1.55306796, 1.55306796, 1.55306796, 0.15657163,
            0.3166087, 0.3166087, 0.3166087, 0.3166087, 0.54670343, 0.54670343, 0.32437529,
            0.12361722, 0.32191694, 0.44092559, 0.56274686, 0.56274686, 0.18746191, 0.18746191,
            -0.15639177, -0.11279491, -0.11279491, -0.11279491, -1.20770676, -1.03832432,
            -0.29776787, -1.25146848, -1.25146848, -0.93630908, -0.5857631, -0.5857631,
            -0.62445861, -0.62445861, -0.64907557, -0.64907557, -0.64907557, 0.58708846,
            -0.61217957, 0.25116575, 0.25116575, 0.80170324, 1.59451011, 0.97097938, 1.77284041,
            1.81940771, 1.81940771, 1.81940771, 1.81940771, 1.95710892, 2.18960348, 2.18960348,
            2.18960348, 2.18960348, 2.63096792, 2.53081269, 2.5482221, 1.42620337, 0.90910891,
            -0.08791792, 0.40729341, 0.23259025, 0.23259025, 0.23259025, 2.76091595, 2.51228118]),
        NUTS: np.array(
            [  1.11832371e+00,   1.11832371e+00,   1.11203151e+00,  -1.08526075e+00,
               2.58200798e-02,   2.03527183e+00,   4.47644923e-01,   8.95141642e-01,
               7.21867642e-01,   8.61681133e-01,   8.61681133e-01,   3.42001064e-01,
              -1.08109692e-01,   1.89399407e-01,   2.76571728e-01,   2.76571728e-01,
              -7.49542468e-01,  -7.25272156e-01,  -5.49940424e-01,  -5.49940424e-01,
               4.39045553e-01,  -9.79313191e-04,   4.08678631e-02,   4.08678631e-02,
              -1.17303762e+00,   4.15335470e-01,   4.80458006e-01,   5.98022153e-02,
               5.26508851e-01,   5.26508851e-01,   6.24759070e-01,   4.55268819e-01,
               8.70608570e-01,   6.56151353e-01,   6.56151353e-01,   1.29968043e+00,
               2.41336915e-01,  -7.78824784e-02,  -1.15368193e+00,  -4.92562283e-01,
              -5.16903724e-02,   4.05389240e-01,   4.05389240e-01,   4.20147769e-01,
               6.88161155e-01,   6.59273169e-01,  -4.28987827e-01,  -4.28987827e-01,
              -4.44203783e-01,  -4.61330842e-01,  -5.23216216e-01,  -1.52821368e+00,
               9.84049809e-01,   9.84049809e-01,   1.02081403e+00,  -5.60272679e-01,
               4.18620552e-01,   1.92542517e+00,   1.12029984e+00,   6.69152820e-01,
               1.56325611e+00,   6.64640934e-01,  -7.43157898e-01,  -7.43157898e-01,
              -3.18049839e-01,   6.87248073e-01,   6.90665184e-01,   1.63009949e+00,
              -4.84972607e-01,  -1.04859669e+00,   8.26455763e-01,  -1.71696305e+00,
              -1.39964174e+00,  -3.87677130e-01,  -1.85966115e-01,  -1.85966115e-01,
               4.54153291e-01,  -8.41705332e-01,  -8.46831314e-01,  -8.46831314e-01,
              -1.57419678e-01,  -3.89604101e-01,   8.15315055e-01,   2.81141081e-03,
               2.81141081e-03,   3.25839131e-01,   1.33638823e+00,   1.59391112e+00,
              -3.91174647e-01,  -2.60664979e+00,  -2.27637534e+00,  -2.81505065e+00,
              -2.24238542e+00,  -1.01648100e+00,  -1.01648100e+00,  -7.60912865e-01,
               1.44384812e+00,   2.07355127e+00,   1.91390340e+00,   1.66559696e+00]),
        smc.SMC: np.array(
      [ 1.30059573, -2.09236482,  1.30059573, -2.09236482,  1.90856591,
       -2.17364931,  1.5960639 , -0.36425894,  1.5960639 , -0.36425894,
        1.14331113, -0.75276125,  1.14331113, -0.75276125,  1.14331113,
       -0.42847475,  0.81864786,  0.28096672,  0.81864786,  0.0651507 ,
        0.81864786, -0.70540882,  0.61616796,  0.87401042,  0.78354273,
       -0.37837636,  1.26663724, -0.03708171,  1.47079839,  0.0514686 ,
        1.23826676,  0.0514686 ,  1.23826676, -0.33102265,  1.27408454,
       -0.33102265,  1.27408454, -0.33102265,  1.78171814, -0.51873434,
        2.15479286, -0.51873434,  1.80940573,  0.50305662,  1.61615171,
        1.18179431,  1.49454445,  1.18179431,  1.24531615, -0.29887706,
        0.81992635,  0.0583591 ,  0.83090574,  0.6001567 ,  0.83090574,
        0.6001567 ,  0.83090574,  0.13329453,  0.631766  ,  0.85347689,
       -0.33685913,  0.2653449 ,  0.55562255, -0.153675  ,  0.5899056 ,
       -0.30368421,  1.3176159 , -0.30368421, -0.10773807, -0.30368421,
       -0.95430079, -0.30368421, -0.79696017, -0.30368421, -0.82295762,
       -0.48531359, -0.55594997, -1.08820188,  1.38036911, -0.38097196,
       -0.52048371,  0.01768498, -1.1281052 , -0.48004372, -0.07315004,
       -0.10131235, -0.07315004, -0.02946858, -0.07315004, -0.02946858,
       -0.07315004, -0.02946858, -0.04628493, -0.02946858, -0.04628493,
        0.30448559, -0.04628493, -0.39443274, -1.01569588,  0.02931213,
       -1.26554183,  0.43723809, -0.61995994,  0.43723809, -0.77759246,
        0.43723809, -0.18242551,  0.36174876, -0.18676902,  1.01363215,
       -0.19462025,  1.01357247,  0.02428945,  0.70940217, -0.23306625,
        0.70940217, -0.49628497,  0.70940217, -0.54714578,  0.58162522,
       -0.54714578,  1.31100328, -0.16713737,  0.41027471,  0.61429875,
        0.41027471,  0.36861437, -0.0036154 ,  0.36861437,  0.32733057,
        0.36861437,  0.32141885,  0.36861437, -0.67466622,  0.7632303 ,
       -0.6185374 ,  0.7632303 ,  0.3266455 ,  0.7632303 , -0.49167547,
        0.49614413, -0.18837686,  0.46732735, -0.18837686,  0.57459841,
        1.18609558,  0.43013098,  0.99155781,  0.36500138,  0.89705333,
        0.69337612,  0.89705333,  1.21032969,  0.89705333,  1.21032969,
       -0.25758511,  1.21032969, -0.25758511,  1.39103118, -0.25758511,
        1.39103118, -0.43364893,  1.45601653, -0.90496012,  1.45601653,
       -0.72002279,  1.45601653, -0.93537717,  1.45601653, -0.32342724,
        1.45601653, -0.15438672,  1.45601653, -0.14222719,  2.41009895,
        0.28052695,  1.79126693, -0.07194438,  1.79126693, -0.20683481,
        1.55675859, -0.20683481,  1.55675859, -0.25009294,  1.78918238,
       -0.25009294,  1.71317267,  1.07014506,  1.8411848 ,  1.07014506,
        1.6260982 ,  1.07014506,  1.87360781,  1.68751729,  1.81975144,
        1.68751729,  1.87260248,  1.68751729,  1.87260248,  0.76528088]),
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
            x = Normal('x', mu=0, sd=1)
            if step_method.__name__ == 'SMC':
                trace = smc.sample_smc(samples=200,
                                       chains=2,
                                       start=[{'x':1.}, {'x':-1.}],
                                       random_seed=1,
                                       cores=1, progressbar=False,
                                       homepath=self.temp_dir)
            elif step_method.__name__ == 'NUTS':
                step = step_method(scaling=model.test_point)
                trace = sample(0, tune=n_steps,
                               discard_tuned_samples=False,
                               step=step, random_seed=1, chains=1)
            else:
                trace = sample(0, tune=n_steps,
                               discard_tuned_samples=False,
                               step=step_method(), random_seed=1, chains=1)
        assert_array_almost_equal(
            trace.get_values('x'),
            self.master_samples[step_method],
            decimal=select_by_precision(float64=6, float32=4))

    def check_stat(self, check, trace, name):
        for (var, stat, value, bound) in check:
            s = stat(trace[var][2000:], axis=0)
            close_to(s, value, bound)

    def test_step_continuous(self):
        start, model, (mu, C) = mv_simple()
        unc = np.diag(C) ** .5
        check = (('x', np.mean, mu, unc / 10.),
                 ('x', np.std, unc, unc / 10.))
        with model:
            steps = (
                Slice(),
                HamiltonianMC(scaling=C, is_cov=True, blocked=False),
                NUTS(scaling=C, is_cov=True, blocked=False),
                Metropolis(S=C, proposal_dist=MultivariateNormalProposal, blocked=True),
                Slice(blocked=True),
                HamiltonianMC(scaling=C, is_cov=True),
                NUTS(scaling=C, is_cov=True),
                CompoundStep([
                    HamiltonianMC(scaling=C, is_cov=True),
                    HamiltonianMC(scaling=C, is_cov=True, blocked=False)]),
            )
        for step in steps:
            trace = sample(0, tune=8000, chains=1,
                           discard_tuned_samples=False, step=step,
                           start=start, model=model, random_seed=1)
            self.check_stat(check, trace, step.__class__.__name__)

    def test_step_discrete(self):
        if theano.config.floatX == "float32":
            return  # Cannot use @skip because it only skips one iteration of the yield
        start, model, (mu, C) = mv_simple_discrete()
        unc = np.diag(C) ** .5
        check = (('x', np.mean, mu, unc / 10.),
                 ('x', np.std, unc, unc / 10.))
        with model:
            steps = (
                Metropolis(S=C, proposal_dist=MultivariateNormalProposal),
            )
        for step in steps:
            trace = sample(20000, tune=0, step=step, start=start, model=model,
                           random_seed=1, chains=1)
            self.check_stat(check, trace, step.__class__.__name__)

    def test_step_categorical(self):
        start, model, (mu, C) = simple_categorical()
        unc = C ** .5
        check = (('x', np.mean, mu, unc / 10.),
                 ('x', np.std, unc, unc / 10.))
        with model:
            steps = (
                CategoricalGibbsMetropolis(model.x, proposal='uniform'),
                CategoricalGibbsMetropolis(model.x, proposal='proportional'),
            )
        for step in steps:
            trace = sample(8000, tune=0, step=step, start=start, model=model, random_seed=1)
            self.check_stat(check, trace, step.__class__.__name__)

    def test_step_elliptical_slice(self):
        start, model, (K, L, mu, std, noise) = mv_prior_simple()
        unc = noise ** 0.5
        check = (('x', np.mean, mu, unc / 10.),
                 ('x', np.std, std, unc / 10.))
        with model:
            steps = (
                EllipticalSlice(prior_cov=K),
                EllipticalSlice(prior_chol=L),
            )
        for step in steps:
            trace = sample(5000, tune=0, step=step, start=start, model=model,
                           random_seed=1, chains=1)
            self.check_stat(check, trace, step.__class__.__name__)


class TestMetropolisProposal(object):
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


class TestCompoundStep(object):
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS, DEMetropolis)

    @pytest.mark.skipif(theano.config.floatX == "float32",
                        reason="Test fails on 32 bit due to linalg issues")
    def test_non_blocked(self):
        """Test that samplers correctly create non-blocked compound steps."""
        _, model = simple_2model_continuous()
        with model:
            for sampler in self.samplers:
                assert isinstance(sampler(blocked=False), CompoundStep)

    @pytest.mark.skipif(theano.config.floatX == "float32",
                        reason="Test fails on 32 bit due to linalg issues")
    def test_blocked(self):
        _, model = simple_2model_continuous()
        with model:
            for sampler in self.samplers:
                sampler_instance = sampler(blocked=True)
                assert not isinstance(sampler_instance, CompoundStep)
                assert isinstance(sampler_instance, sampler)


class TestAssignStepMethods(object):
    def test_bernoulli(self):
        """Test bernoulli distribution is assigned binary gibbs metropolis method"""
        with Model() as model:
            Bernoulli('x', 0.5)
            steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)

    def test_normal(self):
        """Test normal distribution is assigned NUTS method"""
        with Model() as model:
            Normal('x', 0, 1)
            steps = assign_step_methods(model, [])
        assert isinstance(steps, NUTS)

    def test_categorical(self):
        """Test categorical distribution is assigned categorical gibbs metropolis method"""
        with Model() as model:
            Categorical('x', np.array([0.25, 0.75]))
            steps = assign_step_methods(model, [])
        assert isinstance(steps, BinaryGibbsMetropolis)
        with Model() as model:
            Categorical('y', np.array([0.25, 0.70, 0.05]))
            steps = assign_step_methods(model, [])
        assert isinstance(steps, CategoricalGibbsMetropolis)

    def test_binomial(self):
        """Test binomial distribution is assigned metropolis method."""
        with Model() as model:
            Binomial('x', 10, 0.5)
            steps = assign_step_methods(model, [])
        assert isinstance(steps, Metropolis)

    def test_normal_nograd_op(self):
        """Test normal distribution without an implemented gradient is assigned slice method"""
        with Model() as model:
            x = Normal('x', 0, 1)

            # a custom Theano Op that does not have a grad:
            is_64 = theano.config.floatX == "float64"
            itypes = [tt.dscalar] if is_64 else [tt.fscalar]
            otypes = [tt.dscalar] if is_64 else [tt.fscalar]
            @theano.as_op(itypes, otypes)
            def kill_grad(x):
                return x

            data = np.random.normal(size=(100,))
            Normal("y", mu=kill_grad(x), sd=1, observed=data.astype(theano.config.floatX))

            steps = assign_step_methods(model, [])
        assert isinstance(steps, Slice)


class TestPopulationSamplers(object):

    steppers = [DEMetropolis]

    def test_checks_population_size(self):
        """Test that population samplers check the population size."""
        with Model() as model:
            n = Normal('n', mu=0, sd=1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()
                with pytest.raises(ValueError):
                    trace = sample(draws=100, chains=1, step=step)
                trace = sample(draws=100, chains=4, step=step)
        pass

    def test_parallelized_chains_are_random(self):
        with Model() as model:
            x = Normal('x', 0, 1)
            for stepper in TestPopulationSamplers.steppers:
                step = stepper()

                trace = sample(chains=4, draws=20, tune=0, step=DEMetropolis(),
                               parallelize=True)
                samples = np.array(trace.get_values('x', combine=False))[:,5]

                assert len(set(samples)) == 4, 'Parallelized {} ' \
                    'chains are identical.'.format(stepper)
        pass


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestNutsCheckTrace(object):
    def test_multiple_samplers(self, caplog):
        with Model():
            prob = Beta('prob', alpha=5., beta=3.)
            Binomial('outcome', n=1, p=prob)
            caplog.clear()
            sample(3, tune=2, discard_tuned_samples=False,
                   n_init=None, chains=1)
            messages = [msg.msg for msg in caplog.records]
            assert all('boolean index did not' not in msg for msg in messages)

    def test_bad_init(self):
        with Model():
            HalfNormal('a', sd=1, testval=-1, transform=None)
            with pytest.raises(ValueError) as error:
                sample(init=None)
            error.match('Bad initial')

    def test_linalg(self, caplog):
        with Model():
            a = Normal('a', shape=2)
            a = tt.switch(a > 0, np.inf, a)
            b = tt.slinalg.solve(floatX(np.eye(2)), a)
            Normal('c', mu=b, shape=2)
            caplog.clear()
            trace = sample(20, init=None, tune=5, chains=2)
            warns = [msg.msg for msg in caplog.records]
            assert np.any(trace['diverging'])
            assert (
                any('divergences after tuning' in warn
                    for warn in warns)
                or
                any('only diverging samples' in warn
                    for warn in warns))

            with pytest.raises(ValueError) as error:
                trace.report.raise_ok()
            error.match('issues during sampling')

            assert not trace.report.ok
