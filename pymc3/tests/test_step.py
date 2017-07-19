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
                                EllipticalSlice, smc)
from pymc3.theanof import floatX
from pymc3 import SamplingError
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
            -8.13087389e-01, -3.08921856e-01, -6.79377098e-01, 6.50812585e-01, -7.63577596e-01,
            -8.13199793e-01, -1.63823548e+00, -7.03863676e-02, 2.05107771e+00, 1.68598170e+00,
            6.92463695e-01, -7.75120766e-01, -1.62296463e+00, 3.59722423e-01, -2.31421712e-01,
            -7.80686956e-02, -6.05860731e-01, -1.13000202e-01, 1.55675942e-01, -6.78527612e-01,
            6.31052333e-01, 6.09012517e-01, -1.56621643e+00, 5.04330883e-01, 3.14824082e-03,
            -1.31287073e+00, 4.10706927e-01, 8.93815792e-01, 8.19317020e-01, 3.71900919e-01,
            -2.62067312e+00, -3.47616592e+00, 1.50335041e+00, -1.05993351e+00, 2.41571723e-01,
            -1.06258156e+00, 5.87999429e-01, -1.78480091e-01, -3.60278680e-01, 1.90615274e-01,
            -1.24399204e-01, 4.03845589e-01, -1.47797573e-01, 7.90445804e-01, -1.21043819e+00,
            -1.33964776e+00, 1.36366329e+00, -7.50175388e-01, 9.25241839e-01, -4.17493767e-01,
            1.85311339e+00, -2.49715343e+00, -3.18571692e-01, -1.49099668e+00, -2.62079621e-01,
            -5.82376852e-01, -2.53033395e+00, 2.07580503e+00, -9.82615856e-01, 6.00517782e-01,
            -9.83941620e-01, -1.59014118e+00, -1.83931394e-03, -4.71163466e-01, 1.90073737e+00,
            -2.08929125e-01, -6.98388847e-01, 1.64502092e+00, -1.19525944e+00, 1.44424109e+00,
            1.52974876e+00, -5.70140077e-01, 5.08633322e-01, -1.70862492e-02, -1.69887948e-01,
            5.19760297e-01, -4.15149647e-01, 8.63685174e-02, -3.66805233e-01, -9.24988952e-01,
            2.33307122e+00, -2.60391496e-01, -5.86271814e-01, -5.01297170e-01, -1.53866195e+00,
            5.71285373e-01, -1.30571830e+00, 8.59587795e-01, 6.72170694e-01, 9.12433943e-01,
            7.04959179e-01, 8.37863464e-01, -5.24200836e-01, 1.28261340e+00, 9.08774240e-01,
            8.80566763e-01, 7.82911967e-01, 8.01843432e-01, 7.09251098e-01, 5.73803618e-01]),
        HamiltonianMC: np.array([
           -0.74925631, -0.2566773 , -2.12480977,  1.64328926, -1.39315913,
            2.04200003,  0.00706711,  0.34240498,  0.44276674, -0.21368043,
           -0.76398723,  1.19280082, -1.43030242, -0.44896107,  0.0547087 ,
           -1.72170938, -0.20443956,  0.35432546,  1.77695096, -0.31053636,
           -0.26729283,  1.26450201,  0.17049917,  0.27953939, -0.24185153,
            0.95617117, -0.45707061,  0.75837366, -1.73391277,  1.63331612,
           -0.68426038,  0.20499991, -0.43866983,  0.31080195,  0.47104548,
           -0.50331753,  0.7821196 , -1.7544931 ,  1.24106497, -1.0152971 ,
           -0.01949091, -0.33151479,  0.19138253,  0.40349184,  0.31694823,
           -0.01508142, -0.31330951,  0.40874228,  0.40874228,  0.58078882,
            0.68378375,  0.84142914,  0.44756075, -0.87297183,  0.59695222,
            1.96161733, -0.37126652,  0.27552912,  0.74547583, -0.16172925,
            0.79969568, -0.20501522, -0.36181518,  0.13114261, -0.8461323 ,
           -0.07749079, -0.07013026,  0.88022116, -0.5546825 ,  0.25232708,
            0.09483573,  0.84910913,  1.33348018, -1.1971401 ,  0.49203123,
            0.22365435,  1.3801812 ,  0.06885929,  1.07115053, -1.52225141,
            1.50179721, -2.01528399, -1.31610679, -0.32298834, -0.80630885,
           -0.6828592 ,  0.2897919 ,  1.64608125, -0.71793662, -0.5233058 ,
            0.53549836,  0.61119221,  0.24235732, -1.3940593 ,  0.28380114,
           -0.22629978, -0.19318957,  1.12543101, -1.40328285,  0.21054137]),
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
        smc.SMC: np.array([
            -0.26421709, -2.07555186,  1.03443124,  0.16260898, -0.2809841 ,
            -0.35185097, -0.56387677,  0.18332851,  1.59614152,  0.39866217,
            -0.55781016, -0.74446992,  0.41198452,  0.47484429,  0.43417346,
             1.24153494,  1.10037457,  2.55408602, -1.47011338,  0.50824935,
            -2.09842977,  0.74269458,  0.31025837,  0.48376623,  1.74272003,
            -0.3975872 , -0.83735649, -0.33724478,  1.20300335,  1.40710795,
            -0.63740634, -0.33976389, -0.95412333,  1.84658352,  1.2000763 ,
            -1.08264783, -1.55367546,  0.66209331,  0.6577848 ,  0.5727828 ,
             0.30248057,  0.89674302,  0.70148518,  0.56483303,  1.35161821,
             0.06392528,  0.70670242,  1.04846633,  0.54696351, -2.49061003,
            -1.29925327, -1.31906407, -0.36650058, -1.44809118, -0.96224606,
            -0.2501728 , -1.88779999,  0.35774637,  1.06917986,  2.07049617,
            -0.18667668,  0.19360673, -0.37665179,  0.98526962,  1.03010772,
            -0.25348684,  2.43418902,  0.89153789, -1.02035572,  1.77851957,
             0.6408621 ,  0.50163095,  0.59934511,  0.73985647,  0.78719236,
            -0.41001864, -1.99859554,  1.53574307, -1.71336207,  1.04355849,
             0.21864817, -2.03911519, -0.42358936, -0.49666918,  1.64327219,
            -0.86416032,  1.10236002,  0.16396354, -0.13313781,  0.32649281,
            -1.01918397,  0.20525201,  1.04927506,  0.98243013,  2.46970704,
            -0.68709777,  2.05038381,  0.71417231,  1.13267395, -0.48644823]),
    }

    def setup_class(self):
        self.temp_dir = tempfile.mkdtemp()
        print(self.temp_dir)

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
                trace = smc.sample_smc(n_steps=n_steps,
                                       step=step_method(random_seed=1),
                                       n_jobs=1, progressbar=False,
                                       homepath=self.temp_dir)
            elif step_method.__name__ == 'NUTS':
                step = step_method(scaling=model.test_point)
                trace = sample(0, tune=n_steps,
                               discard_tuned_samples=False,
                               step=step, random_seed=1)
            else:
                trace = sample(0, tune=n_steps,
                               discard_tuned_samples=False,
                               step=step_method(), random_seed=1)

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
            trace = sample(0, tune=8000,
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
            trace = sample(20000, tune=0, step=step, start=start, model=model, random_seed=1)
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
            trace = sample(5000, tune=0, step=step, start=start, model=model, random_seed=1)
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
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS)

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


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestNutsCheckTrace(object):
    def test_multiple_samplers(self):
        with Model():
            prob = Beta('prob', alpha=5., beta=3.)
            Binomial('outcome', n=1, p=prob)
            with pytest.warns(None) as warns:
                sample(3, tune=2, discard_tuned_samples=False,
                       n_init=None)
            messages = [warn.message.args[0] for warn in warns]
            assert any("contains only 3" in msg for msg in messages)
            assert all('boolean index did not' not in msg for msg in messages)

    def test_bad_init(self):
        with Model():
            HalfNormal('a', sd=1, testval=-1, transform=None)
            with pytest.raises(ValueError) as error:
                sample(init=None)
            error.match('Bad initial')

    def test_linalg(self):
        with Model():
            a = Normal('a', shape=2)
            a = tt.switch(a > 0, np.inf, a)
            b = tt.slinalg.solve(floatX(np.eye(2)), a)
            Normal('c', mu=b, shape=2)
            with pytest.warns(None) as warns:
                trace = sample(20, init=None, tune=5)
            assert np.any(trace['diverging'])
            assert any('diverging samples after tuning' in str(warn.message)
                       for warn in warns)
            assert any('contains only' in str(warn.message) for warn in warns)

            with pytest.raises(SamplingError):
                sample(20, init=None, nuts_kwargs={'on_error': 'raise'})
