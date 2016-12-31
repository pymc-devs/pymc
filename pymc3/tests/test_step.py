import os
import unittest

from .checks import close_to
from .models import simple_categorical, mv_simple, mv_simple_discrete, simple_2model
from .helpers import SeededTest
from pymc3 import df_summary, traceplot
from pymc3.sampling import assign_step_methods, sample
from pymc3.model import Model
from pymc3.step_methods import (NUTS, BinaryGibbsMetropolis, CategoricalGibbsMetropolis,
                                Metropolis, Slice, CompoundStep,
                                MultivariateNormalProposal, HamiltonianMC)
from pymc3.distributions import Binomial, Normal, Bernoulli, Categorical, InverseGamma

from numpy.testing import assert_array_almost_equal
import numpy as np
from tqdm import tqdm
from scipy import stats


class TestStepMethods(object):  # yield test doesn't work subclassing unittest.TestCase
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
        NUTS: np.array([
            1.11832371,  1.11832371,  0.6296164 , -1.88725852, -0.28085593,
           -0.28085593,  0.51246481,  0.51246481,  0.51524239, -1.07479086,
           -1.42956404, -1.42956404,  0.236747  , -1.04721507, -0.9716711 ,
           -0.9716711 , -0.65903886,  1.392075  ,  0.97569367,  0.16332534,
            0.16332534, -0.36465255,  1.4513665 ,  0.60044829,  0.60044829,
           -0.7582248 , -0.81863678, -0.8432519 ,  0.4756505 , -1.8618275 ,
           -1.28026218, -1.28026218, -1.39438141, -1.33699454,  1.65436385,
            1.18307331,  1.18307331, -1.02586036, -0.49676467, -0.17723852,
           -0.17723852, -0.46433406,  0.80562527,  0.0872473 ,  0.0872473 ,
           -0.5121292 ,  0.2457691 ,  0.2457691 , -0.28227229, -0.56863496,
           -0.13361197,  0.40085491, -0.34774478, -0.34774478, -0.49552973,
           -0.50378818,  0.26151237,  0.26151237, -0.3271885 , -0.33567672,
            0.9339671 ,  0.92457538,  0.92457538,  0.92079262,  0.07603108,
            0.53798844,  0.5509594 , -0.10204449, -0.10204449, -0.27679836,
            0.29559731, -0.63318597,  0.50441029,  0.50441029, -0.64408839,
            0.85784078,  0.83528853,  0.32389337,  0.32389337,  0.32371787,
            0.45515893,  0.42012555,  0.42012555, -0.02036946, -0.1275346 ,
            0.1818874 ,  1.28333928,  0.92705793,  0.92705793,  0.42301906,
            1.97444363,  2.12202997,  1.79560373,  1.79560373,  1.658178  ,
            1.64034201,  2.01737179,  1.45213152,  1.45213152,  1.4536979 ]),
    }

    def test_sample_exact(self):
        for step_method in self.master_samples:
            yield self.check_trace, step_method

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
        test_steps = 100
        n_steps = int(os.getenv('BENCHMARK', 100))
        benchmarking = (n_steps != test_steps)
        if benchmarking:
            tqdm.write('Benchmarking {} with {:,d} samples'.format(step_method.__name__, n_steps))
        else:
            tqdm.write('Checking {} has same trace as on master'.format(step_method.__name__))
        with Model():
            Normal('x', mu=0, sd=1)
            trace = sample(n_steps, step=step_method(), random_seed=1)

        if not benchmarking:
            assert_array_almost_equal(trace.get_values('x'), self.master_samples[step_method])

    def check_stat(self, check, trace):
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
            trace = sample(8000, step=step, start=start, model=model, random_seed=1)
            yield self.check_stat, check, trace

    def test_step_discrete(self):
        start, model, (mu, C) = mv_simple_discrete()
        unc = np.diag(C) ** .5
        check = (('x', np.mean, mu, unc / 10.),
                 ('x', np.std, unc, unc / 10.))
        with model:
            steps = (
                Metropolis(S=C, proposal_dist=MultivariateNormalProposal),
            )
        for step in steps:
            trace = sample(20000, step=step, start=start, model=model, random_seed=1)
            self.check_stat(check, trace)

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
            trace = sample(8000, step=step, start=start, model=model, random_seed=1)
            self.check_stat(check, trace)


class TestCompoundStep(unittest.TestCase):
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS)

    def test_non_blocked(self):
        """Test that samplers correctly create non-blocked compound steps."""
        _, model = simple_2model()
        with model:
            for sampler in self.samplers:
                self.assertIsInstance(sampler(blocked=False), CompoundStep)

    def test_blocked(self):
        _, model = simple_2model()
        with model:
            for sampler in self.samplers:
                sampler_instance = sampler(blocked=True)
                self.assertNotIsInstance(sampler_instance, CompoundStep)
                self.assertIsInstance(sampler_instance, sampler)


class TestAssignStepMethods(unittest.TestCase):
    def test_bernoulli(self):
        """Test bernoulli distribution is assigned binary gibbs metropolis method"""
        with Model() as model:
            Bernoulli('x', 0.5)
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, BinaryGibbsMetropolis)

    def test_normal(self):
        """Test normal distribution is assigned NUTS method"""
        with Model() as model:
            Normal('x', 0, 1)
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, NUTS)

    def test_categorical(self):
        """Test categorical distribution is assigned categorical gibbs metropolis method"""
        with Model() as model:
            Categorical('x', np.array([0.25, 0.75]))
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, BinaryGibbsMetropolis)
        with Model() as model:
            Categorical('y', np.array([0.25, 0.70, 0.05]))
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, CategoricalGibbsMetropolis)

    def test_binomial(self):
        """Test binomial distribution is assigned metropolis method."""
        with Model() as model:
            Binomial('x', 10, 0.5)
            steps = assign_step_methods(model, [])
        self.assertIsInstance(steps, Metropolis)


class TestSampleEstimates(SeededTest):
    def test_posterior_estimate(self):
        alpha_true, sigma_true = 1., 0.5
        beta_true = 1.

        size = 1000

        X = np.random.randn(size)
        Y = alpha_true + beta_true * X + np.random.randn(size) * sigma_true

        decimal = 1
        with Model() as model:
            alpha = Normal('alpha', mu=0, sd=100, testval=alpha_true)
            beta = Normal('beta', mu=0, sd=100, testval=beta_true)
            sigma = InverseGamma('sigma', 10., testval=sigma_true)
            mu = alpha + beta * X
            Y_obs = Normal('Y_obs', mu=mu, sd=sigma, observed=Y)

            for step_method in (NUTS, Slice, Metropolis):
                trace = sample(100000, step=step_method(), progressbar=False)
                trace_ = trace[-300::5]

                # We do the same for beta - using more burnin.
                np.testing.assert_almost_equal(np.mean(trace_.alpha),
                                               alpha_true, decimal=decimal)
                np.testing.assert_almost_equal(np.mean(trace_.beta),
                                               beta_true,
                                               decimal=decimal)
                np.testing.assert_almost_equal(np.mean(trace_.sigma),
                                               sigma_true, decimal=decimal)

                # Make sure posteriors are normal
                _, p_alpha = stats.normaltest(trace_.alpha)
                _, p_beta = stats.normaltest(trace_.beta)
                # p-values should be > .05 to indiciate
                np.testing.assert_array_less(0.05, p_alpha, verbose=True)
                np.testing.assert_array_less(0.05, p_beta, verbose=True)
