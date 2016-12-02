import os
import unittest

from .checks import close_to
from .models import simple_categorical, mv_simple, mv_simple_discrete, simple_2model
from pymc3.sampling import assign_step_methods, sample
from pymc3.model import Model
from pymc3.step_methods import (NUTS, BinaryGibbsMetropolis, CategoricalGibbsMetropolis,
                                Metropolis, Slice, CompoundStep,
                                MultivariateNormalProposal, HamiltonianMC)
from pymc3.distributions import Binomial, Normal, Bernoulli, Categorical

from numpy.testing import assert_array_almost_equal
import numpy as np
from tqdm import tqdm


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
            -1.56440708e-03, -2.37766120e-03, -6.95819902e-03, -4.88882715e-03, -6.54928517e-03,
            -3.38653286e-03, -1.99381372e-03, -1.25904805e-03, -2.97173572e-04, -4.67391216e-04,
            -2.03821237e-03, -1.33693751e-04, -2.17293248e-03, -4.11675406e-03, -4.23091782e-03,
            -7.34120851e-03, -8.43726968e-03, -7.86976139e-03, -3.89551467e-03, -3.00788956e-03,
            -3.82420513e-03, -1.35604792e-03, -2.49066947e-04, 4.03633859e-04, 9.34321408e-05,
            1.77722574e-03, 1.63761359e-03, 2.86208401e-03, -1.72243038e-04, 1.86863525e-03,
            1.76740215e-03, 1.79169049e-03, 1.07164602e-03, 1.41264547e-03, 2.49563456e-03,
            1.76639216e-03, 3.01570589e-03, 1.44186424e-04, 1.45073846e-03, 2.95031617e-04,
            -1.28811479e-04, -7.35945905e-04, -6.00689088e-04, 2.75468405e-04, 1.05245800e-03,
            1.18892307e-03, 6.01165842e-04, 1.21016955e-03, -2.06751271e-03, -8.41426458e-04,
            6.09905557e-04, 2.92765303e-03, 4.15216348e-03, 2.71863268e-03, 3.42922082e-03,
            7.53890188e-03, 7.97507867e-03, 8.27371677e-03, 9.77811135e-03, 9.99705714e-03,
            1.13996054e-02, 1.15745874e-02, 1.08182152e-02, 1.08277279e-02, 9.32254191e-03,
            8.59914793e-03, 8.43927425e-03, 1.01570101e-02, 9.74607039e-03, 9.82868496e-03,
            1.01745777e-02, 1.19312194e-02, 1.53760522e-02, 1.38691940e-02, 1.40131760e-02,
            1.46184561e-02, 1.74382675e-02, 1.84241543e-02, 2.06913002e-02, 1.83520531e-02,
            2.03072531e-02, 1.72912752e-02, 1.38959101e-02, 1.21933473e-02, 1.05084488e-02,
            9.00532336e-03, 9.25863206e-03, 1.23618461e-02, 1.20207293e-02, 1.09334818e-02,
            1.16528011e-02, 1.29967126e-02, 1.38940942e-02, 1.11408833e-02, 1.09263348e-02,
            1.06521352e-02, 1.01622526e-02, 1.21998547e-02, 1.00880470e-02, 9.94787795e-03]),
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
            0.68819657, 0.1767813, -0.59467679, -0.64216066, 1.63681405, 2.13404699, 0.03126563,
            0.31817152, 0.31817152, 0.40191527, 0.40191527, 0.99220141, 0.93036804, -0.41228181,
            -1.80465851, -1.70577291, 0.19406438, 0.19406438, -0.03965181, -0.76135744,
            0.70023098, 1.07183677, 1.07183677, 0.2829979, 1.13524135, -0.26461224,
            -0.39442329, -1.04109657, 0.79971205, 0.79971205, 0.96839778, 0.91868626,
            0.19468837, 0.19468837, -0.67755668, -0.67755668, -0.43722432, 0.12072881,
            0.6267432, 0.6861771, 0.4669198, 0.4669198, -0.08143768, 0.27691068, 0.11510718,
            2.29821426, 2.18308403, 1.16618069, -0.45615197, -0.45615197, -0.37076172,
            -0.37076172, -0.38889599, 0.36200553, -0.55179735, -0.55179735, -0.18946703,
            1.11552335, 0.98985795, 0.98985795, 1.00313687, -0.18458164, 0.44025584, 0.97610126,
            -0.1558578, -0.1558578, -0.01247235, -0.08303131, 0.52019377, -1.52329796,
            -1.72856248, -1.19049049, -1.19049049, -0.8651521, -0.36421118, -0.40590409,
            -0.78925074, -0.53960924, -0.53960924, 0.1069186, 0.40849997, 0.1560954,
            0.35461684, 0.35461684, -0.83935418, -0.85295353, -0.13990269, -0.1412904,
            -0.1412904, -0.30071575, -0.296461, 0.06540186, -0.15145479, -0.15145479,
            -0.21406771, -0.21533218, 0.06833495, 0.06833495, -0.18763595, 0.34138144]),
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
