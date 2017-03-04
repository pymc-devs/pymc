import unittest

import pymc3 as pm
import numpy as np
import numpy.testing as npt
from scipy import stats


from .helpers import SeededTest


class KnownMean(unittest.TestCase):
    def test_mean(self):
        for varname, expected in self.means.items():
            samples = self.samples[varname]
            npt.assert_allclose(expected, samples.mean(0), self.rtol, self.atol)


class KnownVariance(unittest.TestCase):
    def test_var(self):
        for varname, expected in self.variances.items():
            samples = self.samples[varname]
            npt.assert_allclose(expected, samples.var(0), self.rtol, self.atol)


class KnownCDF(unittest.TestCase):
    ks_thin = 5
    alpha = 0.001

    def test_kstest(self):
        for varname, cdf in self.cdfs.items():
            samples = self.samples[varname]
            if samples.ndim == 1:
                t, p = stats.kstest(samples[::self.ks_thin], cdf=cdf)
                self.assertLess(self.alpha, p)
            elif samples.ndim == 2:
                pvals = []
                for samples_, cdf_ in zip(samples.T, cdf):
                    t, p = stats.kstest(samples_[::self.ks_thin], cdf=cdf_)
                    pvals.append(p)
                t, p = stats.combine_pvalues(pvals)
                self.assertLess(self.alpha, p)
            else:
                raise NotImplementedError()



class UniformFixture(KnownMean, KnownVariance, KnownCDF):
    means = {'a': 0}
    variances = {'a': 1.0 / 3}
    cdfs = {'a': stats.uniform(-1, 2).cdf}

    @classmethod
    def make_model(cls):
        model = pm.Model()
        with model:
            a = pm.Uniform("a", lower=-1, upper=1)
        return model


class NormalFixture(KnownMean, KnownVariance, KnownCDF):
    means = {'a': 2 * np.ones(10)}
    variances = {'a': 3 * np.ones(10)}
    cdfs = {'a': [stats.norm(2, np.sqrt(3)).cdf for _ in range(10)]}

    @classmethod
    def make_model(cls):
        with pm.Model() as model:
            a = pm.Normal("a", mu=2, sd=np.sqrt(3), shape=10)
        return model


class BetaBinomialFixture(KnownCDF):
    cdfs = {'p': [stats.beta(a, b).cdf
                  for a, b in zip([1.5, 2.5, 10], [3.5, 10.5, 1])]}

    @classmethod
    def make_model(cls):
        with pm.Model() as model:
            p = pm.Beta("p", [0.5, 0.5, 1.], [0.5, 0.5, 1.], shape=3)
            pm.Binomial("y", p=p, n=[4, 12, 9], observed=[1, 2, 9])
        return model


class StudentTFixture(KnownMean, KnownCDF):
    means = {'a': 0}
    cdfs = {'a': stats.t(df=3).cdf}
    ks_thin = 10

    @classmethod
    def make_model(cls):
        with pm.Model() as model:
            a = pm.StudentT("a", nu=3, mu=0, sd=1)
        return model


class BaseSampler(SeededTest):
    @classmethod
    def setUpClass(cls):
        super(BaseSampler, cls).setUpClass()
        cls.model = cls.make_model()
        with cls.model:
            cls.step = cls.make_step()
            cls.trace = pm.sample(
                cls.n_samples, tune=cls.tune, step=cls.step, njobs=cls.chains)
        cls.samples = {}
        for var in cls.model.unobserved_RVs:
            cls.samples[str(var)] = cls.trace.get_values(var, burn=cls.burn)

    def test_neff(self):
        if hasattr(self, 'min_n_eff'):
            n_eff = pm.effective_n(self.trace[self.burn:])
            for var in n_eff:
                npt.assert_array_less(self.min_n_eff, n_eff[var])

    def test_Rhat(self):
        rhat = pm.gelman_rubin(self.trace[self.burn:])
        for var in rhat:
            npt.assert_allclose(rhat[var], 1, rtol=0.01)


class NutsFixture(BaseSampler):
    @classmethod
    def make_step(cls):
        args = {}
        if hasattr(cls, 'step_args'):
            args.update(cls.step_args)
        return pm.NUTS(**args)

    def test_target_accept(self):
        accept = self.trace[self.burn:]['mean_tree_accept']
        npt.assert_allclose(accept.mean(), self.step.target_accept, 1)


class MetropolisFixture(BaseSampler):
    @classmethod
    def make_step(cls):
        args = {}
        if hasattr(cls, 'step_args'):
            args.update(cls.step_args)
        return pm.Metropolis(**args)


class SliceFixture(BaseSampler):
    @classmethod
    def make_step(cls):
        args = {}
        if hasattr(cls, 'step_args'):
            args.update(cls.step_args)
        return pm.Slice(**args)
