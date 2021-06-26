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

import arviz as az
import numpy as np
import numpy.testing as npt
import theano.tensor as tt

from scipy import stats

import pymc3 as pm

from pymc3.tests.helpers import SeededTest
from pymc3.util import get_var_name


class KnownMean:
    def test_mean(self):
        for varname, expected in self.means.items():
            samples = self.samples[varname]
            npt.assert_allclose(expected, samples.mean(0), self.rtol, self.atol)


class KnownVariance:
    def test_var(self):
        for varname, expected in self.variances.items():
            samples = self.samples[varname]
            npt.assert_allclose(expected, samples.var(0), self.rtol, self.atol)


class KnownCDF:
    ks_thin = 5
    alpha = 0.001

    def test_kstest(self):
        for varname, cdf in self.cdfs.items():
            samples = self.samples[varname]
            if samples.ndim == 1:
                t, p = stats.kstest(samples[:: self.ks_thin], cdf=cdf)
                assert self.alpha < p
            elif samples.ndim == 2:
                pvals = []
                for samples_, cdf_ in zip(samples.T, cdf):
                    t, p = stats.kstest(samples_[:: self.ks_thin], cdf=cdf_)
                    pvals.append(p)
                t, p = stats.combine_pvalues(pvals)
                assert self.alpha < p
            else:
                raise NotImplementedError()


class UniformFixture(KnownMean, KnownVariance, KnownCDF):
    means = {"a": 0}
    variances = {"a": 1.0 / 3}
    cdfs = {"a": stats.uniform(-1, 2).cdf}

    @classmethod
    def make_model(cls):
        model = pm.Model()
        with model:
            a = pm.Uniform("a", lower=-1, upper=1)
        return model


class NormalFixture(KnownMean, KnownVariance, KnownCDF):
    means = {"a": 2 * np.ones(10)}
    variances = {"a": 3 * np.ones(10)}
    cdfs = {"a": [stats.norm(2, np.sqrt(3)).cdf for _ in range(10)]}

    @classmethod
    def make_model(cls):
        with pm.Model() as model:
            a = pm.Normal("a", mu=2, sigma=np.sqrt(3), shape=10)
        return model


class BetaBinomialFixture(KnownCDF):
    cdfs = {"p": [stats.beta(a, b).cdf for a, b in zip([1.5, 2.5, 10], [3.5, 10.5, 1])]}

    @classmethod
    def make_model(cls):
        with pm.Model() as model:
            p = pm.Beta("p", [0.5, 0.5, 1.0], [0.5, 0.5, 1.0], shape=3)
            pm.Binomial("y", p=p, n=[4, 12, 9], observed=[1, 2, 9])
        return model


class StudentTFixture(KnownMean, KnownCDF):
    means = {"a": 0}
    cdfs = {"a": stats.t(df=4).cdf}
    ks_thin = 10

    @classmethod
    def make_model(cls):
        with pm.Model() as model:
            a = pm.StudentT("a", nu=4, mu=0, sigma=1)
        return model


class LKJCholeskyCovFixture(KnownCDF):
    cdfs = {
        "log_stds": [stats.norm(loc=x, scale=x / 10.0).cdf for x in [1, 2, 3, 4, 5]],
        # The entries of the correlation matrix should follow
        # beta(eta - 1 + d/2, eta - 1 + d/2) on (-1, 1).
        # See https://arxiv.org/abs/1309.7268
        "corr_entries_unit": [stats.beta(3 - 1 + 2.5, 3 - 1 + 2.5).cdf for _ in range(10)],
    }

    @classmethod
    def make_model(cls):
        with pm.Model() as model:
            sd_mu = np.array([1, 2, 3, 4, 5])
            sd_dist = pm.Lognormal.dist(mu=sd_mu, sigma=sd_mu / 10.0, shape=5)
            chol_packed = pm.LKJCholeskyCov("chol_packed", eta=3, n=5, sd_dist=sd_dist)
            chol = pm.expand_packed_triangular(5, chol_packed, lower=True)
            cov = tt.dot(chol, chol.T)
            stds = tt.sqrt(tt.diag(cov))
            pm.Deterministic("log_stds", tt.log(stds))
            corr = cov / stds[None, :] / stds[:, None]
            corr_entries_unit = (corr[np.tril_indices(5, -1)] + 1) / 2
            pm.Deterministic("corr_entries_unit", corr_entries_unit)
        return model


class BaseSampler(SeededTest):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.model = cls.make_model()
        with cls.model:
            cls.step = cls.make_step()
            cls.trace = pm.sample(
                cls.n_samples,
                tune=cls.tune,
                step=cls.step,
                cores=cls.chains,
                return_inferencedata=False,
            )
        cls.samples = {}
        for var in cls.model.unobserved_RVs:
            cls.samples[get_var_name(var)] = cls.trace.get_values(var, burn=cls.burn)

    def test_neff(self):
        if hasattr(self, "min_n_eff"):
            n_eff = az.ess(self.trace[self.burn :])
            for var in n_eff:
                npt.assert_array_less(self.min_n_eff, n_eff[var])

    def test_Rhat(self):
        rhat = az.rhat(self.trace[self.burn :])
        for var in rhat:
            npt.assert_allclose(rhat[var], 1, rtol=0.01)


class NutsFixture(BaseSampler):
    @classmethod
    def make_step(cls):
        args = {}
        if hasattr(cls, "step_args"):
            args.update(cls.step_args)
        if "scaling" not in args:
            _, step = pm.sampling.init_nuts(n_init=10000, **args)
        else:
            step = pm.NUTS(**args)
        return step

    def test_target_accept(self):
        accept = self.trace[self.burn :]["mean_tree_accept"]
        npt.assert_allclose(accept.mean(), self.step.target_accept, 1)


class MetropolisFixture(BaseSampler):
    @classmethod
    def make_step(cls):
        args = {}
        if hasattr(cls, "step_args"):
            args.update(cls.step_args)
        return pm.Metropolis(**args)


class SliceFixture(BaseSampler):
    @classmethod
    def make_step(cls):
        args = {}
        if hasattr(cls, "step_args"):
            args.update(cls.step_args)
        return pm.Slice(**args)
