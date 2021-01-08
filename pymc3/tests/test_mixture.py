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

import numpy as np
import pytest
import scipy.stats as st
import theano

from numpy.testing import assert_allclose
from scipy.special import logsumexp
from theano import tensor as tt

import pymc3 as pm

from pymc3 import (
    Dirichlet,
    Exponential,
    Gamma,
    Lognormal,
    Metropolis,
    Mixture,
    Model,
    MvNormal,
    Normal,
    NormalMixture,
    Poisson,
    sample,
)
from pymc3.distributions.shape_utils import to_tuple
from pymc3.tests.helpers import SeededTest
from pymc3.theanof import floatX


# Generate data
def generate_normal_mixture_data(w, mu, sd, size=1000):
    component = np.random.choice(w.size, size=size, p=w)
    mu, sd = np.broadcast_arrays(mu, sd)
    out_size = to_tuple(size) + mu.shape[:-1]
    mu_ = np.array([mu[..., comp] for comp in component.ravel()])
    sd_ = np.array([sd[..., comp] for comp in component.ravel()])
    mu_ = np.reshape(mu_, out_size)
    sd_ = np.reshape(sd_, out_size)
    x = np.random.normal(mu_, sd_, size=out_size)

    return x


def generate_poisson_mixture_data(w, mu, size=1000):
    component = np.random.choice(w.size, size=size, p=w)
    mu = np.atleast_1d(mu)
    out_size = to_tuple(size) + mu.shape[:-1]
    mu_ = np.array([mu[..., comp] for comp in component.ravel()])
    mu_ = np.reshape(mu_, out_size)
    x = np.random.poisson(mu_, size=out_size)

    return x


class TestMixture(SeededTest):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        cls.norm_w = np.array([0.75, 0.25])
        cls.norm_mu = np.array([0.0, 5.0])
        cls.norm_sd = np.ones_like(cls.norm_mu)
        cls.norm_x = generate_normal_mixture_data(cls.norm_w, cls.norm_mu, cls.norm_sd, size=1000)

        cls.pois_w = np.array([0.4, 0.6])
        cls.pois_mu = np.array([5.0, 20.0])
        cls.pois_x = generate_poisson_mixture_data(cls.pois_w, cls.pois_mu, size=1000)

    def test_dimensions(self):
        a1 = Normal.dist(mu=0, sigma=1)
        a2 = Normal.dist(mu=10, sigma=1)
        mix = Mixture.dist(w=np.r_[0.5, 0.5], comp_dists=[a1, a2])

        assert mix.mode.ndim == 0
        assert mix.logp(0.0).ndim == 0

        value = np.r_[0.0, 1.0, 2.0]
        assert mix.logp(value).ndim == 1

    def test_mixture_list_of_normals(self):
        with Model() as model:
            w = Dirichlet("w", floatX(np.ones_like(self.norm_w)), shape=self.norm_w.size)
            mu = Normal("mu", 0.0, 10.0, shape=self.norm_w.size)
            tau = Gamma("tau", 1.0, 1.0, shape=self.norm_w.size)
            Mixture(
                "x_obs",
                w,
                [Normal.dist(mu[0], tau=tau[0]), Normal.dist(mu[1], tau=tau[1])],
                observed=self.norm_x,
            )
            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed, progressbar=False, chains=1)

        assert_allclose(np.sort(trace["w"].mean(axis=0)), np.sort(self.norm_w), rtol=0.1, atol=0.1)
        assert_allclose(
            np.sort(trace["mu"].mean(axis=0)), np.sort(self.norm_mu), rtol=0.1, atol=0.1
        )

    def test_normal_mixture(self):
        with Model() as model:
            w = Dirichlet("w", floatX(np.ones_like(self.norm_w)), shape=self.norm_w.size)
            mu = Normal("mu", 0.0, 10.0, shape=self.norm_w.size)
            tau = Gamma("tau", 1.0, 1.0, shape=self.norm_w.size)
            NormalMixture("x_obs", w, mu, tau=tau, observed=self.norm_x)
            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed, progressbar=False, chains=1)

        assert_allclose(np.sort(trace["w"].mean(axis=0)), np.sort(self.norm_w), rtol=0.1, atol=0.1)
        assert_allclose(
            np.sort(trace["mu"].mean(axis=0)), np.sort(self.norm_mu), rtol=0.1, atol=0.1
        )

    @pytest.mark.parametrize(
        "nd,ncomp", [(tuple(), 5), (1, 5), (3, 5), ((3, 3), 5), (3, 3), ((3, 3), 3)], ids=str
    )
    def test_normal_mixture_nd(self, nd, ncomp):
        nd = to_tuple(nd)
        ncomp = int(ncomp)
        comp_shape = nd + (ncomp,)
        test_mus = np.random.randn(*comp_shape)
        test_taus = np.random.gamma(1, 1, size=comp_shape)
        observed = generate_normal_mixture_data(
            w=np.ones(ncomp) / ncomp, mu=test_mus, sd=1 / np.sqrt(test_taus), size=10
        )

        with Model() as model0:
            mus = Normal("mus", shape=comp_shape)
            taus = Gamma("taus", alpha=1, beta=1, shape=comp_shape)
            ws = Dirichlet("ws", np.ones(ncomp), shape=(ncomp,))
            mixture0 = NormalMixture("m", w=ws, mu=mus, tau=taus, shape=nd, comp_shape=comp_shape)
            obs0 = NormalMixture(
                "obs", w=ws, mu=mus, tau=taus, shape=nd, comp_shape=comp_shape, observed=observed
            )

        with Model() as model1:
            mus = Normal("mus", shape=comp_shape)
            taus = Gamma("taus", alpha=1, beta=1, shape=comp_shape)
            ws = Dirichlet("ws", np.ones(ncomp), shape=(ncomp,))
            comp_dist = [
                Normal.dist(mu=mus[..., i], tau=taus[..., i], shape=nd) for i in range(ncomp)
            ]
            mixture1 = Mixture("m", w=ws, comp_dists=comp_dist, shape=nd)
            obs1 = Mixture("obs", w=ws, comp_dists=comp_dist, shape=nd, observed=observed)

        with Model() as model2:
            # Expected to fail if comp_shape is not provided,
            # nd is multidim and it does not broadcast with ncomp. If by chance
            # it does broadcast, an error is raised if the mixture is given
            # observed data.
            # Furthermore, the Mixture will also raise errors when the observed
            # data is multidimensional but it does not broadcast well with
            # comp_dists.
            mus = Normal("mus", shape=comp_shape)
            taus = Gamma("taus", alpha=1, beta=1, shape=comp_shape)
            ws = Dirichlet("ws", np.ones(ncomp), shape=(ncomp,))
            if len(nd) > 1:
                if nd[-1] != ncomp:
                    with pytest.raises(ValueError):
                        NormalMixture("m", w=ws, mu=mus, tau=taus, shape=nd)
                    mixture2 = None
                else:
                    mixture2 = NormalMixture("m", w=ws, mu=mus, tau=taus, shape=nd)
            else:
                mixture2 = NormalMixture("m", w=ws, mu=mus, tau=taus, shape=nd)
            observed_fails = False
            if len(nd) >= 1 and nd != (1,):
                try:
                    np.broadcast(np.empty(comp_shape), observed)
                except Exception:
                    observed_fails = True
            if observed_fails:
                with pytest.raises(ValueError):
                    NormalMixture("obs", w=ws, mu=mus, tau=taus, shape=nd, observed=observed)
                obs2 = None
            else:
                obs2 = NormalMixture("obs", w=ws, mu=mus, tau=taus, shape=nd, observed=observed)

        testpoint = model0.test_point
        testpoint["mus"] = test_mus
        testpoint["taus"] = test_taus
        assert_allclose(model0.logp(testpoint), model1.logp(testpoint))
        assert_allclose(mixture0.logp(testpoint), mixture1.logp(testpoint))
        assert_allclose(obs0.logp(testpoint), obs1.logp(testpoint))
        if mixture2 is not None and obs2 is not None:
            assert_allclose(model0.logp(testpoint), model2.logp(testpoint))
        if mixture2 is not None:
            assert_allclose(mixture0.logp(testpoint), mixture2.logp(testpoint))
        if obs2 is not None:
            assert_allclose(obs0.logp(testpoint), obs2.logp(testpoint))

    def test_poisson_mixture(self):
        with Model() as model:
            w = Dirichlet("w", floatX(np.ones_like(self.pois_w)), shape=self.pois_w.shape)
            mu = Gamma("mu", 1.0, 1.0, shape=self.pois_w.size)
            Mixture("x_obs", w, Poisson.dist(mu), observed=self.pois_x)
            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed, progressbar=False, chains=1)

        assert_allclose(np.sort(trace["w"].mean(axis=0)), np.sort(self.pois_w), rtol=0.1, atol=0.1)
        assert_allclose(
            np.sort(trace["mu"].mean(axis=0)), np.sort(self.pois_mu), rtol=0.1, atol=0.1
        )

    def test_mixture_list_of_poissons(self):
        with Model() as model:
            w = Dirichlet("w", floatX(np.ones_like(self.pois_w)), shape=self.pois_w.shape)
            mu = Gamma("mu", 1.0, 1.0, shape=self.pois_w.size)
            Mixture("x_obs", w, [Poisson.dist(mu[0]), Poisson.dist(mu[1])], observed=self.pois_x)
            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed, progressbar=False, chains=1)

        assert_allclose(np.sort(trace["w"].mean(axis=0)), np.sort(self.pois_w), rtol=0.1, atol=0.1)
        assert_allclose(
            np.sort(trace["mu"].mean(axis=0)), np.sort(self.pois_mu), rtol=0.1, atol=0.1
        )

    def test_mixture_of_mvn(self):
        mu1 = np.asarray([0.0, 1.0])
        cov1 = np.diag([1.5, 2.5])
        mu2 = np.asarray([1.0, 0.0])
        cov2 = np.diag([2.5, 3.5])
        obs = np.asarray([[0.5, 0.5], mu1, mu2])
        with Model() as model:
            w = Dirichlet("w", floatX(np.ones(2)), transform=None, shape=(2,))
            mvncomp1 = MvNormal.dist(mu=mu1, cov=cov1)
            mvncomp2 = MvNormal.dist(mu=mu2, cov=cov2)
            y = Mixture("x_obs", w, [mvncomp1, mvncomp2], observed=obs)

        # check logp of each component
        complogp_st = np.vstack(
            (
                st.multivariate_normal.logpdf(obs, mu1, cov1),
                st.multivariate_normal.logpdf(obs, mu2, cov2),
            )
        ).T
        complogp = y.distribution._comp_logp(theano.shared(obs)).eval()
        assert_allclose(complogp, complogp_st)

        # check logp of mixture
        testpoint = model.test_point
        mixlogp_st = logsumexp(np.log(testpoint["w"]) + complogp_st, axis=-1, keepdims=False)
        assert_allclose(y.logp_elemwise(testpoint), mixlogp_st)

        # check logp of model
        priorlogp = st.dirichlet.logpdf(
            x=testpoint["w"],
            alpha=np.ones(2),
        )
        assert_allclose(model.logp(testpoint), mixlogp_st.sum() + priorlogp)

    def test_mixture_of_mixture(self):
        if theano.config.floatX == "float32":
            rtol = 1e-4
        else:
            rtol = 1e-7
        nbr = 4
        with Model() as model:
            # mixtures components
            g_comp = Normal.dist(
                mu=Exponential("mu_g", lam=1.0, shape=nbr, transform=None), sigma=1, shape=nbr
            )
            l_comp = Lognormal.dist(
                mu=Exponential("mu_l", lam=1.0, shape=nbr, transform=None), sigma=1, shape=nbr
            )
            # weight vector for the mixtures
            g_w = Dirichlet("g_w", a=floatX(np.ones(nbr) * 0.0000001), transform=None, shape=(nbr,))
            l_w = Dirichlet("l_w", a=floatX(np.ones(nbr) * 0.0000001), transform=None, shape=(nbr,))
            # mixture components
            g_mix = Mixture.dist(w=g_w, comp_dists=g_comp)
            l_mix = Mixture.dist(w=l_w, comp_dists=l_comp)
            # mixture of mixtures
            mix_w = Dirichlet("mix_w", a=floatX(np.ones(2)), transform=None, shape=(2,))
            mix = Mixture("mix", w=mix_w, comp_dists=[g_mix, l_mix], observed=np.exp(self.norm_x))

        test_point = model.test_point

        def mixmixlogp(value, point):
            floatX = theano.config.floatX
            priorlogp = (
                st.dirichlet.logpdf(
                    x=point["g_w"],
                    alpha=np.ones(nbr) * 0.0000001,
                ).astype(floatX)
                + st.expon.logpdf(x=point["mu_g"]).sum(dtype=floatX)
                + st.dirichlet.logpdf(
                    x=point["l_w"],
                    alpha=np.ones(nbr) * 0.0000001,
                ).astype(floatX)
                + st.expon.logpdf(x=point["mu_l"]).sum(dtype=floatX)
                + st.dirichlet.logpdf(
                    x=point["mix_w"],
                    alpha=np.ones(2),
                ).astype(floatX)
            )
            complogp1 = st.norm.logpdf(x=value, loc=point["mu_g"]).astype(floatX)
            mixlogp1 = logsumexp(
                np.log(point["g_w"]).astype(floatX) + complogp1, axis=-1, keepdims=True
            )
            complogp2 = st.lognorm.logpdf(value, 1.0, 0.0, np.exp(point["mu_l"])).astype(floatX)
            mixlogp2 = logsumexp(
                np.log(point["l_w"]).astype(floatX) + complogp2, axis=-1, keepdims=True
            )
            complogp_mix = np.concatenate((mixlogp1, mixlogp2), axis=1)
            mixmixlogpg = logsumexp(
                np.log(point["mix_w"]).astype(floatX) + complogp_mix, axis=-1, keepdims=False
            )
            return priorlogp, mixmixlogpg

        value = np.exp(self.norm_x)[:, None]
        priorlogp, mixmixlogpg = mixmixlogp(value, test_point)

        # check logp of mixture
        assert_allclose(mixmixlogpg, mix.logp_elemwise(test_point), rtol=rtol)

        # check model logp
        assert_allclose(priorlogp + mixmixlogpg.sum(), model.logp(test_point), rtol=rtol)

        # check input and check logp again
        test_point["g_w"] = np.asarray([0.1, 0.1, 0.2, 0.6])
        test_point["mu_g"] = np.exp(np.random.randn(nbr))
        priorlogp, mixmixlogpg = mixmixlogp(value, test_point)
        assert_allclose(mixmixlogpg, mix.logp_elemwise(test_point), rtol=rtol)
        assert_allclose(priorlogp + mixmixlogpg.sum(), model.logp(test_point), rtol=rtol)

    def test_sample_prior_and_posterior(self):
        def build_toy_dataset(N, K):
            pi = np.array([0.2, 0.5, 0.3])
            mus = [[1, 1, 1], [-1, -1, -1], [2, -2, 0]]
            stds = [[0.1, 0.1, 0.1], [0.1, 0.2, 0.2], [0.2, 0.3, 0.3]]
            x = np.zeros((N, 3), dtype=np.float32)
            y = np.zeros((N,), dtype=np.int)
            for n in range(N):
                k = np.argmax(np.random.multinomial(1, pi))
                x[n, :] = np.random.multivariate_normal(mus[k], np.diag(stds[k]))
                y[n] = k
            return x, y

        N = 100  # number of data points
        K = 3  # number of mixture components
        D = 3  # dimensionality of the data

        X, y = build_toy_dataset(N, K)

        with pm.Model() as model:
            pi = pm.Dirichlet("pi", np.ones(K), shape=(K,))

            comp_dist = []
            mu = []
            packed_chol = []
            chol = []
            for i in range(K):
                mu.append(pm.Normal("mu%i" % i, 0, 10, shape=D))
                packed_chol.append(
                    pm.LKJCholeskyCov(
                        "chol_cov_%i" % i, eta=2, n=D, sd_dist=pm.HalfNormal.dist(2.5)
                    )
                )
                chol.append(pm.expand_packed_triangular(D, packed_chol[i], lower=True))
                comp_dist.append(pm.MvNormal.dist(mu=mu[i], chol=chol[i], shape=D))

            pm.Mixture("x_obs", pi, comp_dist, observed=X)
        with model:
            trace = pm.sample(30, tune=10, chains=1)

        n_samples = 20
        with model:
            ppc = pm.sample_posterior_predictive(trace, n_samples)
            prior = pm.sample_prior_predictive(samples=n_samples)
        assert ppc["x_obs"].shape == (n_samples,) + X.shape
        assert prior["x_obs"].shape == (n_samples,) + X.shape
        assert prior["mu0"].shape == (n_samples, D)
        assert prior["chol_cov_0"].shape == (n_samples, D * (D + 1) // 2)


class TestMixtureVsLatent(SeededTest):
    def setup_method(self, *args, **kwargs):
        super().setup_method(*args, **kwargs)
        self.nd = 3
        self.npop = 3
        self.mus = tt.as_tensor_variable(
            np.tile(
                np.reshape(
                    np.arange(self.npop),
                    (
                        1,
                        -1,
                    ),
                ),
                (
                    self.nd,
                    1,
                ),
            )
        )

    def test_1d_w(self):
        nd = self.nd
        npop = self.npop
        mus = self.mus
        size = 100
        with pm.Model() as model:
            m = pm.NormalMixture(
                "m", w=np.ones(npop) / npop, mu=mus, sigma=1e-5, comp_shape=(nd, npop), shape=nd
            )
            z = pm.Categorical("z", p=np.ones(npop) / npop)
            latent_m = pm.Normal("latent_m", mu=mus[..., z], sigma=1e-5, shape=nd)

        m_val = m.random(size=size)
        latent_m_val = latent_m.random(size=size)
        assert m_val.shape == latent_m_val.shape
        # Test that each element in axis = -1 comes from the same mixture
        # component
        assert all(np.all(np.diff(m_val) < 1e-3, axis=-1))
        assert all(np.all(np.diff(latent_m_val) < 1e-3, axis=-1))

        self.samples_from_same_distribution(m_val, latent_m_val)
        self.logp_matches(m, latent_m, z, npop, model=model)

    def test_2d_w(self):
        nd = self.nd
        npop = self.npop
        mus = self.mus
        size = 100
        with pm.Model() as model:
            m = pm.NormalMixture(
                "m",
                w=np.ones((nd, npop)) / npop,
                mu=mus,
                sigma=1e-5,
                comp_shape=(nd, npop),
                shape=nd,
            )
            z = pm.Categorical("z", p=np.ones(npop) / npop, shape=nd)
            mu = tt.as_tensor_variable([mus[i, z[i]] for i in range(nd)])
            latent_m = pm.Normal("latent_m", mu=mu, sigma=1e-5, shape=nd)

        m_val = m.random(size=size)
        latent_m_val = latent_m.random(size=size)
        assert m_val.shape == latent_m_val.shape
        # Test that each element in axis = -1 can come from independent
        # components
        assert not all(np.all(np.diff(m_val) < 1e-3, axis=-1))
        assert not all(np.all(np.diff(latent_m_val) < 1e-3, axis=-1))

        self.samples_from_same_distribution(m_val, latent_m_val)
        self.logp_matches(m, latent_m, z, npop, model=model)

    def samples_from_same_distribution(self, *args):
        # Test if flattened samples distributions match (marginals match)
        _, p_marginal = st.ks_2samp(*[s.flatten() for s in args])
        # Test if correlations within non independent draws match
        _, p_correlation = st.ks_2samp(
            *[np.array([np.corrcoef(ss) for ss in s]).flatten() for s in args]
        )
        assert p_marginal >= 0.05 and p_correlation >= 0.05

    def logp_matches(self, mixture, latent_mix, z, npop, model):
        if theano.config.floatX == "float32":
            rtol = 1e-4
        else:
            rtol = 1e-7
        test_point = model.test_point
        test_point["latent_m"] = test_point["m"]
        mix_logp = mixture.logp(test_point)
        logps = []
        for component in range(npop):
            test_point["z"] = component * np.ones(z.distribution.shape)
            # Count the number of axes that should be broadcasted from z to
            # modify the logp
            sh1 = test_point["z"].shape
            sh2 = test_point["latent_m"].shape
            if len(sh1) > len(sh2):
                sh2 = (1,) * (len(sh1) - len(sh2)) + sh2
            elif len(sh2) > len(sh1):
                sh1 = (1,) * (len(sh2) - len(sh1)) + sh1
            reps = np.prod([s2 if s1 != s2 else 1 for s1, s2 in zip(sh1, sh2)])
            z_logp = z.logp(test_point) * reps
            logps.append(z_logp + latent_mix.logp(test_point))
        latent_mix_logp = logsumexp(np.array(logps), axis=0)
        assert_allclose(mix_logp, latent_mix_logp, rtol=rtol)


class TestMixtureSameFamily(SeededTest):
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.size = 50
        cls.n_samples = 1000
        cls.mixture_comps = 10

    @pytest.mark.parametrize("batch_shape", [(3, 4), (20,)], ids=str)
    def test_with_multinomial(self, batch_shape):
        p = np.random.uniform(size=(*batch_shape, self.mixture_comps, 3))
        n = 100 * np.ones((*batch_shape, 1))
        w = np.ones(self.mixture_comps) / self.mixture_comps
        mixture_axis = len(batch_shape)
        with pm.Model() as model:
            comp_dists = pm.Multinomial.dist(p=p, n=n, shape=(*batch_shape, self.mixture_comps, 3))
            mixture = pm.MixtureSameFamily(
                "mixture",
                w=w,
                comp_dists=comp_dists,
                mixture_axis=mixture_axis,
                shape=(*batch_shape, 3),
            )
            prior = pm.sample_prior_predictive(samples=self.n_samples)

        assert prior["mixture"].shape == (self.n_samples, *batch_shape, 3)
        assert mixture.random(size=self.size).shape == (self.size, *batch_shape, 3)

        if theano.config.floatX == "float32":
            rtol = 1e-4
        else:
            rtol = 1e-7

        comp_logp = comp_dists.logp(model.test_point["mixture"].reshape(*batch_shape, 1, 3))
        log_sum_exp = logsumexp(
            comp_logp.eval() + np.log(w)[..., None], axis=mixture_axis, keepdims=True
        ).sum()
        assert_allclose(
            model.logp(model.test_point),
            log_sum_exp,
            rtol,
        )

    # TODO: Handle case when `batch_shape` == `sample_shape`.
    # See https://github.com/pymc-devs/pymc3/issues/4185 for details.
    def test_with_mvnormal(self):
        # 10 batch, 3-variate Gaussian
        mu = np.random.randn(self.mixture_comps, 3)
        mat = np.random.randn(3, 3)
        cov = mat @ mat.T
        chol = np.linalg.cholesky(cov)
        w = np.ones(self.mixture_comps) / self.mixture_comps

        with pm.Model() as model:
            comp_dists = pm.MvNormal.dist(mu=mu, chol=chol, shape=(self.mixture_comps, 3))
            mixture = pm.MixtureSameFamily(
                "mixture", w=w, comp_dists=comp_dists, mixture_axis=0, shape=(3,)
            )
            prior = pm.sample_prior_predictive(samples=self.n_samples)

        assert prior["mixture"].shape == (self.n_samples, 3)
        assert mixture.random(size=self.size).shape == (self.size, 3)

        if theano.config.floatX == "float32":
            rtol = 1e-4
        else:
            rtol = 1e-7

        comp_logp = comp_dists.logp(model.test_point["mixture"].reshape(1, 3))
        log_sum_exp = logsumexp(
            comp_logp.eval() + np.log(w)[..., None], axis=0, keepdims=True
        ).sum()
        assert_allclose(
            model.logp(model.test_point),
            log_sum_exp,
            rtol,
        )

    def test_broadcasting_in_shape(self):
        with pm.Model() as model:
            mu = pm.Gamma("mu", 1.0, 1.0, shape=2)
            comp_dists = pm.Poisson.dist(mu, shape=2)
            mix = pm.MixtureSameFamily(
                "mix", w=np.ones(2) / 2, comp_dists=comp_dists, shape=(1000,)
            )
            prior = pm.sample_prior_predictive(samples=self.n_samples)

        assert prior["mix"].shape == (self.n_samples, 1000)
