import numpy as np
from numpy.testing import assert_allclose

from .helpers import SeededTest
import pymc3 as pm
from pymc3 import Dirichlet, Gamma, Normal, Lognormal, Poisson, Exponential, \
    Mixture, NormalMixture, MvNormal, sample, Metropolis, Model
import scipy.stats as st
from scipy.special import logsumexp
from pymc3.theanof import floatX
import theano

# Generate data
def generate_normal_mixture_data(w, mu, sd, size=1000):
    component = np.random.choice(w.size, size=size, p=w)
    x = np.random.normal(mu[component], sd[component], size=size)

    return x


def generate_poisson_mixture_data(w, mu, size=1000):
    component = np.random.choice(w.size, size=size, p=w)
    x = np.random.poisson(mu[component], size=size)

    return x


class TestMixture(SeededTest):
    @classmethod
    def setup_class(cls):
        super().setup_class()

        cls.norm_w = np.array([0.75, 0.25])
        cls.norm_mu = np.array([0., 5.])
        cls.norm_sd = np.ones_like(cls.norm_mu)
        cls.norm_x = generate_normal_mixture_data(cls.norm_w, cls.norm_mu, cls.norm_sd, size=1000)

        cls.pois_w = np.array([0.4, 0.6])
        cls.pois_mu = np.array([5., 20.])
        cls.pois_x = generate_poisson_mixture_data(cls.pois_w, cls.pois_mu, size=1000)

    def test_mixture_list_of_normals(self):
        with Model() as model:
            w = Dirichlet('w', floatX(np.ones_like(self.norm_w)))
            mu = Normal('mu', 0., 10., shape=self.norm_w.size)
            tau = Gamma('tau', 1., 1., shape=self.norm_w.size)
            Mixture('x_obs', w,
                    [Normal.dist(mu[0], tau=tau[0]), Normal.dist(mu[1], tau=tau[1])],
                    observed=self.norm_x)
            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed,
                           progressbar=False, chains=1)

        assert_allclose(np.sort(trace['w'].mean(axis=0)),
                        np.sort(self.norm_w),
                        rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace['mu'].mean(axis=0)),
                        np.sort(self.norm_mu),
                        rtol=0.1, atol=0.1)

    def test_normal_mixture(self):
        with Model() as model:
            w = Dirichlet('w', floatX(np.ones_like(self.norm_w)))
            mu = Normal('mu', 0., 10., shape=self.norm_w.size)
            tau = Gamma('tau', 1., 1., shape=self.norm_w.size)
            NormalMixture('x_obs', w, mu, tau=tau, observed=self.norm_x)
            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed,
                           progressbar=False, chains=1)

        assert_allclose(np.sort(trace['w'].mean(axis=0)),
                        np.sort(self.norm_w),
                        rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace['mu'].mean(axis=0)),
                        np.sort(self.norm_mu),
                        rtol=0.1, atol=0.1)

    def test_normal_mixture_nd(self):
        nd, ncomp = 3, 5

        with Model() as model0:
            mus = Normal('mus', shape=(nd, ncomp))
            taus = Gamma('taus', alpha=1, beta=1, shape=(nd, ncomp))
            ws = Dirichlet('ws', np.ones(ncomp))
            mixture0 = NormalMixture('m', w=ws, mu=mus, tau=taus, shape=nd)

        with Model() as model1:
            mus = Normal('mus', shape=(nd, ncomp))
            taus = Gamma('taus', alpha=1, beta=1, shape=(nd, ncomp))
            ws = Dirichlet('ws', np.ones(ncomp))
            comp_dist = [Normal.dist(mu=mus[:, i], tau=taus[:, i])
                         for i in range(ncomp)]
            mixture1 = Mixture('m', w=ws, comp_dists=comp_dist, shape=nd)

        testpoint = model0.test_point
        testpoint['mus'] = np.random.randn(nd, ncomp)
        assert_allclose(model0.logp(testpoint), model1.logp(testpoint))
        assert_allclose(mixture0.logp(testpoint), mixture1.logp(testpoint))

    def test_poisson_mixture(self):
        with Model() as model:
            w = Dirichlet('w', floatX(np.ones_like(self.pois_w)))
            mu = Gamma('mu', 1., 1., shape=self.pois_w.size)
            Mixture('x_obs', w, Poisson.dist(mu), observed=self.pois_x)
            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed,
                           progressbar=False, chains=1)

        assert_allclose(np.sort(trace['w'].mean(axis=0)),
                        np.sort(self.pois_w),
                        rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace['mu'].mean(axis=0)),
                        np.sort(self.pois_mu),
                        rtol=0.1, atol=0.1)

    def test_mixture_list_of_poissons(self):
        with Model() as model:
            w = Dirichlet('w', floatX(np.ones_like(self.pois_w)))
            mu = Gamma('mu', 1., 1., shape=self.pois_w.size)
            Mixture('x_obs', w,
                    [Poisson.dist(mu[0]), Poisson.dist(mu[1])],
                    observed=self.pois_x)
            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed,
                           progressbar=False, chains=1)

        assert_allclose(np.sort(trace['w'].mean(axis=0)),
                        np.sort(self.pois_w),
                        rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace['mu'].mean(axis=0)),
                        np.sort(self.pois_mu),
                        rtol=0.1, atol=0.1)

    def test_mixture_of_mvn(self):
        mu1 = np.asarray([0., 1.])
        cov1 = np.diag([1.5, 2.5])
        mu2 = np.asarray([1., 0.])
        cov2 = np.diag([2.5, 3.5])
        obs = np.asarray([[.5, .5], mu1, mu2])
        with Model() as model:
            w = Dirichlet('w', floatX(np.ones(2)), transform=None)
            mvncomp1 = MvNormal.dist(mu=mu1, cov=cov1)
            mvncomp2 = MvNormal.dist(mu=mu2, cov=cov2)
            y = Mixture('x_obs', w, [mvncomp1, mvncomp2],
                    observed=obs)

        # check logp of each component
        complogp_st = np.vstack((st.multivariate_normal.logpdf(obs, mu1, cov1),
                                 st.multivariate_normal.logpdf(obs, mu2, cov2))
                                ).T
        complogp = y.distribution._comp_logp(theano.shared(obs)).eval()
        assert_allclose(complogp, complogp_st)

        # check logp of mixture
        testpoint = model.test_point
        mixlogp_st = logsumexp(np.log(testpoint['w']) + complogp_st,
                               axis=-1, keepdims=True)
        assert_allclose(y.logp_elemwise(testpoint),
                        mixlogp_st)

        # check logp of model
        priorlogp = st.dirichlet.logpdf(x=testpoint['w'],
                                        alpha=np.ones(2),
                                        )
        assert_allclose(model.logp(testpoint),
                        mixlogp_st.sum() + priorlogp)

    def test_mixture_of_mixture(self):
        nbr = 4
        with Model() as model:
            # mixtures components
            g_comp = Normal.dist(
                mu=Exponential('mu_g', lam=1.0, shape=nbr, transform=None),
                sigma=1,
                shape=nbr)
            l_comp = Lognormal.dist(
                mu=Exponential('mu_l', lam=1.0, shape=nbr, transform=None),
                sigma=1,
                shape=nbr)
            # weight vector for the mixtures
            g_w = Dirichlet('g_w', a=floatX(np.ones(nbr)*0.0000001), transform=None)
            l_w = Dirichlet('l_w', a=floatX(np.ones(nbr)*0.0000001), transform=None)
            # mixture components
            g_mix = Mixture.dist(w=g_w, comp_dists=g_comp)
            l_mix = Mixture.dist(w=l_w, comp_dists=l_comp)
            # mixture of mixtures
            mix_w = Dirichlet('mix_w', a=floatX(np.ones(2)), transform=None)
            mix = Mixture('mix', w=mix_w,
                          comp_dists=[g_mix, l_mix],
                          observed=np.exp(self.norm_x))

        test_point = model.test_point

        def mixmixlogp(value, point):
            priorlogp = st.dirichlet.logpdf(x=point['g_w'],
                                            alpha=np.ones(nbr)*0.0000001,
                                            ) + \
                        st.expon.logpdf(x=point['mu_g']).sum() + \
                        st.dirichlet.logpdf(x=point['l_w'],
                                            alpha=np.ones(nbr)*0.0000001,
                                            ) + \
                        st.expon.logpdf(x=point['mu_l']).sum() + \
                        st.dirichlet.logpdf(x=point['mix_w'],
                                            alpha=np.ones(2),
                                            )
            complogp1 = st.norm.logpdf(x=value,
                                       loc=point['mu_g'])
            mixlogp1 = logsumexp(np.log(point['g_w']) + complogp1,
                                 axis=-1, keepdims=True)
            complogp2 = st.lognorm.logpdf(value, 1., 0., np.exp(point['mu_l']))
            mixlogp2 = logsumexp(np.log(point['l_w']) + complogp2,
                                 axis=-1, keepdims=True)
            complogp_mix = np.concatenate((mixlogp1, mixlogp2), axis=1)
            mixmixlogpg = logsumexp(np.log(point['mix_w']) + complogp_mix,
                                    axis=-1, keepdims=True)
            return priorlogp, mixmixlogpg

        value = np.exp(self.norm_x)[:, None]
        priorlogp, mixmixlogpg = mixmixlogp(value, test_point)

        # check logp of mixture
        assert_allclose(mixmixlogpg, mix.logp_elemwise(test_point))

        # check model logp
        assert_allclose(priorlogp + mixmixlogpg.sum(),
                        model.logp(test_point))

        # check input and check logp again
        test_point['g_w'] = np.asarray([.1, .1, .2, .6])
        test_point['mu_g'] = np.exp(np.random.randn(nbr))
        priorlogp, mixmixlogpg = mixmixlogp(value, test_point)
        assert_allclose(mixmixlogpg, mix.logp_elemwise(test_point))
        assert_allclose(priorlogp + mixmixlogpg.sum(),
                        model.logp(test_point))

    def test_sample_prior_and_posterior(self):
        def build_toy_dataset(N, K):
            pi = np.array([0.2, 0.5, 0.3])
            mus = [[1, 1, 1], [-1, -1, -1], [2, -2, 0]]
            stds = [[0.1, 0.1, 0.1], [0.1, 0.2, 0.2], [0.2, 0.3, 0.3]]
            x = np.zeros((N, 3), dtype=np.float32)
            y = np.zeros((N,), dtype=np.int)
            for n in range(N):
                k = np.argmax(np.random.multinomial(1, pi))
                x[n, :] = np.random.multivariate_normal(mus[k],
                                                        np.diag(stds[k]))
                y[n] = k
            return x, y

        N = 100  # number of data points
        K = 3  # number of mixture components
        D = 3  # dimensionality of the data

        X, y = build_toy_dataset(N, K)

        with pm.Model() as model:
            pi = pm.Dirichlet('pi', np.ones(K))

            comp_dist = []
            mu = []
            packed_chol = []
            chol = []
            for i in range(K):
                mu.append(pm.Normal('mu%i' % i, 0, 10, shape=D))
                packed_chol.append(
                    pm.LKJCholeskyCov('chol_cov_%i' % i,
                                      eta=2,
                                      n=D,
                                      sd_dist=pm.HalfNormal.dist(2.5))
                )
                chol.append(pm.expand_packed_triangular(D, packed_chol[i],
                                                        lower=True))
                comp_dist.append(pm.MvNormal.dist(mu=mu[i], chol=chol[i]))

            pm.Mixture('x_obs', pi, comp_dist, observed=X)
        with model:
            trace = pm.sample(30, tune=10, chains=1)

        n_samples = 20
        with model:
            ppc = pm.sample_posterior_predictive(trace, n_samples)
            prior = pm.sample_prior_predictive(samples=n_samples)
        assert ppc['x_obs'].shape == (n_samples,) + X.shape
        assert prior['x_obs'].shape == (n_samples,) + X.shape
        assert prior['mu0'].shape == (n_samples, D)
        assert prior['chol_cov_0'].shape == (n_samples, D * (D + 1) // 2)
