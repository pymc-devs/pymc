import numpy as np
from numpy.testing import assert_allclose

from .helpers import SeededTest
from pymc3 import Dirichlet, Gamma, Metropolis, Mixture, Model, Normal, NormalMixture, Poisson, sample


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
    def setUpClass(cls):
        super(TestMixture, cls).setUpClass()

        cls.norm_w = np.array([0.75, 0.25])
        cls.norm_mu = np.array([0., 5.])
        cls.norm_sd = np.ones_like(cls.norm_mu)
        cls.norm_x = generate_normal_mixture_data(cls.norm_w, cls.norm_mu, cls.norm_sd, size=1000)

        cls.pois_w = np.array([0.4, 0.6])
        cls.pois_mu = np.array([5., 20.])
        cls.pois_x = generate_poisson_mixture_data(cls.pois_w, cls.pois_mu, size=1000)

    def test_mixture_list_of_normals(self):
        with Model() as model:
            w = Dirichlet('w', np.ones_like(self.norm_w))

            mu = Normal('mu', 0., 10., shape=self.norm_w.size)
            tau = Gamma('tau', 1., 1., shape=self.norm_w.size)

            x_obs = Mixture('x_obs', w,
                            [Normal.dist(mu[0], tau=tau[0]),
                             Normal.dist(mu[1], tau=tau[1])],
                            observed=self.norm_x)

            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed, progressbar=False)

        assert_allclose(np.sort(trace['w'].mean(axis=0)),
                        np.sort(self.norm_w),
                        rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace['mu'].mean(axis=0)),
                        np.sort(self.norm_mu),
                        rtol=0.1, atol=0.1)

    def test_normal_mixture(self):
        with Model() as model:
            w = Dirichlet('w', np.ones_like(self.norm_w))

            mu = Normal('mu', 0., 10., shape=self.norm_w.size)
            tau = Gamma('tau', 1., 1., shape=self.norm_w.size)

            x_obs = NormalMixture('x_obs', w, mu, tau=tau, observed=self.norm_x)

            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed, progressbar=False)

        assert_allclose(np.sort(trace['w'].mean(axis=0)),
                        np.sort(self.norm_w),
                        rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace['mu'].mean(axis=0)),
                        np.sort(self.norm_mu),
                        rtol=0.1, atol=0.1)

    def test_poisson_mixture(self):
        with Model() as model:
            w = Dirichlet('w', np.ones_like(self.pois_w))

            mu = Gamma('mu', 1., 1., shape=self.pois_w.size)

            x_obs = Mixture('x_obs', w, Poisson.dist(mu), observed=self.pois_x)

            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed, progressbar=False)

        assert_allclose(np.sort(trace['w'].mean(axis=0)),
                        np.sort(self.pois_w),
                        rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace['mu'].mean(axis=0)),
                        np.sort(self.pois_mu),
                        rtol=0.1, atol=0.1)

    def test_mixture_list_of_poissons(self):
        with Model() as model:
            w = Dirichlet('w', np.ones_like(self.pois_w))

            mu = Gamma('mu', 1., 1., shape=self.pois_w.size)

            x_obs = Mixture('x_obs', w,
                            [Poisson.dist(mu[0]), Poisson.dist(mu[1])],
                            observed=self.pois_x)

            step = Metropolis()
            trace = sample(5000, step, random_seed=self.random_seed, progressbar=False)

        assert_allclose(np.sort(trace['w'].mean(axis=0)),
                        np.sort(self.pois_w),
                        rtol=0.1, atol=0.1)
        assert_allclose(np.sort(trace['mu'].mean(axis=0)),
                        np.sort(self.pois_mu),
                        rtol=0.1, atol=0.1)
