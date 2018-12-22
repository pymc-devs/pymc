import pymc3 as pm
import numpy as np
import theano.tensor as tt

from .helpers import SeededTest


class TestSMC(SeededTest):

    def setup_class(self):
        super().setup_class()
        self.samples = 1000
        n = 4
        mu1 = np.ones(n) * (1. / 2)
        mu2 = - mu1

        stdev = 0.1
        sigma = np.power(stdev, 2) * np.eye(n)
        isigma = np.linalg.inv(sigma)
        dsigma = np.linalg.det(sigma)

        w1 = stdev
        w2 = (1 - stdev)

        def two_gaussians(x):
            log_like1 = - 0.5 * n * tt.log(2 * np.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu1).T.dot(isigma).dot(x - mu1)
            log_like2 = - 0.5 * n * tt.log(2 * np.pi) \
                        - 0.5 * tt.log(dsigma) \
                        - 0.5 * (x - mu2).T.dot(isigma).dot(x - mu2)
            return tt.log(w1 * tt.exp(log_like1) + w2 * tt.exp(log_like2))

        with pm.Model() as self.SMC_test:
            X = pm.Uniform('X', lower=-2, upper=2., shape=n)
            llk = pm.Potential('muh', two_gaussians(X))

        self.muref = mu1


    def test_sample(self):
        with self.SMC_test:
            mtrace = pm.sample(draws=self.samples,
                               step = pm.SMC())

        x = mtrace['X']
        mu1d = np.abs(x).mean(axis=0)
        np.testing.assert_allclose(self.muref, mu1d, rtol=0., atol=0.03)


    def test_discrete_continuous(self):
        with pm.Model() as model:
            a = pm.Poisson('a', 5)
            b = pm.HalfNormal('b', 10)
            y = pm.Normal('y', a, b, observed=[1, 2, 3, 4])
            trace = pm.sample(step=pm.SMC())


    def test_ml(self):
        data = np.repeat([1, 0], [50, 50])
        marginals = []
        a_prior_0, b_prior_0 = 1., 1.
        a_prior_1, b_prior_1 = 20., 20.

        for alpha, beta in ((a_prior_0, b_prior_0), (a_prior_1, b_prior_1)):
            with pm.Model() as model:
                a = pm.Beta('a', alpha, beta)
                y = pm.Bernoulli('y', a, observed=data)
                trace = pm.sample(2000, step=pm.SMC())
                marginals.append(model.marginal_likelihood)
        # compare to the analytical result
        assert abs((marginals[1] / marginals[0]) - 4.0) <= 1


