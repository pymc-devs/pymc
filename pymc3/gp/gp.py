import theano
import theano.tensor as tt
import numpy as np

from scipy import stats
import theano.tensor.slinalg
from theano.tensor.nlinalg import det, matrix_inverse, trace

from ..distributions.distribution import Continuous, draw_values, generate_samples
from ..model import Deterministic

__all__ = ['GP']


class GP(Continuous):
    def __init__(self, mu, cov=None, *args, **kwargs):
        super(GP, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.cov = cov

    def random(self, point=None, size=None):
        mu, cov = draw_values([self.mu, self.cov], point=point)

        def _random(mean, cov, size=None):
            return stats.multivariate_normal.rvs(
                mean, cov, None if size == mean.shape else size)

        samples = generate_samples(_random,
                                   mean=mu, cov=cov,
                                   dist_shape=self.shape,
                                   broadcast_shape=mu.shape,
                                   size=size)
        return samples

    def logp(self, value):
        mu = self.mu
        cov = self.cov

        diff = value - mu
        n = cov.shape[0]
        L = tt.slinalg.cholesky(cov + 1e-8 * tt.eye(n))
        tmp = tt.slinalg.solve_lower_triangular(L, diff)
        logdet = 2.0 * tt.sum(tt.log(tt.diag(L)))
        return -0.5 * (tt.dot(tmp.T, tmp) + logdet + n*tt.log(2*np.pi))

