import numpy as np
import theano
from theano import tensor as tt

from pymc3.distributions.dist_math import log_normal, rho2sd, log_normal_mv
from pymc3.variational.base import Operator, Approximation, TestFunction


__all__ = [
    'TestFunction',
    'KL',
    'MeanField',
    'FullRank'
]
# OPERATORS


class KL(Operator):
    def apply(self, f):
        """
        KL divergence between posterior and approximation for input `z`
            :math:`z ~ Approximation`
        """
        z = self.input
        return self.logq(z) - self.logp(z)

# APPROXIMATIONS


class MeanField(Approximation):
    def create_shared_params(self):
        return {'mu': theano.shared(
                    self.input.tag.test_value[self.global_slc]),
                'rho': theano.shared(
                    np.zeros((self.global_size,), dtype=theano.config.floatX))
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.shared_params['mu']
        rho = self.shared_params['rho']
        mu = self.scale_grad(mu)
        rho = self.scale_grad(rho)
        logq = tt.sum(log_normal(z[self.global_slc], mu, rho=rho))
        return logq

    def random_global(self, samples=None, no_rand=False):
        initial = self.initial(samples, no_rand, l=self.global_size)
        sd = rho2sd(self.shared_params['rho'])
        mu = self.shared_params['mu']
        return sd * initial + mu


class FullRank(Approximation):
    def create_shared_params(self):
        return {'mu': theano.shared(
                    self.input.tag.test_value[self.global_slc]),
                'L': theano.shared(
                    np.eye(self.global_size, dtype=theano.config.floatX))
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.shared_params['mu']
        L = self.shared_params['L']
        mu = self.scale_grad(mu)
        L = self.scale_grad(L)
        return log_normal_mv(z, mu, chol=L)

    def random_global(self, samples=None, no_rand=False):
        initial = self.initial(samples, no_rand, l=self.global_size)
        L = self.shared_params['L']
        mu = self.shared_params['mu']
        return initial.dot(L) + mu
