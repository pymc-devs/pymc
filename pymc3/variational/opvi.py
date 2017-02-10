import numpy as np
import theano
from theano import tensor as tt

import pymc3 as pm
from pymc3.distributions.dist_math import log_normal, rho2sd, log_normal_mv
from pymc3.theanof import identity
from pymc3.variational.base import Operator, Approximation, TestFunction


# OPERATORS

class KL(Operator):
    def apply(self, f):
        """
        KL divergence between posterior and approximation for input `z`
            :math:`z ~ Approximation`
        """
        z = self.input
        return self.logq(z) - self.logp(z)


class LS(Operator):
    def apply(self, f):
        z = self.input
        logp = self.logp
        jacobian = theano.gradient.jacobian
        trace = tt.nlinalg.trace
        return (tt.Rop(logp(z), z, f(z)) +
                trace(jacobian(f(z), z)))


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
                    np.eye(self.global_size, dtype=theano.config.floatX).ravel())
                }

    def log_q_W_global(self, z):
        """
        log_q_W samples over q for global vars
        Gradient wrt mu, rho in density parametrization
        is set to zero to lower variance of ELBO
        """
        mu = self.shared_params['mu']
        L = self.shared_params['L'].reshape((self.global_size, self.global_size))
        mu = self.scale_grad(mu)
        L = self.scale_grad(L)
        return log_normal_mv(z, mu, chol=L)

    def random_global(self, samples=None, no_rand=False):
        initial = self.initial(samples, no_rand, l=self.global_size)
        L = self.shared_params['L'].reshape((self.global_size, self.global_size))
        mu = self.shared_params['mu']
        return initial.dot(L) + mu


class NeuralNetwork(Approximation):
    initial_dist_name = 'uniform'
    initial_dist_map = 0.5

    def __init__(self, local_rv=None, model=None, layers=1, hidden_size=None, activations=tt.nnet.relu):
        dim = sum(v.dsize for v in pm.modelcontext(model).vars)
        if hidden_size is not None:
            layers = len(hidden_size) + 1
        else:
            hidden_size = (dim, ) * (layers - 1)
        self.layers = layers
        dd = [dim] + list(hidden_size) + [dim]
        self.shapes = list(zip(dd[:-1], dd[1:]))
        if not isinstance(activations, (list, tuple)):
            activations = [activations] * layers
            activations[-1] = identity
        self.activations = activations
        super(NeuralNetwork, self).__init__(local_rv, model)

    def create_shared_params(self):
        weights = [theano.shared(
            np.random.normal(size=shape).astype(dtype=theano.config.floatX).ravel()
        )
                   for shape in self.shapes]
        bias = [theano.shared(
            np.random.normal(size=(shape[1], )).astype(dtype=theano.config.floatX).ravel()
        )
                for shape in self.shapes]
        return weights + bias

    def random(self, size=None, no_rand=False):
        z = self.initial(size, no_rand, self.global_size)
        weights = self.shared_params[:self.layers]
        weights = [w.reshape(shape) for w, shape in zip(weights, self.shapes)]
        bias = self.shared_params[self.layers:]
        activations = self.activations
        for w, b, a in zip(weights, bias, activations):
            z = a(z.dot(w) + b)
        return z


# TEST FUNCTIONS

class TestNeuralNetwork(TestFunction):
    def __init__(self, dim, layers=2, hidden_size=None, activations=tt.tanh):
        if hidden_size is not None:
            layers = len(hidden_size) + 1
        else:
            hidden_size = (dim, ) * (layers - 1)
        self.layers = layers
        dd = [dim] + list(hidden_size) + [dim]
        self.shapes = list(zip(dd[:-1], dd[1:]))
        if not isinstance(activations, (list, tuple)):
            activations = [activations] * layers
            activations[-1] = tt.tanh
        self.activations = activations
        super(TestNeuralNetwork, self).__init__(dim=dim)

    def create_shared_params(self):
        weights = [theano.shared(
            np.random.normal(size=shape).astype(dtype=theano.config.floatX).ravel()
        )
                   for shape in self.shapes]
        bias = [theano.shared(
            np.random.normal(size=(shape[1], )).astype(dtype=theano.config.floatX).ravel()
        )
                for shape in self.shapes]
        return weights + bias

    def __call__(self, z):
        weights = self.shared_params[:self.layers]
        weights = [w.reshape(shape) for w, shape in zip(weights, self.shapes)]
        bias = self.shared_params[self.layers:]
        activations = self.activations
        for w, b, a in zip(weights, bias, activations):
            z = a(z.dot(w) + b)
        return z