'''
Created on Aug 29, 2014

@author: Heiko Strathmann
'''

from scipy.spatial.distance import squareform, pdist

from theano import function
import theano
from theano.configparser import TheanoConfigParser

import numpy as np
from pymc.model import modelcontext
from pymc.step_methods.arraystep import ArrayStep, metrop_select
from pymc.step_methods.metropolis import tune
import scipy as sp
import theano.tensor as T

from ..core import *


# To avoid Theano complaining about missing test values
TheanoConfigParser().compute_test_value = 'off'


__all__ = ['KameleonOracle']


class KameleonOracle(ArrayStep):
    """
    Kernel Adaptive Metropolis-Hastings sampling step with a fixed set of orcale
    samples. Automatic tuning of the scaling is possible via a simple schedule
    for a given number of iterations.
    
    Based on "Kernel Adaptive Metopolis Hastings" by D. Sejdinovic, H. Strathmann,
    M. Lomeli, C. Andrieu, A. Gretton
    http://jmlr.org/proceedings/papers/v32/sejdinovic14.html
    
    See also https://github.com/karlnapf/kameleon-mcmc for experimental code.

    Parameters
    ----------
    vars : list
        List of variables for sampler
    Z : 2d numpy array
        Oracle sample to represent target covariance structure
    proposal_dist : function
        Function that returns zero-mean deviates when parameterized with
        S (and n). Defaults to quad_potential.
    gamma2 : scalar
        Exploration term in proposal
    nu2 : scalar
        Scaling of the covariance part of the proposal
    kernel : Kernel instance
        Kernel to use for representing covariance structure in feature space.
        Must implement interface for kernel and gradient, see for example GaussianKernel
    model : PyMC Model
        Optional model for sampling step. Defaults to None (taken from context).

    """
    def __init__(self, vars=None, Z=None, gamma2=0.1, nu2=1., kernel=None,
                 tune=True, tune_interval=100, tune_stop=1000, model=None, dist=None):
        model = modelcontext(model)
        if vars is None:
            vars = model.vars
            
        self.Z = Z
        self.kernel = kernel
        self.gamma2 = gamma2
        self.nu2 = nu2
        self.tune = tune
        self.tune_stop = tune_stop
        
        # empty proposal distribution and last likelihood
        self.q_dist = None
        self.log_target = -np.inf
        
        # statistics for tuning scaling
        self.iterations = 0
        self.steps_since_tune = 0
        self.accepted = 0
        self.tune = tune
        self.tune_interval = tune_interval
        
        super(KameleonOracle, self).__init__(vars, [model.fastlogp])

    def astep(self, q0, logp):
        # sample from kernel based Gaussian proposal
        q_dist = self.construct_proposal(q0)
        q = np.ravel(q_dist.sample())
            
        # evaluate target log probability
        logp_q = logp(q)
            
        # MH accept/reject step
        if self.q_dist is None:
            q_new = q
        else:
            q_new = metrop_select(logp_q + q_dist.log_pdf(q0) \
                                  - self.log_pdf_target - self.q_dist.log_pdf(q), q, q0)
        
        # adapt
        if self.iterations <= self.tune_stop and \
           self.tune \
           and self.steps_since_tune == self.tune_interval:
            # tune scaling parameter using metropolis  method
            self.nu2 = tune(self.nu2, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_since_tune = 0
            self.accepted = 0
        self.steps_since_tune += 1
        self.iterations += 1
        
        # update log-pdf and proposal distribution object on accept
        if any(q_new != q0):
            self.q_dist = q_dist
            self.log_pdf_target = logp_q
            self.accepted += 1

        return q_new

    def compute_constants(self, y):
        """
        Pre-computes constants of the log density of the proposal distribution,
        which is Gaussian as p(x|y) ~ N(mu, R)
        where
        mu = y-a
        a = 0
        R  = gamma^2 I + M M^T
        M  = 2 [\nabla_x k(x,z_i]|_x=y
        
        Returns (mu,L_R), where L_R is lower Cholesky factor of R
        """
        assert(len(np.shape(y)) == 1)
        
        # M = 2 [\nabla_x k(x,z_i]|_x=y
        R = self.gamma2 * np.eye(len(y))
        if self.Z is not None:
            M = 2 * self.kernel.gradient(y, self.Z)
            # R = gamma^2 I + \nu^2 * M H M^T
            H = np.eye(len(self.Z)) - 1.0 / len(self.Z)
            R += self.nu2 * M.T.dot(H.dot(M))
            
        L_R = np.linalg.cholesky(R)
        
        return y.copy(), L_R
    
    def construct_proposal(self, y):
        """
        Constructs the Kameleon MCMC proposal centred at y, using history Z
        
        The proposal is a Gaussian based on the kernel values between y and all
        points in the chain history.
        """
        mu, L = self.compute_constants(y)
        
        return Gaussian(mu, L, is_cholesky=True)

class Gaussian():
    """
    Helper class to sample from and evaluate log-pdf of a multivariate Gaussian,
    using efficient Cholesky based representation (Cholesky only computed once)
    """
    def __init__(self, mu, Sigma, is_cholesky=False):
        self.mu = mu
        self.is_cholesky = is_cholesky
        
        if self.is_cholesky:
            self.L = Sigma
        else:
            self.L = np.linalg.cholesky(Sigma)
            
        self.dimension = len(mu)
    
    def log_pdf(self, X):
        # duck typing for shape
        if len(np.shape(X)) == 1:
            X = X.reshape(1, len(X))
        
        log_determinant_part = -sum(np.log(np.diag(self.L)))
        
        quadratic_parts = np.zeros(len(X))
        for i in range(len(X)):
            x = X[i] - self.mu
            
            # solve y=K^(-1)x = L^(-T)L^(-1)x
            y = sp.linalg.solve_triangular(self.L, x.T, lower=True)
            y = sp.linalg.solve_triangular(self.L.T, y, lower=False)
            quadratic_parts[i] = -0.5 * x.dot(y)
            
        const_part = -0.5 * len(self.L) * np.log(2 * np.pi)
        
        return const_part + log_determinant_part + quadratic_parts
    
    def sample(self, n=1):
        V = np.random.randn(self.dimension, n)

        # map to our desired Gaussian and transpose to have row-wise vectors
        return self.L.dot(V).T + self.mu

def theano_sq_dists_mat_expr(X, Y):
    return (-2 * X.dot(Y.T).T + T.sum(X ** 2, 1).T).T + T.sum(Y ** 2, 1)

def theano_gaussian_kernel_expr(sq_dists, sigma):
    return T.exp(-sq_dists / (2.*sigma ** 2))

def theano_sq_dists_vec_expr(x, Y):
        # element wise vector norm
        Y_norm = T.sum(Y ** 2, 1)
        xY_terms = x.T.dot(Y.T)
    
        # expanded sq euclidean distance
        return T.sum(x ** 2) + Y_norm - 2 * xY_terms

class GaussianKernel():
    """
    Helper class to represent a Gaussian kernel, with methods to compute kernel
    function and its gradient wrt the left argument.
    Uses Theano's autodiff for computing kernel gradients.
    """
    
    # compile theano functions
    X = T.dmatrix('X')
    x = T.dvector('x')
    Y = T.dmatrix('Y')
    sigma = T.dscalar('sigma')
    
    # kernel expressions as for left input being matrix or vector
    sq_dist_mat_expr = theano_sq_dists_mat_expr(X, Y)
    sq_dist_vec_expr = theano_sq_dists_vec_expr(x, Y)
    K_expr = theano_gaussian_kernel_expr(sq_dist_mat_expr, sigma)
    k_expr = theano_gaussian_kernel_expr(sq_dist_vec_expr, sigma)
    
    # compile
    theano_kernel_mat = function(inputs=[X, Y, sigma], outputs=K_expr)
    theano_kernel_vec = function(inputs=[x, Y, sigma], outputs=k_expr)
    theano_kernel_vec_grad_x = function(inputs=[x, Y, sigma],
                                             outputs=theano.gradient.jacobian(k_expr, x))
    
    @staticmethod
    def gaussian_median_heuristic(X):
        dists = squareform(pdist(X, 'sqeuclidean'))
        median_dist = np.median(dists[dists > 0])
        sigma = np.sqrt(0.5 * median_dist)
        return sigma
    
    def __init__(self, width):
        self.width = width
        
    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)
        
        X - 2d array, samples on right hand side
        Y - 2d array, samples on left hand side, can be None in which case its replaced by X
        """
        return self.theano_kernel(X, Y, self.width)
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the Gaussian kernel wrt. to the left argument, i.e.
        k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2), which is
        \nabla_x k(x,y)=1.0/sigma**2 k(x,y)(y-x)
        Given a set of row vectors Y, this computes the
        gradient for every pair (x,y) for y in Y.
        
        x - single sample on right hand side (1d array)
        Y - samples on left hand side (2d array)
        """
        return self.theano_kernel_vec_grad_x(x, Y, self.width)