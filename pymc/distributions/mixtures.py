

from pymc import *

from numpy.random import normal
import numpy as np
import pylab as pl
from itertools import product
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from numpy import pi
from numpy.random import uniform, multivariate_normal
from pymc.model import FreeRV, ObservedRV
import numpy as np
import numpy.linalg as np_linalg
from pymc.step_methods.metropolis_hastings import GenericProposal

"""
Multivariate Mixture distributions. 

These distributions have a lot of potential uses. For one thing: They can be used to approximate every other density function,
and there are a lot of algorithms for efficiently creating density estimates (i.e. Kernel Density Estimates (KDE),
Gaussian Mixture Model learning and others.

Once we have learned such a density, we could use it as a prior density. To do that, we need to be able to
efficiently sample from it. Luckily, that's simple and efficient. Unless you use the wrong proposal distribution, of course.

That's where these classes come in handy.

@author Kai Londenberg ( Kai.Londenberg@gmail.com )    

"""
class MvGaussianMixture(Continuous):
    
    def __init__(self, shape, cat_var, mus, taus, model=None, *args, **kwargs):
        '''
        Creates a continuous mixture distribution which can be efficiently evaluated and sampled from.
         
        Args:
            shape: Shape of the distribution. All kernels need to have this same shape as well.
            weights: weight vector (may be a theano shared variable or expression. Must be able to evaluate this in the context of the model).
            kernels: list of mixture component distributions. These have to be MixtureKernel instances (for example MvNormalKernel ) of same shape
            testval: Test value for logp computations
        '''
        super(MvGaussianMixture, self).__init__(*args, shape=shape, **kwargs)
        assert isinstance(cat_var, FreeRV)
        assert isinstance(cat_var.distribution, Categorical)
        self.cat_var = cat_var
        model = modelcontext(model)
        weights = cat_var.distribution.p
        self.weights = weights
        self.mus = mus
        self.taus = taus
        self.mu_t = T.stacklists(mus)
        self.tau_t = T.stacklists(taus)
        self.shape = shape
        self.testval = np.zeros(self.shape, self.dtype)
        self.param_fn = model.fastfn([self.mu_t[self.cat_var], self.tau_t[self.cat_var]])
        self.last_cov_value = None
        self.last_tau_value = None
        
    def logp(self, value):
        mu = self.mu_t[self.cat_var]
        tau = self.tau_t[self.cat_var]

        delta = value - mu
        k = tau.shape[0]

        return 1/2. * (-k * log(2*pi) + log(det(tau)) - dot(delta.T, dot(tau, delta)))
    
    def draw_mvn(self, mu, tau):
        if (tau==self.last_tau_value):
            mcov = self.last_cov_value
        else:
            mcov = np_linalg.inv(tau)
            self.last_cov_value = mcov
        
        return np.random.multivariate_normal(mean=mu, mcov=tau)
    
    def draw(self, point):
        mu, tau = self.param_fn(point)
        if (tau==self.last_tau_value):
            mcov = self.last_cov_value
        else:
            mcov = np_linalg.inv(tau)
            self.last_cov_value = mcov
        return self.draw_mvn
    

class GMMProposal(GenericProposal):
    
    def __init__(self, gmm_var, model=None):
        assert isinstance(gmm_var.distribution, MvGaussianMixture)
        model = modelcontext(model)
        self.gmm_var = gmm_var
        self.logpfunc = model.fastfn(gmm_var.distribution.logp)
        self._proposal_logp_difference = 0.0
        
    def propose_move(self, point):
        old_logp = self.logpfunc(point)
        point[self.gmm_var.name] = self.gmm_var.draw(point)
        new_logp = self.logpfunc(point)
        self._proposal_logp_difference = old_logp-new_logp
        return point

    def proposal_logp_difference(self):
        return self._proposal_logp_difference

