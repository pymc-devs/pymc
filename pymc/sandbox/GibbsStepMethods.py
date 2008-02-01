"""
Gibbs step methods apply to conjugate submodels. In other words, if in the
following model:

B|A ~ d2(A, p2)
A ~ d1(p1)

d1 is a standard distribution and A's distribution conditional on B is d1 
with parameters p1_*, A can be Gibbs sampled using standard random variables.

If, on the other hand, the likelihood p(B|A) regarded as a function of A is 
proportional to a standard distribution d3, but A's distribution conditional
on its parents is /not/ d3, A can be Metropolis sampled using p(B|A) as a 
proposal distribution. In this case the Metropolis-Hastings acceptance 
threshold is

    min(1, p(A_p|parents) / p(A|parents)).
    
Each Gibbs step method has a fully conjugate version and a nonconjugate version.
"""

from pymc import *
import numpy as np


# TODO:
# GammaGamma, maybe, though no one will ever use it.
# GaussianSubmodel

# If we implement Pareto:
# ParetoUniform
# GammaPareto

def safe_len(val):
    if np.isscalar(val):
        return 1
    else:
        return np.prod(np.shape(val))

class Gibbs(Metropolis):
    
    def __init__(self, stochastic, verbose=0):
        Metropolis.__init__(self, stochastic, verbose=verbose)
        
        @dtrm
        def N(d=self.d):
            """The total number of observations."""
            return sum([safe_len(d_now) for d_now in d])
        
        @dtrm
        def sum_d(d=self.d):
            """The sum of the number of 'successes' for each 'experiment'"""
            return sum([sum(d_now) for d_now in d])
        
        self.N_d = len(self.d)
        self.N = N
        self.sum_d = sum_d
        
    # Override Metropolis's competence.
    competence = staticmethod(StepMethod.competence)
    
    def step(self):
        if not self.conjugate:
            logp = self.stochastic.logp

        self.propose()

        if not self.conjugate:

            try:
                logp_p = self.stochastic.logp
            except ZeroProbability:
                self.reject()

            if log(np.random.random()) > logp_p - logp:
                self.reject()
    
    def tune(self, verbose):
        return False
        
    def check_children(self, child_class, parent_key):
        self.d = []
        for name in child_class.parent_names:
            if not name == parent_key:
                setattr(self, name, [])
        for child in self.stochastic.children:
            if not isinstance(child, child_class):
                raise ValueError, 'Stochastic %s must have all %s children for %s\n \
                                    to be able to handle it.' \
                                    %(self.stochastic.__name__, child_class.__name__, self.__class__.__name__)
            self.d.append(child)
            for name in child_class.parent_names:
                if not name == parent_key:
                    getattr(self, name).append(child.parents[name])
            
    def check_conjugacy(self, target_class):
        if not isinstance(self.stochastic, target_class):
            for name in self.stochastic.parents:
                setattr(self, name, None)
            self.conjugate = False
        else:
            for name in self.stochastic.parents:
                setattr(self, name, lam_dtrm(name, lambda parent = self.stochastic.parents[name]: parent))
            self.conjugate = True
            
    def check_linear_extended_children(self):
        pass

class GammaNormal(Gibbs):
    """
    Applies to tau in the following submodel:
    
    d ~ind N(mu, tau * theta)
    tau ~ Gamma(alpha, beta) [optional]
    
    The argument tau must be a Stochastic.    
    """
    def __init__(self, tau, verbose=0):
        
        self.stochastic = tau
        self.mu = []
        self.theta = []
        self.d = []
        
        # Get distributional parameters of children and make sure children are Normal.
        for child in tau.children:
            if isinstance(child, Normal):# or isinstance(child, MvNormal):                
                self.d.append(child)
                self.theta.append(1)
                self.mu.append(child.parents['mu'])
            else:
                # TODO: allow child to be LinearCombination of self.
                raise ValueError, 'Stochastic %s must have all normal children for GammaNormal\n \
                                    to be able to handle it.' %self.stochastic.__name__
        
        # Check for conjugate or nonconjugate case.
        self.check_conjugacy(Gamma)
        Gibbs.__init__(self, tau, verbose)
        
        @dtrm
        def quad_term(d=self.d, mu=self.mu, theta=self.theta):
            """The quadratic term in the likelihood."""
            quad_term = 0
            for i in xrange(self.N_d):
                
                delta_now = d[i] - mu[i]
    
                # if theta is not None:
                #     if not np.isscalar(theta[i]):
                #         quad_term += np.dot(dot(delta_now, theta[i]), delta_now)
                #     else:
                #         quad_term += np.dot(delta_now, delta_now) * theta[i]
                # else:
                quad_term += np.dot(delta_now, delta_now)
            return quad_term*.5
                
        self.quad_term = quad_term

    def propose(self):

        alpha = .5*self.N.value
        beta = self.quad_term.value
        if self.conjugate:
            alpha += self.alpha.value
            beta += self.beta.value
        else:
            alpha += 1.

        self.stochastic.value = np.random.gamma(alpha, 1./beta)


class WishartMvNormal(Gibbs):
    """
    Applies to tau in the following submodel:
    
    d_i ~ind Normal(mu_i, tau)
    tau ~ WishartCov(n, Tau) [optional]
    
    where the stochastics d are parametrized by precision, not covariance.
    """
    def __init__(self, tau, verbose=0):
        
        self.stochastic = tau
        
        # Get distributional parameters from children and make sure children are Normal.
        self.check_children(MvNormal, 'tau')        
        
        # See whether to use conjugate or non-conjugate version.
        self.check_conjugacy(Wishart)        
        Gibbs.__init__(self, tau, verbose)
        
        @dtrm
        def quad_term(d=self.d, mu=self.mu, N = self.stochastic.value.shape[0]):
            """The quadratic term in the likelihood."""
            quad_array = np.asmatrix(np.empty((len(self.d), N)))
            for i in xrange(len(d)):
                quad_array[i,:] = d[i] - mu[i]
            return quad_array.T * quad_array
                        
        self.quad_term = quad_term
        
    def propose(self):
        n = self.N
        if self.conjugate:
            n += self.n.value
            Tau = self.quad_term.value + self.Tau.value
        else:
            n += 1.
            Tau = self.quad_term.value
        
        # print n, Tau, Tau.I*n   
        self.stochastic.value = rwishart(n, Tau)

class GammaPoisson(Gibbs):
    """
    Applies to p in the following submodel:
    
    d_i ~ind Poisson(mu)
    mu ~ Gamma(alpha, beta) [optional]
    
    The argument mu must be a Stochastic.
    """
    def __init__(self, mu, verbose=0):
        
        self.stochastic = mu
        
        # Get distributional parameters of children, etc.
        self.check_children(Poisson, 'mu')
        Gibbs.__init__(self, mu, verbose)
        self.check_conjugacy(Gamma)
                
                    
    def propose(self):
        beta = self.N.value        
        alpha = self.sum_d.value
        if self.conjugate:
            alpha = alpha + self.alpha.value
            beta = beta + self.beta.value
        else:
            alpha += 1.
            beta += 1.
        self.stochastic.value = np.random.gamma(alpha, 1./beta)

class BetaGeometric(Gibbs):
    """
    Applies to p in the following submodel:

    d_i ~ind Geometric(p)
    p ~ Beta(alpha, beta) [optional]

    The argument p must be a Stochastic.
    """
    def __init__(self, p, verbose=0):

        self.stochastic = p

        # Get distributional parameters of children, etc.
        self.check_children(Geometric, 'p')
        Gibbs.__init__(self, p, verbose)
        self.check_conjugacy(Beta)

    def propose(self):
        alpha = self.N.value        
        beta = self.sum_d.value
        if self.conjugate:
            alpha = alpha + self.alpha.value
            beta = beta + self.beta.value
        else:
            alpha += 1.
            beta += 1.
        self.stochastic.value = np.random.beta(alpha, beta)

class GammaExponential(Gibbs):
    """
    Applies to p in the following submodel:
    
    d_i ~ind Exponential(beta)
    beta ~ Gamma(alpha, beta) [optional]
    
    The argument beta must be a Stochastic.
    """
    def __init__(self, beta, verbose=0):
        
        self.stochastic = beta
        self.check_children(Exponential, 'beta')
        Gibbs.__init__(self, beta, verbose)
        self.check_conjugacy(Gamma)
                    
    def propose(self):
        alpha = self.N.value        
        beta = self.sum_d.value
        if self.conjugate:
            alpha = alpha + self.alpha.value
            beta = beta + self.beta.value
        else:
            alpha += 1.
            beta += 1.
        self.stochastic.value = np.random.gamma(alpha, 1./beta)

# TODO: Consider implementing BetaBernoulli and DirichletCategorical also.
class BetaBinomial(Gibbs):
    """
    Applies to p in the following submodel:
    
    d_i ~ind Binomial(n_i, p)
    p ~ Beta(alpha, beta) [optional]
    
    The argument p must be a Stochastic.
    """
    def __init__(self, p, verbose=0):
        
        self.stochastic = p
        
        self.check_children(Binomial, 'p')
        Gibbs.__init__(self, p, verbose)
        self.check_conjugacy(Beta)
        
        @dtrm
        def sum_nmd(sum_d=self.sum_d,n=self.n,d=self.d):
            """The sum of the total number of 'failures' for each 'experiment'"""
            out = -sum_d

            for i in xrange(self.N_d):
                if np.isscalar(n[i]):
                    out += n[i]*safe_len(d[i])
                else:
                    out += sum(n[i])
                    
            return out
            
        
        self.sum_nmd = sum_nmd
                    
    def propose(self):
        alpha = self.sum_d.value
        beta = self.sum_nmd.value
        if self.conjugate:
            alpha = alpha + self.alpha.value
            beta = beta + self.beta.value
        else:
            alpha += 1.
            beta += 1.
        self.stochastic.value = np.random.beta(alpha, beta)

class DirichletMultinomial(Gibbs):            
    """
    Applies to p in the following submodel:
    
    d_i ~ind Multinomial(n_i, p)
    p ~ Dirichlet(theta) [optional] 
    
    p must be a Stochastic, preferably a Dirichlet.   
    """
    def __init__(self, p, verbose=0):
        
        self.stochastic = p
        
        # Get distributional parameters of children, etc.  
        self.check_children(Multinomial, 'p')
        Gibbs.__init__(self, p, verbose)
        self.check_conjugacy(Dirichlet)
        
        self.sum_d = lam_dtrm('sum_d', lambda d=self.d: sum(np.array([sum(np.atleast_2d(d_now),0) for d_now in d]),0))

    def propose(self):
        
        theta = self.sum_d.value
        if self.conjugate:
            theta = theta + self.theta.value
        else:
            theta += 1.
        self.stochastic.value = np.random.dirichlet(theta)
    

class BernoulliAnything(Gibbs):
    """
    Formerly known as BinaryMetropolis    
    """
    
    def __init__(self, stochastic, dist=None):
        # BinaryMetropolis class initialization
        
        # Initialize superclass
        Metropolis.__init__(self, stochastic, dist=dist)
        
        # Initialize verbose feedback string
        self._id = stochastic.__name__
        
    def set_stochastic_val(self, i, val, to_value):
        """
        Utility method for setting a particular element of a stochastic's value.
        """
        
        if self._len>1:
            # Vector-valued stochastics
            
            val[i] = to_value
            self.stochastic.value = reshape(val, check_type(self.stochastic)[1])
        
        else:
            # Scalar stochastics
            
            self.stochastic.value = to_value
    
    def step(self):
        """
        This method is substituted for the default step() method in
        BinaryMetropolis.
        """
            
        # Make local variable for value
        if self._len > 1:
            val = self.stochastic.value.ravel()
        else:
            val = self.stochastic.value
        
        for i in xrange(self._len):
            
            self.set_stochastic_val(i, val, True)
            
            try:
                logp_true = self.stochastic.logp
                loglike_true = self.loglike
            except ZeroProbability:
                self.set_stochastic_val(i, val, False)
                continue
            
            self.set_stochastic_val(i, val, False)
            
            try:
                logp_false = self.stochastic.logp
                loglike_false = self.loglike
            except ZeroProbability:
                self.set_stochastic_val(i,val,True)
                continue
            
            p_true = exp(logp_true + loglike_true)
            p_false = exp(logp_false + loglike_false)
            
            # Stochastically set value according to relative
            # probabilities of True and False
            if np.random.random() > p_true / (p_true + p_false):
                self.set_stochastic_val(i,val,True)
        
        # Increment accepted count
        self._accepted += 1
