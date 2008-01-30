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
        return len(val)

class Gibbs(Metropolis):
    
    def __init__(self, stochastic, verbose=0):
        Metropolis.__init__(self, stochastic, verbose=verbose)
    
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

class BernoulliAnything(Gibbs):
    """
    Formerly known as BinaryMetropolis
    
    Like Metropolis, but with a modified step() method.
    Good for binary variables.
    
    NOTE this is not compliant with the Metropolis standard
    yet because it lacks a reject() method.
    (??? But, it is a subclass of Metropolis, which has a reject() method)
    True... but it's never called, this is really a Gibbs sampler since there
    are only 2 states available.
    
    This should be a subclass of Gibbs, not Metropolis.
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
        if not isinstance(self.stochastic, Gamma):
            self.a = None
            self.b = None
            self.conjugate = False
        else:
            self.a = lam_dtrm('alpha',lambda alpha=tau.parents['alpha']: alpha)
            self.b = lam_dtrm('beta',lambda beta=tau.parents['beta']: beta)            
            self.conjugate = True
        
        Gibbs.__init__(self, tau, verbose)
        
        @dtrm
        def N(d=self.d):
            """The total number of observations"""
            return sum([safe_len(d_now) for d_now in d])
    
        self.N = N
        self.N_d = len(self.d)
        
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

        a = .5*self.N.value
        b = self.quad_term.value
        if self.conjugate:
            a += self.a.value
            b += self.b.value
        else:
            a += 1.

        self.stochastic.value = np.random.gamma(a, 1./b)


class WishartMvNormal(Gibbs):
    """
    Applies to tau in the following submodel:
    
    d_i ~ind Normal(mu_i, tau)
    tau ~ WishartCov(n, Tau) [optional]
    
    where the stochastics d are parametrized by precision, not covariance.
    """
    def __init__(self, tau, verbose=0):
        
        self.stochastic = tau
        self.mu = []
        self.d = []
        
        # Get distributional parameters from children and make sure children are Normal.
        for child in tau.children:
            if isinstance(child, MvNormal):                
                # TODO: Allow covariance to be scalar multiple of self plus something else. Use LinearTauombination class.
                self.d.append(child)
                self.mu.append(child.parents['mu'])
            else:
                raise ValueError, 'Stochastic %s must have all multivariate normal children for WishartMvNormal\n \
                                    to be able to handle it.' %self.stochastic.__name__
        
        self.N = self.stochastic.value.shape[0]
        
        # See whether to use conjugate or non-conjugate version.
        if not isinstance(self.stochastic, Wishart):
            self.n = None
            self.Tau = None
            self.conjugate = False
        else:
            self.n = lam_dtrm('n',lambda n=tau.parents['n']: n)
            self.Tau = lam_dtrm('Tau',lambda Tau=tau.parents['Tau']: Tau)            
            self.conjugate = True
        
        Gibbs.__init__(self, tau, verbose)
        
        @dtrm
        def quad_term(d=self.d, mu=self.mu):
            """The quadratic term in the likelihood."""
            quad_array = np.asmatrix(np.empty((len(self.d), self.N)))
            for i in xrange(len(d)):
                quad_array[i,:] = d[i] - mu[i]
            return quad_array.T * quad_array
                        
        self.quad_term = quad_term
        
    def propose(self):
        n = len(self.d)
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
        self.d = []
        
        # Get distributional parameters of children, etc.
        for child in mu.children:
            if not isinstance(child, Poisson):
                raise ValueError, 'Stochastic %s must have all Poisson children for GammaPoisson\n \
                                    to be able to handle it.' %self.stochastic.__name__
            self.d.append(child)
        
        Gibbs.__init__(self, mu, verbose)
        
        # Conjugate or non-conjugate?
        if not isinstance(self.stochastic, Gamma):
            self.a = None
            self.b = None
            self.conjugate = False
        else:
            self.a = lam_dtrm('a',lambda a=mu.parents['alpha']: a)
            self.b = lam_dtrm('b',lambda b=mu.parents['beta']: b)            
            self.conjugate = True
                
        @dtrm
        def N(d=self.d):
            """The total number of observations."""
            return sum([safe_len(d_now) for d_now in d])
    
        self.N, self.N_d = N, len(self.d)
        
        @dtrm
        def sum_d(d=self.d):
            """The sum of the number of 'successes' for each 'experiment'"""
            return sum([sum(d_now) for d_now in d])
                    
        self.sum_d = sum_d
                    
    def propose(self):
        b = self.N.value        
        a = self.sum_d.value
        if self.conjugate:
            a = a + self.a.value
            b = b + self.b.value
        else:
            a += 1.
            b += 1.
        self.stochastic.value = np.random.gamma(a, 1./b)

class BetaGeometric(Gibbs):
    """
    Applies to p in the following submodel:

    d_i ~ind Geometric(p)
    p ~ Beta(alpha, beta) [optional]

    The argument p must be a Stochastic.
    """
    def __init__(self, p, verbose=0):

        self.stochastic = p
        self.d = []

        # Get distributional parameters of children, etc.
        for child in p.children:
            if not isinstance(child, Geometric):
                raise ValueError, 'Stochastic %s pst have all Geometric children for BetaGeometric\n \
                                    to be able to handle it.' %self.stochastic.__name__
            self.d.append(child)

        Gibbs.__init__(self, p, verbose)

        # Conjugate or non-conjugate?
        if not isinstance(self.stochastic, Beta):
            self.a = None
            self.b = None
            self.conjugate = False
        else:
            self.a = lam_dtrm('a',lambda a=p.parents['alpha']: a)
            self.b = lam_dtrm('b',lambda b=p.parents['beta']: b)            
            self.conjugate = True

        @dtrm
        def N(d=self.d):
            """The total number of observations."""
            return sum([safe_len(d_now) for d_now in d])

        self.N, self.N_d = N, len(self.d)

        @dtrm
        def sum_d(d=self.d):
            """The sum of the number of 'successes' for each 'experiment'"""
            return sum([sum(d_now) for d_now in d])

        self.sum_d = sum_d

    def propose(self):
        a = self.N.value        
        b = self.sum_d.value
        if self.conjugate:
            a = a + self.a.value
            b = b + self.b.value
        else:
            a += 1.
            b += 1.
        self.stochastic.value = np.random.beta(a, b)

class GammaExponential(Gibbs):
    """
    Applies to p in the following submodel:
    
    d_i ~ind Exponential(beta)
    beta ~ Gamma(alpha, beta) [optional]
    
    The argument beta must be a Stochastic.
    """
    def __init__(self, beta, verbose=0):
        
        self.stochastic = beta
        self.d = []
        
        # Get distributional parameters of children, etc.        
        for child in beta.children:
            if not isinstance(child, Exponential):
                raise ValueError, 'Stochastic %s betast have all Exponential children for GammaPoisson\n \
                                    to be able to handle it.' %self.stochastic.__name__
            self.d.append(child)
        
        Gibbs.__init__(self, beta, verbose)

        # Conjugate or non-conjugate?        
        if not isinstance(self.stochastic, Gamma):
            self.a = None
            self.b = None
            self.conjugate = False
        else:
            self.a = lam_dtrm('a',lambda a=beta.parents['alpha']: a)
            self.b = lam_dtrm('b',lambda b=beta.parents['beta']: b)            
            self.conjugate = True
                
        @dtrm
        def N(d=self.d):
            """The total number of observations."""
            return sum([safe_len(d_now) for d_now in d])
    
        self.N, self.N_d = N, len(self.d)
        
        @dtrm
        def sum_d(d=self.d):
            """The sum of the number of 'successes' for each 'experiment'"""
            return sum([sum(d_now) for d_now in d])
                    
        self.sum_d = sum_d
                    
    def propose(self):
        a = self.N.value        
        b = self.sum_d.value
        if self.conjugate:
            a = a + self.a.value
            b = b + self.b.value
        else:
            a += 1.
            b += 1.
        self.stochastic.value = np.random.gamma(a, 1./b)

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
        self.n = []
        self.d = []
        
        # Get distributional parameters of children, etc.        
        for child in p.children:
            if not isinstance(child, Binomial):
                raise ValueError, 'Stochastic %s must have all binomial children for BetaBinomial\n \
                                    to be able to handle it.' %self.stochastic.__name__
            self.d.append(child)
            self.n.append(child.parents['n'])
        
        Gibbs.__init__(self, p, verbose)

        # Conjugate or non-conjugate?        
        if not isinstance(self.stochastic, Beta):
            self.a = None
            self.b = None
            self.conjugate = False
        else:
            self.a = lam_dtrm('a',lambda a=p.parents['alpha']: a)
            self.b = lam_dtrm('b',lambda b=p.parents['beta']: b)            
            self.conjugate = True
                
        @dtrm
        def N(d=self.d):
            """The total number of observations."""
            return sum([safe_len(d_now) for d_now in d])
    
        self.N, self.N_d = N, len(self.d)
        
        @dtrm
        def sum_d(d=self.d):
            """The sum of the number of 'successes' for each 'experiment'"""
            return sum([sum(d_now) for d_now in d])
        
        @dtrm
        def sum_nmd(sum_d=sum_d,n=self.n,d=self.d):
            """The sum of the total number of 'failures' for each 'experiment'"""
            out = -sum_d

            for i in xrange(self.N_d):
                if np.isscalar(n[i]):
                    out += n[i]*safe_len(d[i])
                else:
                    out += sum(n[i])
                    
            return out
            
        self.sum_d = sum_d
        self.sum_nmd = sum_nmd
                    
    def propose(self):
        a = self.sum_d.value
        b = self.sum_nmd.value
        if self.conjugate:
            a = a + self.a.value
            b = b + self.b.value
        else:
            a += 1.
            b += 1.
        self.stochastic.value = np.random.beta(a, b)

class DirichletMultinomial(Gibbs):            
    """
    Applies to p in the following submodel:
    
    d_i ~ind Multinomial(n_i, p)
    p ~ Dirichlet(theta) [optional] 
    
    p must be a Stochastic, preferably a Dirichlet.   
    """
    def __init__(self, p, verbose=0):
        
        self.stochastic = p
        self.n = []
        self.d = []
        
        # Get distributional parameters of children, etc.        
        for child in p.children:
            if not isinstance(child, Multinomial):
                raise ValueError, 'Stochastic %s must have all multinomial children for DirichletMultinomial\n \
                                    to be able to handle it.' %self.stochastic.__name__
            self.d.append(child)
            self.n.append(child.parents['n'])
        
        Gibbs.__init__(self, p, verbose)

        # Conjugate or non-conjugate?        
        if not isinstance(self.stochastic, Dirichlet):
            self.theta = None
            self.conjugate = False
        else:
            self.theta = lam_dtrm('theta',lambda theta=p.parents['theta']: theta)
            self.conjugate = True
        
        self.sum_d = lam_dtrm('sum_d', lambda d=self.d: sum(np.array([sum(np.atleast_2d(d_now),0) for d_now in d]),0))

    def propose(self):
        
        theta = self.sum_d.value
        if self.conjugate:
            theta = theta + self.theta.value
        else:
            theta += 1.
        self.stochastic.value = np.random.dirichlet(theta)
    
