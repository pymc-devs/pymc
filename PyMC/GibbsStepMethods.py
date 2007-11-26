from StepMethods import Metropolis, StepMethod
from InstantiationDecorators import dtrm
from Node import ZeroProbability, Variable
from Container import Container
from utils import msqrt
from numpy import asarray, diag, dot, zeros, log, shape, isscalar, sum
from numpy.random import normal, random, gamma, beta
from numpy.linalg import cholesky, solve
from flib import dpotrs_wrap, dtrsm_wrap

def check_list(thing, label):
    if thing is not None:
        if thing.__class__ is not list:
            raise TypeError, 'Argument '+label+' must be a list.'
        return thing


# To do:
# GammaPoisson
# DirichletMultinomial
# BetaGeometric
# ParetoUniform
# GammaExponential
# GammaPareto
# GammaGammaScale
# WishartNormal
# Other parametrizations of the normal/gamma business
# Test case for NormalNormal (blech)

def safe_len(val):
    if isscalar(val):
        return 1
    else:
        return len(val)

class Gibbs(Metropolis):
    
    def __init__(self, stoch, verbose=0):
        Metropolis.__init__(self, stoch, verbose=verbose)
    
    competence = staticmethod(StepMethod.competence)
    
    def step(self):
        if not self.conjugate:
            logp = self.stoch.logp

        self.propose()

        if not self.conjugate:

            try:
                logp_p = self.stoch.logp
            except ZeroProbability:
                self.reject()

            if log(random()) > logp_p - logp:
                self.reject()
    
    def tune(self, verbose):
        return False



# TODO: Automatically fill in when m and all children are normal parameters from class factory.
# TODO: Let A be diagonal.
# TODO: Allow sampling of scalar tau scale factor too.



class GammaNormal(Gibbs):
    """
    Applies to tau in the following submodel:
    
    d ~ind N(mu, tau * theta)
    tau ~ Gamma(alpha, beta) [optional]
    
    The argument tau must be a Stochastic.
    
    The arguments alpha and beta may be:
    - Arrays
    - Scalars
    - Stochastics
    - Deterministics
    - None. In this case, a non-conjugate updating procedure is used.
      tau's value is proposed from its likelihood and accepted based on 
      its prior.
    
      The argument d must be a list or array of lists or arrays of Stochastics.

      The arguments mu and theta must be lists or arrays of:
      - Arrays
      - Scalars
      - Stochastics
      - Deterministics
      These arguments may be lists of length 1 or of the same length as d.
      theta may be a matrix or a vector. If a vector, it is asssumed to be diagonal.
    """
    def __init__(self, tau, d, mu, theta=None, alpha=None, beta=None, verbose=0):
        
        print 'WARNING: GammaNormal is untested'
        self.tau = tau

        self.d = check_list(d, 'd')
        self.mu = check_list(mu, 'mu')
        self.theta = check_list(theta, 'theta')

        self.alpha = alpha
        self.beta = beta
        
        Gibbs.__init__(self, tau, verbose)
        
        if self.alpha is None or self.beta is None:
            self.conjugate = False
        else:
            self.conjugate = True
        
        @dtrm
        def N(d=d):
            """The total number of observations"""
            return sum([safe_len(d_now) for d_now in d])
    
        self.N = N
        self.N_d = len(d)
        
        @dtrm
        def quad_term(d=d, mu=mu, theta=theta):
            """The quadratic term in the likelihood."""
            quad_term = 0
            for i in xrange(self.N_d):
                
                if len(mu)>1:
                    delta_now = d[i] - mu[i]
                else:
                    delta_now = d[i] - mu[0]
    
                if theta is not None:
                    if not isscalar(theta[i]):
                        quad_term += dot(dot(delta_now, theta[i]), delta_now)
                    else:
                        quad_term += dot(delta_now, delta_now) * theta[i]
                else:
                    quad_term += dot(delta_now, delta_now)

            return quad_term*.5
                        
        self.quad_term = quad_term
        
    def propose(self):
        shape = .5*self.N.value
        if self.conjugate:
            shape += self.alpha
            scale = 1./(self.quad_term.value + 1./self.beta)
        else:
            shape += 1.
            scale = 1./self.quad_term.value
            
        self.stoch.value = gamma(shape, scale)
        

class BetaBinomial(Gibbs):
    """
    Applies to p in the following submodel:
    
    d_i ~ind Binomial(n_i, p)
    p ~ Beta(a, b) [optional]
    
    The argument p must be a Stochastic.
    
    The arguments a and b may be:
    - Arrays
    - Scalars
    - Stochastics
    - Deterministics
    - None. In this case, a non-conjugate updating procedure is used.
      p's value is proposed from its likelihood and accepted based on 
      its prior.
    
      The argument d must be a list or array of Stochastics.

      The argument n must be a list of:
      - Arrays
      - Scalars
      - Stochastics
      - Deterministics
      These arguments may be lists of length 1 or of the same length as d.
      theta may be a matrix or a vector. If a vector, it is asssumed to be diagonal.
    """
    def __init__(self, p, d, n, alpha=None, beta=None, verbose=0):
        
        print 'WARNING: BetaBinomial is untested'
        
        self.p = p
        self.d = check_list(d, 'd')
        self.n = check_list(n, 'n')
        self.a = alpha
        self.b = beta
        
        Gibbs.__init__(self, p, verbose)
        
        if self.a is None or self.b is None:
            self.conjugate = False
        else:
            self.conjugate = True
        
        @dtrm
        def N(d=d):
            """The total number of observations."""
            return sum([safe_len(d_now) for d_now in d])
    
        self.N, self.N_d = N, len(d)
        
        @dtrm
        def sum_d(d=d):
            """The sum of the number of 'successes' for each 'experiment'"""
            return sum([sum(d_now) for d_now in d])
        
        @dtrm
        def sum_nmd(sum_d=sum_d,n=n,d=d):
            """The sum of the total number of 'failures' for each 'experiment'"""
            out = -sum_d

            for i in xrange(self.N_d):
                if isscalar(n[i]):
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
            a = a + self.a
            b = b + self.b
        else:
            a += 1.
            b += 1.
        self.stoch.value = beta(a, b)
            
            
class VecBetaBinomial(Gibbs):        
    """
    """
    def __init__(self, p, d, n, alpha=None, beta=None, verbose=0):
        
        print 'WARNING: VecBetaBinomial is untested'
        
        self.p = p
        self.d = d
        self.a = alpha
        self.b = beta
        
        Gibbs.__init__(self, p, verbose)
        
        if self.a is None or self.b is None:
            self.conjugate = False
        else:
            self.conjugate = True
            
        @dtrm
        def n(n=n):
            """n = function(n)"""
            return n
            
        self.n = n
            
    def propose(self):
        a = self.d.value
        b = self.n.value - self.d.value
        if self.conjugate:
            a = a + self.a
            b = b + self.b
        else:
            a += 1.
            b += 1.
        self.stoch.value = beta(a, b)
    
    
    
class NormalNormal(Gibbs):
    """
    Applies to m in following submodel:
    
    d_i ~ind N(A_i m - b_i, theta_i)
    m ~ N(mu, tau) [optional]
    
    S = NormalGibbs(m, mu, tau, d, A, b, theta)
    
    The argument m must be a Stochastic.
    
    The arguments mu and tau may be:
    - Arrays
    - Scalars
    - Stochastics
    - Deterministics
    - None. In this case, a non-conjugate updating procedure is used.
      m's value is proposed from its likelihood and accepted based on 
      its prior.
    tau may be a matrix or vector. If a vector, it is assumed to be diagonal.
    If mu and tau are not provided, it is assumed that the submodel is non-
    conjugate. m's value is proposed from its likelihood and accepted
    according to its prior.
    
    The argument d must be a list or array of Stochastics.
    
    The arguments A, b, and theta must be lists of:
    - Arrays
    - Stochastics
    - Deterministics
    These arguments may be lists of length 1 or of the same length as d.
    theta may be a matrix or a vector. If a vector, it is asssumed to be diagonal.
    
    """
    def __init__(self, m, d, theta, mu=None, tau=None, A=None, b=None, verbose=0):
        
        print 'WARNING: NormalNormal is untested.'
        
        self.d=check_list(d,'d')
        self.theta=check_list(theta,'theta')
        self.A=check_list(A,'A')
        self.b=check_list(b,'b')

        Gibbs.__init__(self, m, verbose)
        
        self.m = m
        
        if mu is None:
            if self.m.parents.has_key('mu') and self.m.parents.has_key('tau'):
                mu = self.m.parents['mu']
                tau = self.m.parents['tau']
            
        self.mu = mu
        self.tau = tau
        
        if self.mu is None or self.tau is None:
            self.conjugate = False
        else:
            self.conjugate = True        
        
        length = safe_len(self.m.value)
        self.length = length

        self.N_d = len(d)
        
        
        # Is the full conditional distribution independent?
        @dtrm
        def all_diag_prec(tau=tau, theta = theta, A=A):
            all_diag_prec = True
            if tau is not None:
                if len(shape(tau))>1:
                    all_diag_prec = False
            
            if A is not None:
                all_diag_prec = False
            
            if not all([len(shape(theta_now))<2 for theta_now in theta]):
                all_diag_prec = False
                
            return all_diag_prec

        self.all_diag_prec = all_diag_prec.value
        
        
        @dtrm
        def prec_and_mean(d=self.d, A=self.A, b=self.b, theta=self.theta, tau=tau, mu=mu):
            """The full conditional precision and mean."""
            
            
            # tau and tau * mu parts.
            if not self.all_diag_prec:
                if self.conjugate:
                    if len(shape(tau))==2:
                        prec = tau                                  
                        mean = dot(tau, mu)
                    else:                                                   
                        prec = diag(tau)                            
                        mean = tau*mu
                else:                                                       
                    prec = zeros((self.length, self.length), dtype=float)
                    mean=zeros(self.length, dtype=float)

            else:
                if self.conjugate:
                    prec = tau
                    mean = dot(tau, mu)
                else:
                    prec = zeros(self.length, dtype=float)
                    mean = zeros(self.length, dtype=float)
            
            
            # Add in A.T theta A and A.T theta (d-b) parts
            for i in xrange(self.N_d):                                                                        
                
                if len(theta)>1:                                        
                    theta_now = theta[i]                                
                else:                                                   
                    theta_now = theta[0]
                
                if b is not None:    
                    if len(b)>1:                                        
                        b_now = d[i] + b[i]
                    else:                                                   
                        b_now = d[i] + b[0]
                else:
                    b_now = d[i]
                
                if self.all_diag_prec:
                    prec += theta_now
                    mean += theta_now * b_now
                else:
                    if A is not None:                                  
                        if len(A)>1:                                    
                            A_now = A[i]
                            # print A[i]                                        
                        else:                                           
                            A_now = A[0]                                        

                        if len(shape(theta_now))==2:
                            A_theta = dot(A_now.T, theta_now)                                 
                        else:                                                   
                            A_theta = A_now.T*theta_now
                        
                        prec += dot(A_theta, A_now)
                        mean += dot(A_theta, b_now)
                        
                    elif len(shape(theta_now))==2:
                        prec += theta_now
                        mean += dot(theta_now, b_now)
                    else:
                        prec += diag(theta_now)
                        mean += theta_now * b_now
            
                        
            # Divide precision into mean.
            # TODO: Accomodate low ranks here.
            if self.all_diag_prec:
                chol_prec = sqrt(prec)
                piv = None
                mean /= prec
            else:
                chol_prec = cholesky(prec)
                dpotrs_wrap(chol_prec, mean, uplo='L')
            return chol_prec, mean
            
        @dtrm
        def chol_prec(prec_and_mean = prec_and_mean):
            """A Cholesky factor of the full conditional precision"""
            return prec_and_mean[0]
                
        @dtrm
        def mean(prec_and_mean = prec_and_mean):
            return prec_and_mean[1]
        
        self.prec_and_mean = prec_and_mean
        self.chol_prec = chol_prec
        self.mean = mean


    def propose(self):
        """
        Sample from the likelihood or the full conditional.
        """
        out = normal(size=self.length)

        chol = self.chol_prec.value
        
        if len(shape(chol))>1:
            dtrsm_wrap(chol, out, uplo='L', transa='T', alpha=1.)
        else:
            out /= chol

        out += self.mean.value
        self.stoch.value = out
