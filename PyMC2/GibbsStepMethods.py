from StepMethods import Metropolis
from InstantiationDecorators import dtrm
from PyMCBase import Variable
from Container import Container
from utils import msqrt
from numpy import asarray, diag, dot, zeros, log
from numpy.random import normal, random
from numpy.linalg import cholesky, solve
from flib import dpotrs_wrap, dtrsm_wrap

def check_list(thing, label):
    if thing is not None:
        if thing.__class__ is not list:
            raise TypeError, 'Argument '+label+' must be a list.'
        return thing

# TODO: Automatically fill in when m and all children are normal parameters from class factory.
# TODO: Let A be diagonal.
class NormalGibbs(Metropolis):
    """
    Applies to m in following submodel, where i indexes Stochastic/ Deterministic objects:
    
    d_i ~ind N(A_i m + b_i, d_tau_i)
    m ~ N(mu, tau)
    
    S = NormalGibbs(m, mu, tau, d, A, b, d_tau)
    
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
    
    The argument d must be a list or array of Stochastics.
    
    The arguments A, b, and d_tau must be lists of:
    - Arrays
    - Stochastics
    - Deterministics
    These arguments may be lists of length 1 or of the same length as d.
    d_tau may be a matrix or a vector. If a vector, it is asssumed to be diagonal.
    
    """
    def __init__(self, m, d, d_tau, mu=None, tau=None, A=None, b=None):
        
        self.d=check_list(d,'d')
        self.d_tau=check_list(d_tau,'d_tau')
        self.A=check_list(A,'A')
        self.b=check_list(b,'b')

        Metropolis.__init__(self, m)
        
        self.m = m
        
        if mu is None:
            if self.m.parents.has_key('mu') and self.m.parents.has_key('tau'):
                mu = self.m.parents['mu']
                tau = self.m.parents['tau']
            
        self.mu = mu
        self.tau = tau
        
        length = len(self.m.value)
        self.length = length

        self.N_d = len(d)
        
        
        # Is the full conditional distribution independent?
        @dtrm
        def all_diag_prec(tau=tau, d_tau = d_tau, A=A):
            all_diag_prec = True
            if tau is not None:
                if len(tau.shape)>1:
                    all_diag_prec = False
                
            if not all([len(d_tau_now.shape)==1 for d_tau_now in d_tau]) and A is None:
                self.all_diag_prec = False
                
            return all_diag_prec
        
        self.all_diag_prec = all_diag_prec.value
        
        
        @dtrm
        def prec_and_mean(d=self.d, A=self.A, b=self.b, d_tau=self.d_tau, tau=tau, mu=mu):
            """The full conditional precision and mean."""
            
            
            # tau and tau * mu parts.
            if not self.all_diag_prec:
                if tau is not None:
                    if len(tau.shape)==2:
                        prec = tau                                  
                        mean = dot(tau, mu)
                    else:                                                   
                        prec = diag(tau)                            
                        mean = tau*mu
                else:                                                       
                    prec = zeros((self.length, self.length), dtype=float)
                    mean=zeros(self.length, dtype=float)

            else:
                if tau is not None:
                    prec = tau
                    mean = dot(tau, mu)
                else:
                    prec = zeros(self.length, dtype=float)
                    mean = zeros(self.length, dtype=float)
            
            
            # Add in A.T d_tau A and A.T d_tau (d-b) parts
            for i in xrange(self.N_d):                                                                        
                
                if len(d_tau)>1:                                        
                    d_tau_now = d_tau[i]                                
                else:                                                   
                    d_tau_now = d_tau[0]
                
                if b is not None:    
                    if len(b)>1:                                        
                        b_now = d[i] + b[i]
                    else:                                                   
                        b_now = d[i] + b[0]
                else:
                    b_now = d[i]
                
                if self.all_diag_prec:
                    prec += d_tau_now
                    mean += d_tau_now * b_now
                else:
                    if A is not None:                                  
                        if len(A)>1:                                    
                            A_now = A[i]
                            # print A[i]                                        
                        else:                                           
                            A_now = A[0]                                        

                        if len(d_tau_now.shape)==2:
                            A_d_tau = dot(A_now.T, d_tau_now)                                 
                        else:                                                   
                            A_d_tau = A_now.T*d_tau_now
                        
                        prec += dot(A_d_tau, A_now)
                        mean += dot(A_d_tau, b_now)
                        
                    elif len(d_tau_now.shape)==2:
                        prec += d_tau_now
                        mean += dot(d_tau_now, b_now)
                    else:
                        prec += diag(d_tau_now)
                        mean += d_tau_now * b_now
            
                        
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
                
    def step(self):
        """
        In the non-conjugate case, propose from the likelihood and accept based on the prior.
        In the conjugate case, Gibbs sample.
        """
        if self.tau is None:
            logp = self.m.logp
        
        self.propose()
            
        if self.tau is None:

            try:
                logp_p = self.m.logp
            except ZeroProbability:
                self.reject()
                
            if log(random()) > logp_p - logp:
                self.reject()

    def propose(self):
        """
        Sample from the likelihood or the full conditional.
        """
        out = normal(size=self.length)

        chol = self.chol_prec.value
        
        if len(chol.shape)>1:
            dtrsm_wrap(chol, out, uplo='L', transa='N', alpha=1.r)
        else:
            out /= chol

        out += self.mean.value
        self.m.value = out        

    def reject(self):
        self.m.value = self.m.last_value
        
    def tune(self):
        pass