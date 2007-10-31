# TODO: Test case for EM algorithm.
# TODO: Derivatives from the EM algorithm can't actually be used to compute the Hessian for the normal approx.

# Post-2.0-release:
# TODO: Think about what to do about int-valued stochastics.
# TODO: Allow constraints if fmin_l_bfgs_b is used.


__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from PyMCObjects import Stochastic, Potential
from Node import ZeroProbability
from Model import Model
from numpy import zeros, inner, asmatrix, ndarray, reshape, shape, arange, matrix, where, diag, asarray, isnan, isinf, ravel, log, Inf
from numpy.random import normal
from numpy.linalg import solve
from utils import msqrt, check_type, round_array
from copy import copy

try:
    from scipy.optimize import fmin_ncg, fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg, fmin_l_bfgs_b
    from scipy import derivative
    scipy_imported = True
except ImportError:
    scipy_imported = False

class NormApproxMu(object):
    """
    Returns the mean vector of some variables.
    
    Usage: If p1 and p2 are array-valued stochastic variables and N is a 
    NormApprox or MAP object,
    
    N.mu(p1,p2)
    
    will give the approximate posterior mean of the ravelled, concatenated
    values of p1 and p2.
    """
    def __init__(self, owner):
        self.owner = owner
    
    def __getitem__(self, *stochs):
        
        if not owner.fitted:
            raise ValueError, 'NormApprox object must be fitted before mu can be accessed.'
        
        tot_len = 0
        
        try:
            for p in stochs[0]:
                pass
            stoch_tuple = stochs[0]
        except:
            stoch_tuple = stochs
        
        for p in stoch_tuple:
            tot_len += self.owner.stoch_len[p]
            
        mu = zeros(tot_len, dtype=float)
        
        start_index = 0
        for p in stoch_tuple:
            mu[start_index:(start_index + self.owner.stoch_len[p])] = self.owner._mu[self.owner._slices[p]]
            start_index += self.owner.stoch_len[p]
            
        return mu
        

class NormApproxC(object):
    """
    Returns the covariance matrix of some variables.
    
    Usage: If p1 and p2 are array-valued stochastic variables and N is a
    NormApprox or MAP object,
    
    N.C(p1,p2)
    
    will give the approximate covariance matrix of the ravelled, concatenated 
    values of p1 and p2
    """
    def __init__(self, owner):
        self.owner = owner
            
    def __getitem__(self, *stochs):
        
        if not owner.fitted:
            raise ValueError, 'NormApprox object must be fitted before C can be accessed.'
        
        tot_len = 0
        
        try:
            for p in stochs[0]:
                pass
            stoch_tuple = stochs[0]
        except:
            stoch_tuple = stochs
        
        for p in stoch_tuple:
            tot_len += self.owner.stoch_len[p]

        C = asmatrix(zeros((tot_len, tot_len)), dtype=float)
            
        start_index1 = 0
        for p1 in stoch_tuple:
            start_index2 = 0
            for p2 in stoch_tuple:                
                C[start_index1:(start_index1 + self.owner.stoch_len[p1]), \
                start_index2:(start_index2 + self.owner.stoch_len[p2])] = \
                self.owner._C[self.owner._slices[p1],self.owner._slices[p2]]
                
                start_index2 += self.owner.stoch_len[p2]
                
            start_index1 += self.owner.stoch_len[p1]
            
        return C

class MAP(Model):
    """
    N = MAP(input, db='ram', eps=.001, diff_order = 5)

    Sets all parameters to maximum a posteriori values.

    Useful methods:
    revert_to_max:  Sets all stochastic variables to mean value under normal approximation
    fit:            Finds the normal approximation.

    Useful attributes (after fit() is called):
    mu[p1, p2, ...]:    Returns the posterior mean vector of stochastic variables p1, p2, ...
    logp:               Returns the log-probability of the model
    logp_at_max:        Returns the maximum log-probability of the model
    len:                The number of free stochastic variables in the model ('k' in AIC and BIC)
    data_len:           The number of datapoints used ('n' in BIC)
    AIC:                Akaike's Information Criterion for the model
    BIC:                Bayesian Information Criterion for the model

    :Arguments:
    input: As for Model
    db: A database backend
    eps: 'h' for computing numerical derivatives. May be a dictionary keyed by stochastic variable 
      as well as a scalar.
    diff_order: The order of the approximation used to compute derivatives.

    :SeeAlso: Model, EM, Sampler, scipy.optimize
    """
    def __init__(self, input=None, db='ram', eps=.001, diff_order = 5, verbose=0):
        if not scipy_imported:
            raise ImportError, 'Scipy must be installed to use NormApprox and MAP.'

        Model.__init__(self, input, db=db, verbose=verbose)

        # Allocate memory for internal traces and get stoch slices
        self._slices = {}
        self.len = 0
        self.stoch_len = {}
        self.fitted = False

        self.stoch_list = list(self.stochs)
        self.N_stochs = len(self.stoch_list)
        self.stoch_indices = []
        self.stoch_types = []
        self.stoch_type_dict = {}

        for i in xrange(len(self.stoch_list)):

            stoch = self.stoch_list[i]

            # Check types of all stochs.
            type_now = check_type(stoch)[0]
            self.stoch_type_dict[stoch] = type_now

            if not type_now is float:
                raise TypeError,    "Stochastic " + stoch.__name__ + "'s value must be numerical with " + \
                                    "floating-point dtype for NormApprox or MAP to be applied."

            # Inspect shapes of all stochs and create stoch slices.
            if isinstance(stoch.value, ndarray):
                self.stoch_len[stoch] = len(ravel(stoch.value))
            else:
                self.stoch_len[stoch] = 1
            self._slices[stoch] = slice(self.len, self.len + self.stoch_len[stoch])
            self.len += self.stoch_len[stoch]

            # Record indices that correspond to each stoch.
            for j in range(len(ravel(stoch.value))):
                self.stoch_indices.append((stoch, j))
                self.stoch_types.append(type_now)

        self.data_len = 0
        for datum in self.data:
            self.data_len += len(ravel(datum.value))

        # Unpack step    
        self.eps = zeros(self.len,dtype=float)
        if isinstance(eps,dict):        
            for stoch in self.stochs:
                self.eps[self._slices[stoch]] = eps[stoch]
        else:
            self.eps[:] = eps

        self.diff_order = diff_order
        self.verbose = verbose

        self._len_range = arange(self.len)

        # Initialize gradient and Hessian matrix.
        self.grad = zeros(self.len, dtype=float)
        self.hess = asmatrix(zeros((self.len, self.len), dtype=float))

        self._mu = None

        # Initialize NormApproxMu object.
        self.mu = NormApproxMu(self)

        def func_for_diff(val, index):
            """
            The function that gets passed to the derivatives.
            """
            self[index] = val
            return self.i_logp(index)

        self.func_for_diff = func_for_diff

    def fit(self, method = 'fmin', iterlim=1000, tol=.0001):
        """
        N.fit(method='fmin', iterlim=1000, tol=.001):

        Causes the normal approximation object to fit itself.

        method: May be one of the following, from the scipy.optimize package:
            -fmin_l_bfgs_b
            -fmin_ncg
            -fmin_cg
            -fmin_powell
            -fmin
        """
        self.tol = tol
        self.method = method

        p = zeros(self.len,dtype=float)
        for stoch in self.stochs:
            p[self._slices[stoch]] = ravel(stoch.value)

        if not self.method == 'newton':
            if not scipy_imported:
                raise ImportError, 'Scipy is required to use EM and NormApprox'

        if self.verbose:
            def callback(p):
                print 'Current log-probability : %f' %self.logp
        else:
            def callback(p):
                pass

        if self.method == 'fmin_ncg':
            p=fmin_ncg( f = self.func, 
                        x0 = p, 
                        fprime = self.gradfunc, 
                        fhess = self.hessfunc, 
                        epsilon=self.eps, 
                        maxiter=iterlim,
                        callback=callback, 
                        avextol=tol)

        elif self.method == 'fmin':
	    print iterlim
            p=fmin( func = self.func, 
                    x0=p, 
                    callback=callback, 
                    maxiter=iterlim,
                    ftol=tol)

        elif self.method == 'fmin_powell':
            p=fmin_powell(  func = self.func, 
                            x0=p, 
                            callback=callback, 
                            maxiter=iterlim,
                            ftol=tol)

        elif self.method == 'fmin_cg':
            p=fmin_cg(  f = self.func, x0 = p, 
                        fprime = self.gradfunc, 
                        epsilon=self.eps, 
                        callback=callback, 
                        maxiter=iterlim,
                        gtol=tol)

        elif self.method == 'fmin_l_bfgs_b':
            p=fmin_l_bfgs_b(func = self.func, 
                            x0 = p, 
                            fprime = self.gradfunc, 
                            epsilon = self.eps, 
                            # callback=callback, 
                            pgtol=tol)[0]

        else:
            raise ValueError, 'Method unknown.'

        self._set_stochs(p)
        self._mu = p

        try:
            self.logp_at_max = self.logp
        except:
            raise RuntimeError, 'Posterior probability optimization converged to value with zero probability.'

        self.AIC = 2. * (self.len - self.logp_at_max) # 2k - 2 ln(L)
        self.BIC = self.len * log(self.data_len) - 2. * self.logp_at_max # k ln(n) - 2 ln(L)

        self.fitted = True

    def func(self, p):
        """
        The function that gets passed to the optimizers.
        """
        self._set_stochs(p)
        try:
            return -1. * self.logp
        except ZeroProbability:
            return Inf

    def gradfunc(self, p):
        """
        The gradient-computing function that gets passed to the optimizers, 
        if needed.
        """
        self._set_stochs(p)
        for i in xrange(self.len):
            self.grad[i] = self.diff(i)            

        return -1 * self.grad


    def _set_stochs(self, p):
        for stoch in self.stochs:
            if self.stoch_type_dict[stoch] is int:
                stoch.value = round_array(reshape(ravel(p)[self._slices[stoch]],shape(stoch.value)))
            else:
                stoch.value = reshape(ravel(p)[self._slices[stoch]],shape(stoch.value))

    def __setitem__(self, index, value):
        p, i = self.stoch_indices[index]
        val = ravel(p.value).copy()
        val[i] = value
        p.value = reshape(val, shape(p.value))

    def __getitem__(self, index):
        p, i = self.stoch_indices[index]
        val = ravel(p.value)
        return val[i]

    def i_logp(self, index):
        """
        Evaluates the log-probability of the Markov blanket of
        a stoch owning a particular index.
        """
        all_relevant_stochs = set()
        p,i = self.stoch_indices[index]
        try:
            return p.logp + sum([child.logp for child in self.extended_children[p]])
        except ZeroProbability:
            return -Inf

    def diff(self, i, order=1):
        """
        N.diff(i, order=1)

        Derivative wrt index i to given order.
        """

        old_val = copy(self[i])
        d = derivative(func=self.func_for_diff, x0=old_val, dx=self.eps[i], n=order, args=[i], order=self.diff_order)
        self[i] = old_val
        return d

    def diff2(self, i, j):
        """
        N.diff2(i,j)

        Mixed second derivative. Differentiates wrt both indices.
        """

        old_val = copy(self[j])

        if not self.stoch_indices[i][0] in self.moral_neighbors[self.stoch_indices[j][0]]:
            return 0.

        def diff_for_diff(val):
            self[j] = val
            return self.diff(i)

        d = derivative(func=diff_for_diff, x0=old_val, dx=self.eps[j], n=1, order=self.diff_order)

        self[j] = old_val
        return d

    def grad_and_hess(self):
        """
        Computes self's gradient and Hessian. Used if the
        optimization method for a NormApprox doesn't
        use gradients and hessians, for instance fmin.
        """
        for i in xrange(self.len):

            di = self.diff(i)            
            self.grad[i] = di
            self.hess[i,i] = self.diff(i,2)

            if i < self.len - 1:

                for j in xrange(i+1, self.len):
                    dij = self.diff2(i,j)

                    self.hess[i,j] = dij
                    self.hess[j,i] = dij

    def hessfunc(self, p):
        """
        The Hessian function that will be passed to the optimizer,
        if needed.
        """
        self._set_stochs(p)
        for i in xrange(self.len):

            di = self.diff(i)            
            self.hess[i,i] = self.diff(i,2)

            if i < self.len - 1:

                for j in xrange(i+1, self.len):
                    dij = self.diff2(i,j)

                    self.hess[i,j] = dij
                    self.hess[j,i] = dij 

        return -1. * self.hess                   

    def revert_to_max(self):
        """
        N.revert_to_max()

        Sets all N's stochs to their MAP values.
        """
        self._set_stochs(self.mu)

class NormApprox(MAP):
    """
    N = NormApprox(input, db='ram', eps=.001, diff_order = 5)
    
    Normal approximation to the posterior of a model.
    
    Useful methods:
    draw:           Draws values for all stochastic variables using normal approximation
    revert_to_max:  Sets all stochastic variables to mean value under normal approximation
    fit:            Finds the normal approximation.
    
    Useful attributes (after fit() is called):
    mu[p1, p2, ...]:    Returns the posterior mean vector of stochastic variables p1, p2, ...
    C[p1, p2, ...]:     Returns the posterior covariance of stochastic variables p1, p2, ...
    logp:               Returns the log-probability of the model
    logp_at_max:        Returns the maximum log-probability of the model
    len:                The number of free stochastic variables in the model ('k' in AIC and BIC)
    data_len:           The number of datapoints used ('n' in BIC)
    AIC:                Akaike's Information Criterion for the model
    BIC:                Bayesian Information Criterion for the model
    
    :Arguments:
    input: As for Model
    db: A database backend
    eps: 'h' for computing numerical derivatives. May be a dictionary keyed by stochastic variable 
      as well as a scalar.
    diff_order: The order of the approximation used to compute derivatives.
        
    :SeeAlso: Model, EM, Sampler, scipy.optimize
    """
    def __init__(self, input=None, db='ram', eps=.001, diff_order = 5, verbose=0):
        if not scipy_imported:
            raise ImportError, 'Scipy must be installed to use NormApprox and MAP.'
        
        MAP.__init__(self, input, db, eps, diff_order, verbose)

    def fit(self, *args, **kwargs):
        MAP.fit(self, *args, **kwargs)
        self.fitted = False
        self.grad_and_hess()
        
        self._C = -1. * self.hess.I
        self._sig = msqrt(self._C).T
        
        self.fitted = True

    def sample(self, iter):
        """
        N.sample(iter)
        
        Generates iter samples from the normal approximation and stores
        them in traces, as if they were output from an MCMC loop.
        """
        self.db._initialize(iter)
        for i in xrange(iter):
            self.draw()
            self.tally(i)
        
    def draw(self):
        """
        N.draw()
        
        Sets all N's stochs to random values drawn from
        the normal approximation to the posterior.
        """
        devs = normal(size=self._sig.shape[1])
        p = inner(self._sig,devs)
        self._set_stochs(p)