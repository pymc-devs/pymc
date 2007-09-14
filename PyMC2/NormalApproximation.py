__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from PyMCObjects import Parameter, Potential
from PyMCBase import ZeroProbability
from Model import Model
from numpy import zeros, inner, asmatrix, ndarray, reshape, shape, arange, matrix, where, diag, asarray, isnan, isinf, ravel, log, Inf
from numpy.random import normal
from numpy.linalg import solve
from utils import msqrt, check_type, round_array, extend_children
from copy import copy

try:
    from scipy.optimize import fmin_ncg, fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg, fmin_l_bfgs_b
    from scipy import derivative
    scipy_imported = True
except ImportError:
    scipy_imported = False

class NormApproxMu(object):
    """
    Returns the mean vector of some parameters.
    
    Usage: If p1 and p2 are array-valued parameters and N is a 
    NormalApproximation or MAP object,
    
    N.mu(p1,p2)
    
    will give the approximate posterior mean of the ravelled, concatenated
    values of p1 and p2.
    """
    def __init__(self, owner):
        self.owner = owner
    
    def __getitem__(self, *params):
        tot_len = 0
        
        try:
            for p in params[0]:
                pass
            param_tuple = params[0]
        except:
            param_tuple = params
        
        for p in param_tuple:
            tot_len += self.owner.param_len[p]
            
        mu = zeros(tot_len, dtype=float)
        
        start_index = 0
        for p in param_tuple:
            mu[start_index:(start_index + self.owner.param_len[p])] = self.owner._mu[self.owner._slices[p]]
            start_index += self.owner.param_len[p]
            
        return mu
        

class NormApproxC(object):
    """
    Returns the covariance matrix of some parameters.
    
    Usage: If p1 and p2 are array-valued parameters and N is a
    NormalApproximation or MAP object,
    
    N.C(p1,p2)
    
    will give the approximate covariance matrix of the ravelled, concatenated 
    values of p1 and p2
    """
    def __init__(self, owner):
        self.owner = owner
            
    def __getitem__(self, *params):
        tot_len = 0
        
        try:
            for p in params[0]:
                pass
            param_tuple = params[0]
        except:
            param_tuple = params
        
        for p in param_tuple:
            tot_len += self.owner.param_len[p]

        C = asmatrix(zeros((tot_len, tot_len)), dtype=float)
            
        start_index1 = 0
        for p1 in param_tuple:
            start_index2 = 0
            for p2 in param_tuple:                
                C[start_index1:(start_index1 + self.owner.param_len[p1]), \
                start_index2:(start_index2 + self.owner.param_len[p2])] = \
                self.owner._C[self.owner._slices[p1],self.owner._slices[p2]]
                
                start_index2 += self.owner.param_len[p2]
                
            start_index1 += self.owner.param_len[p1]
            
        return C
        
class MAP(Model):
    """
    M = MAP(input, db='ram', eps=.001, diff_order=5, verbose=False)
    
    On instantiation, sets all parameters in the model to their maximal a-posteriori
    values.
    
    Useful methods:
    revert_to_max: Sets all parameters to mean value under normal approximation
    fit:            Finds the MAP estimate.
    
    Useful attributes (after fit() is called):
    mu[p1, p2, ...]:    Returns the posterior mean vector of parameters p1, p2, ...
    logp:               Returns the log-probability of the model
    logp_at_max:        Returns the maximum log-probability of the model
    len:                The number of free parameters in the model ('k' in AIC and BIC)
    data_len:           The number of datapoints used ('n' in BIC)
    AIC:                Akaike's Information Criterion for the model
    BIC:                Bayesian Information Criterion for the model
    
    :Arguments:
    input: A dictionary or module, as for Model
    db: A database backend
    eps: 'h' for computing numerical derivatives.
    diff_order: The order of the approximation used to compute derivatives.
        
    :SeeAlso: Model, NormalApproximation, Sampler, scipy.optimize, scipy.derivative
    """
    def __init__(self, input, db='ram', eps=.001, diff_order = 5, verbose=False):
        if not scipy_imported:
            raise ImportError, 'Scipy must be installed to use NormalApproximation and MAP.'
        
        Model.__init__(self, input, db)

        # Allocate memory for internal traces and get parameter slices
        self._slices = {}
        self.len = 0
        self.param_len = {}
        
        self.param_list = list(self.parameters)
        self.N_params = len(self.param_list)
        self.param_indices = []
        self.param_types = []
        self.param_type_dict = {}
        self.extended_children = {}
        class dummy(object):
            pass
        d = dummy()    
        
        for i in xrange(len(self.param_list)):

            parameter = self.param_list[i]
            
            # Extend children of parameter
            d.children = copy(parameter.children)
            extend_children(d)
            self.extended_children[parameter] = d.children
            
            # Check types of all parameters.
            type_now = check_type(parameter)[0]
            self.param_type_dict[parameter] = type_now
            
            if not type_now is float:
                raise TypeError,    "Parameter " + parameter.__name__ + "'s value must be numerical with " + \
                                    "floating-point dtype for NormalApproximation or MAP to be applied."
            
            # Inspect shapes of all parameters and create parameter slices.
            if isinstance(parameter.value, ndarray):
                self.param_len[parameter] = len(ravel(parameter.value))
            else:
                self.param_len[parameter] = 1
            self._slices[parameter] = slice(self.len, self.len + self.param_len[parameter])
            self.len += self.param_len[parameter]
            
            # Record indices that correspond to each parameter.
            for j in range(len(ravel(parameter.value))):
                self.param_indices.append((parameter, j))
                self.param_types.append(type_now)
                
        self.data_len = 0
        for datum in self.data:
            self.data_len += len(ravel(datum.value))
            
        self.eps = eps
        self.diff_order = diff_order
        self.verbose = verbose
        
        self._len_range = arange(self.len)
        
        # Initialize gradient and Hessian matrix.
        self.grad = zeros(self.len, dtype=float)
        self.hess = asmatrix(zeros((self.len, self.len), dtype=float))
        
        self._mu = None
        
        # Initialize NormApproxMu object.
        self.mu = NormApproxMu(self)
        
        
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
        print method
        p = zeros(self.len,dtype=float)
        for parameter in self.parameters:
            p[self._slices[parameter]] = ravel(parameter.value)

        if not self.method == 'newton':
            if not scipy_imported:
                raise ImportError, 'Scipy is required for any method other than Newton in MAP and NormalApproximation'

        if self.verbose:
            def callback(p):
                print 'Current log-probability :', self.logp
        else:
            def callback(p):
                pass
                
        def func_for_diff(val, index):
            """
            The function that gets passed to the derivatives.
            """
            self[index] = val
            return self.i_logp(index)
            
        self.func_for_diff = func_for_diff

        if self.method == 'fmin_ncg':
            p=fmin_ncg( f = self.func, 
                        x0 = p, 
                        fprime = self.gradfunc, 
                        fhess = self.hessfunc, 
                        epsilon=self.eps, 
                        callback=callback, 
                        avextol=tol)

        elif self.method == 'fmin':
            p=fmin( func = self.func, 
                    x0=p, 
                    callback=callback, 
                    ftol=tol)

        elif self.method == 'fmin_powell':
            p=fmin_powell(  func = self.func, 
                            x0=p, 
                            callback=callback, 
                            ftol=tol)

        elif self.method == 'fmin_cg':
            p=fmin_cg(  f = self.func, x0 = p, 
                        fprime = self.gradfunc, 
                        epsilon=self.eps, 
                        callback=callback, 
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

        self._set_parameters(p) 
        self.grad_and_hess()
        self._mu = p
        try:
            self.logp_at_max = self.logp
        except:
            raise RuntimeError, 'Posterior probability optimization converged to value with zero probability.'
        self.AIC = 2. * (self.len - self.logp_at_max) # 2k - 2 ln(L)
        self.BIC = self.len * log(self.data_len) - 2. * self.logp_at_max # k ln(n) - 2 ln(L)

    def func(self, p):
        """
        The function that gets passed to the optimizers.
        """
        self._set_parameters(p)
        try:
            return -1. * self.logp
        except ZeroProbability:
            return Inf

    def gradfunc(self, p):
        """
        The gradient-computing function that gets passed to the optimizers, 
        if needed.
        """
        self._set_parameters(p)
        for i in xrange(self.len):
            self.grad[i] = self.diff(i)            

        return -1 * self.grad


    def _set_parameters(self, p):
        for parameter in self.parameters:
            if self.param_type_dict[parameter] is int:
                parameter.value = round_array(reshape(ravel(p)[self._slices[parameter]],shape(parameter.value)))
            else:
                parameter.value = reshape(ravel(p)[self._slices[parameter]],shape(parameter.value))

    def __setitem__(self, index, value):
        p, i = self.param_indices[index]
        val = ravel(p.value).copy()
        val[i] = value
        p.value = reshape(val, shape(p.value))

    def __getitem__(self, index):
        p, i = self.param_indices[index]
        val = ravel(p.value)
        return val[i]
        
    def set_h(self, oldval):
        """
        Computes h, the derivative step size, from the size of the 
        parameter's value and self.eps. Currently not used.
        """
        # if not oldval == 0.:
        #     h = self.eps * oldval
        # else:
        #     h = self.eps
        # return h
        return self.eps
    
    def i_logp(self, index):
        """
        Evaluates the log-probability of the Markov blanket of
        a parameter owning a particular index.
        """
        all_relevant_params = set()
        p,i = self.param_indices[index]
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
        d = derivative(func=self.func_for_diff, x0=old_val, dx=self.eps, n=order, args=[i], order=self.diff_order)
        self[i] = old_val
        return d

    def diff2(self, i, j):
        """
        N.diff2(i,j)
        
        Mixed second derivative. Differentiates wrt both indices.
        """
        # TODO: Figure out ahead of time which parameters won't have
        # any cross-derivative with each other by checking whether 
        # one is in the other's Markov blanket. Note: this means the
        # normal approximation will call parameters a posteriori 
        # independent if they are not in each other's Markov blankets. 
        # Huh.
        old_val = copy(self[j])

        def diff_for_diff(val):
            self[j] = val
            return self.diff(i)

        d = derivative(func=diff_for_diff, x0=old_val, dx=self.eps, n=1, order=self.diff_order)
        self[j] = old_val
        return d

    def grad_and_hess(self):
        """
        Computes self's gradient and Hessian. Used if the
        optimization method for a NormalApproximation doesn't
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
        self._set_parameters(p)
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
        
        Sets all N's parameters to their MAP values.
        """
        self._set_parameters(self.mu)
        

class NormalApproximation(MAP):
    """
    N = NormalApproximation(input, db='ram', eps=.001, diff_order = 5, method = 'fmin')
    
    Normal approximation to the posterior of a model. Fits self on instantiation.
    
    Useful methods:
    draw:           Draws values for all parameters using normal approximation
    revert_to_max: Sets all parameters to mean value under normal approximation
    fit:            Finds the normal approximation.
    
    Useful attributes (after fit() is called):
    mu[p1, p2, ...]:    Returns the posterior mean vector of parameters p1, p2, ...
    C[p1, p2, ...]:     Returns the posterior covariance of parameters p1, p2, ...
    logp:               Returns the log-probability of the model
    logp_at_max:        Returns the maximum log-probability of the model
    len:                The number of free parameters in the model ('k' in AIC and BIC)
    data_len:           The number of datapoints used ('n' in BIC)
    AIC:                Akaike's Information Criterion for the model
    BIC:                Bayesian Information Criterion for the model
    
    :Arguments:
    input: A dictionary or module, as for Model
    db: A database backend
    eps: 'h' for computing numerical derivatives.
    diff_order: The order of the approximation used to compute derivatives.
        
    :SeeAlso: Model, MAP, Sampler, scipy.optimize
    """

    def __init__(self, input, db='ram', eps=.01, diff_order = 5, verbose=False):
        MAP.__init__(self, input, db, eps, diff_order, verbose)
        self.C = NormApproxC(self)
    
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
        
        Sets all N's parameters to random values drawn from
        the normal approximation to the posterior.
        """
        devs = normal(size=self._sig.shape[1])
        p = inner(self._sig,devs)
        self._set_parameters(p)
    
    def fit(self, method='fmin', iterlim=1000, tol=.00001):
        """
        N.fit()
        
        Computes the normal approximation to the posterior,
        endows self with new attributes.
        """
        MAP.fit(self, method, iterlim, tol)
        self._C = -1. * self.hess.I
        self._sig = msqrt(self._C).T