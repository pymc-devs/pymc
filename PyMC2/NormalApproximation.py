# TODO: Smarter derivatives.
# TODO: Handle likelihood errors.
# TODO: Explicitly consider only probabilities of relevant parameters' Markov blankets
# TODO: to reduce roundoff errors.

__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'

from PyMCObjects import Parameter, Potential
from Model import Model
from numpy import zeros, inner, asmatrix, ndarray, reshape, shape, arange, matrix, where, diag, asarray, isnan, isinf, ravel, log
from numpy.random import normal
from numpy.linalg import solve
from utils import msqrt, check_type, round_array
from copy import copy

try:
    from scipy.optimize import fmin_ncg, fmin, fmin_powell, fmin_cg, fmin_bfgs, fmin_ncg, fmin_l_bfgs_b
    scipy_imported = True
except ImportError:
    scipy_imported = False

class NormApproxMu(object):
    """
    Returns the mean vector of some parameters.
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
    M = MAP(input, db='ram', eps=.001, method = 'fmin')
    
    On instantiation, sets all parameters in the model to their maximal a-posteriori
    values.
    
    Useful methods:
    revert_to_max: Sets all parameters to mean value under normal approximation
    fit:            Finds the MAP estimate. Call this if something changes.
                    Will be useful in EM algorithms.
    
    Useful attributes:
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
    method: May be one of the following, from the scipy.optimize package:
        -fmin_l_bfgs_b
        -fmin_ncg
        -fmin_cg
        -fmin_powell
        -fmin
        -or newton, which is a simple implementation of Newton's method.
        
    :SeeAlso: Model, NormalApproximation, Sampler, scipy.optimize
    """
    def __init__(self, input, db='ram', eps=.000001, tol=.00001, method = 'fmin', verbose=False):
        Model.__init__(self, input, db)

        # Allocate memory for internal traces and get parameter slices
        self._slices = {}
        self.len = 0
        self.param_len = {}
        self.method = method
        
        self.param_list = list(self.parameters)
        self.N_params = len(self.param_list)
        self.param_indices = []
        self.param_types = []
        self.param_type_dict = {}
        
        for i in xrange(len(self.param_list)):

            parameter = self.param_list[i]
            
            type_now = check_type(parameter)[0]
            self.param_type_dict[parameter] = type_now
            
            if not type_now is float or type_now is int:
                raise TypeError,    "Parameter " + parameter.__name__ + "'s value must be numerical or ndarray with " + \
                                    "numerical dtype for NormalApproximation or MAP to be applied."
            
            if isinstance(parameter.value, ndarray):
                self.param_len[parameter] = len(ravel(parameter.value))
            else:
                self.param_len[parameter] = 1
            self._slices[parameter] = slice(self.len, self.len + self.param_len[parameter])
            self.len += self.param_len[parameter]
            
            for j in range(len(ravel(parameter.value))):
                self.param_indices.append((parameter, j))
                self.param_types.append(type_now)
                
        self.data_len = 0
        for datum in self.data:
            self.data_len += len(ravel(datum.value))
            
        self.eps = eps
        
        self._len_range = arange(self.len)
        self.grad = zeros(self.len, dtype=float)
        self.hess = asmatrix(zeros((self.len, self.len), dtype=float))
        
        self._mu = None
        
        self.mu = NormApproxMu(self)
        self.verbose = verbose
        self.tol = tol
        # self.fit(tol=self.tol)

    def _get_logp(self):
        return sum([p.logp for p in self.parameters]) + sum([p.logp for p in self.data]) + sum([p.logp for p in self.potentials])

    logp = property(_get_logp)

    def printp(self, p):
        print p, self.logp

    def fit(self, iterlim=1000, tol=.00001):
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

        if self.method == 'fmin_ncg':
            p=fmin_ncg( f = self.func, 
                        x0 = p, 
                        fprime = self.gradfunc, 
                        fhess = self.hessfunc, 
                        epsilon=self.eps, 
                        callback=callback, 
                        ftol=tol)

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
                        ftol=tol)

        elif self.method == 'fmin_l_bfgs_b':
            p=fmin_l_bfgs_b(func = self.func, 
                            x0 = p, 
                            fprime = self.gradfunc, 
                            epsilon = self.eps, 
                            callback=callback, 
                            ftol=tol)[0]

        elif self.method == 'newton':
            last_logp = self.logp

            for i in xrange(iterlim):
                p_last = p

                self.grad_and_hess()      
                p = p_last - solve(self.hess, self.grad)

                self._set_parameters(p)

                logp = self.logp

                if self.logp < last_logp or isnan(logp) or isinf(logp):
                    p = p_last
                    break
                else:
                    last_logp = logp

                if (abs((p-p_last) / p) < tol).all():
                    break

            if i == iterlim-1:
                raise RuntimeError, "Newton's method failed to converge."
            else:
                print "Newton's method converged in",i,"iterations."
        else:
            raise ValueError, 'Method unknown.'

        self._set_parameters(p) 
        self.grad_and_hess()
        self._mu = p
        self.logp_at_max = self.logp
        self.AIC = 2. * (self.len - self.logp_at_max) # 2k - 2 ln(L)
        self.BIC = self.len * log(self.data_len) - 2. * self.logp_at_max # k ln(n) - 2 ln(L)

    def func(self, p):
        self._set_parameters(p)
        return -1. * self.logp

    def gradfunc(self, p):
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
        # if not oldval == 0.:
        #     h = self.eps * oldval
        # else:
        #     h = self.eps
        # return h
        return self.eps

    def diff(self, index):
        base = self.logp

        oldval = copy(self[index])
        if self.param_types[index] is int:
            return 1
        else:
            h=self.set_h(oldval)

        self[index] = oldval + h
        up = self.logp
        self[index] = oldval

        return (up - base) / h

    def diff2(self, i, j):
        oldval = copy(self[j])
        if self.param_types[j] is int:
            return 1
        else:
            h=self.set_h(oldval)

        base = self.diff(i)

        self[j] = oldval + h
        up = self.diff(i)
        self[j] = oldval

        return (up - base) / h

    def diff2_diag(self, index):
        base = self.logp

        oldval = copy(self[index])
        if self.param_types[index] is int:
            return 1
        else:
            h=self.set_h(oldval)

        self[index] = oldval + h
        up = self.logp

        self[index] = oldval - h
        down = self.logp

        self[index] = oldval

        return (up + down - 2. * base) / h / h


    def grad_and_hess(self):
        for i in xrange(self.len):

            di = self.diff(i)            
            self.grad[i] = di
            self.hess[i,i] = self.diff2_diag(i)

            if i < self.len - 1:

                for j in xrange(i+1, self.len):
                    dij = self.diff2(i,j)

                    self.hess[i,j] = dij
                    self.hess[j,i] = dij

    def hessfunc(self, p):
        self._set_parameters(p)
        for i in xrange(self.len):

            di = self.diff(i)            
            self.hess[i,i] = self.diff2_diag(i)

            if i < self.len - 1:

                for j in xrange(i+1, self.len):
                    dij = self.diff2(i,j)

                    self.hess[i,j] = dij
                    self.hess[j,i] = dij 

        return -1. * self.hess                   

    def revert_to_max(self):
        self._set_parameters(self.mu)
        

class NormalApproximation(MAP):
    """
    N = NormalApproximation(input, db='ram', eps=.001, method = 'fmin')
    
    Normal approximation to the posterior of a model. Fits self on instantiation.
    
    Useful methods:
    draw:           Draws values for all parameters using normal approximation
    revert_to_max: Sets all parameters to mean value under normal approximation
    fit:            Finds the normal approximation. Call this if something changes.
                    Will be useful in EM algorithms.
    
    Useful attributes:
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
    method: May be one of the following, from the scipy.optimize package:
        -fmin_l_bfgs_b
        -fmin_ncg
        -fmin_cg
        -fmin_powell
        -fmin
        -or newton, which is a simple implementation of Newton's method.
        
    :SeeAlso: Model, MAP, Sampler, scipy.optimize
    """

    def __init__(self, input, db='ram', eps=.000001, method = 'fmin'):
        MAP.__init__(self, input, db, eps, method)

        self._C = None
        self._sig = None
        
        self.C = NormApproxC(self)
        self.fit()
        
    def draw(self):
        devs = normal(size=self.len)
        p = inner(devs, self.sig)
        self._set_parameters(p)
    
    def fit(self, iterlim=1000, tol=.00001):
        MAP.fit(self, iterlim, tol)
        
        self._C = -1. * self.hess.I
        self.sig = msqrt(self._C).T