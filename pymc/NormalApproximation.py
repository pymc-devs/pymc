# TODO: Test case for EM algorithm.
# TODO: Derivatives from the EM algorithm can't actually be used to compute the Hessian for the normal approx.

# Post-2.0-release:
# TODO: Think about what to do about int-valued stochastics.
# TODO: Allow constraints if fmin_l_bfgs_b is used.


__docformat__='reStructuredText'

__author__ = 'Anand Patil, anand.prabhakar.patil@gmail.com'
__all__ = ['NormApproxMu', 'NormApproxC', 'MAP', 'NormApprox']

from PyMCObjects import Stochastic, Potential
from Node import ZeroProbability
from Model import Model, Sampler
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

    def __getitem__(self, *stochastics):

        if not self.owner.fitted:
            raise ValueError, 'NormApprox object must be fitted before mu can be accessed.'

        tot_len = 0

        try:
            for p in stochastics[0]:
                pass
            stochastic_tuple = stochastics[0]
        except:
            stochastic_tuple = stochastics

        for p in stochastic_tuple:
            tot_len += self.owner.stochastic_len[p]

        mu = zeros(tot_len, dtype=float)

        start_index = 0
        for p in stochastic_tuple:
            mu[start_index:(start_index + self.owner.stochastic_len[p])] = self.owner._mu[self.owner._slices[p]]
            start_index += self.owner.stochastic_len[p]

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

    def __getitem__(self, *stochastics):

        if not self.owner.fitted:
            raise ValueError, 'NormApprox object must be fitted before C can be accessed.'

        tot_len = 0

        try:
            for p in stochastics[0]:
                pass
            stochastic_tuple = stochastics[0]
        except:
            stochastic_tuple = stochastics

        for p in stochastic_tuple:
            tot_len += self.owner.stochastic_len[p]

        C = asmatrix(zeros((tot_len, tot_len)), dtype=float)

        start_index1 = 0
        for p1 in stochastic_tuple:
            start_index2 = 0
            for p2 in stochastic_tuple:
                C[start_index1:(start_index1 + self.owner.stochastic_len[p1]), \
                start_index2:(start_index2 + self.owner.stochastic_len[p2])] = \
                self.owner._C[self.owner._slices[p1],self.owner._slices[p2]]

                start_index2 += self.owner.stochastic_len[p2]

            start_index1 += self.owner.stochastic_len[p1]

        return C

class MAP(Model):
    """
    N = MAP(input, eps=.001, diff_order = 5)

    Sets all parameters to maximum a posteriori values.

    Useful methods:
    revert_to_max:  Sets all stochastic variables to mean value under normal approximation
    fit:            Finds the normal approximation.

    Useful attributes (after fit() is called):
    logp:               Returns the log-probability of the model
    logp_at_max:        Returns the maximum log-probability of the model
    len:                The number of free stochastic variables in the model ('k' in AIC and BIC)
    data_len:           The number of datapoints used ('n' in BIC)
    AIC:                Akaike's Information Criterion for the model
    BIC:                Bayesian Information Criterion for the model

    :Arguments:
    input: As for Model
    eps: 'h' for computing numerical derivatives. May be a dictionary keyed by stochastic variable
      as well as a scalar.
    diff_order: The order of the approximation used to compute derivatives.

    :SeeAlso: Model, EM, Sampler, scipy.optimize
    """
    def __init__(self, input=None, eps=.001, diff_order = 5, verbose=0):
        if not scipy_imported:
            raise ImportError, 'Scipy must be installed to use NormApprox and MAP.'

        Model.__init__(self, input, verbose=verbose)

        # Allocate memory for internal traces and get stochastic slices
        self._slices = {}
        self.len = 0
        self.stochastic_len = {}
        self.fitted = False

        self.stochastic_list = list(self.stochastics)
        self.N_stochastics = len(self.stochastic_list)
        self.stochastic_indices = []
        self.stochastic_types = []
        self.stochastic_type_dict = {}

        for i in xrange(len(self.stochastic_list)):

            stochastic = self.stochastic_list[i]

            # Check types of all stochastics.
            type_now = check_type(stochastic)[0]
            self.stochastic_type_dict[stochastic] = type_now

            if not type_now is float:
                print "Warning: Stochastic " + stochastic.__name__ + "'s value is neither numerical nor array with " + \
                            "floating-point dtype. Recommend fitting method fmin (default)."

            # Inspect shapes of all stochastics and create stochastic slices.
            if isinstance(stochastic.value, ndarray):
                self.stochastic_len[stochastic] = len(ravel(stochastic.value))
            else:
                self.stochastic_len[stochastic] = 1
            self._slices[stochastic] = slice(self.len, self.len + self.stochastic_len[stochastic])
            self.len += self.stochastic_len[stochastic]

            # Record indices that correspond to each stochastic.
            for j in range(len(ravel(stochastic.value))):
                self.stochastic_indices.append((stochastic, j))
                self.stochastic_types.append(type_now)

        self.data_len = 0
        for datum in self.observed_stochastics:
            self.data_len += len(ravel(datum.value))

        # Unpack step
        self.eps = zeros(self.len,dtype=float)
        if isinstance(eps,dict):
            for stochastic in self.stochastics:
                self.eps[self._slices[stochastic]] = eps[stochastic]
        else:
            self.eps[:] = eps

        self.diff_order = diff_order

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

    def fit(self, method = 'fmin', iterlim=1000, tol=.0001, verbose=0):
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
        self.verbose = verbose

        p = zeros(self.len,dtype=float)
        for stochastic in self.stochastics:
            p[self._slices[stochastic]] = ravel(stochastic.value)

        if not self.method == 'newton':
            if not scipy_imported:
                raise ImportError, 'Scipy is required to use EM and NormApprox'

        if self.verbose > 0:
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
                        avextol=tol,
                        disp=verbose)

        elif self.method == 'fmin':

            p=fmin( func = self.func,
                    x0=p,
                    callback=callback,
                    maxiter=iterlim,
                    ftol=tol,
                    disp=verbose)

        elif self.method == 'fmin_powell':
            p=fmin_powell(  func = self.func,
                            x0=p,
                            callback=callback,
                            maxiter=iterlim,
                            ftol=tol,
                            disp=verbose)

        elif self.method == 'fmin_cg':
            p=fmin_cg(  f = self.func, x0 = p,
                        fprime = self.gradfunc,
                        epsilon=self.eps,
                        callback=callback,
                        maxiter=iterlim,
                        gtol=tol,
                        disp=verbose)

        elif self.method == 'fmin_l_bfgs_b':
            p=fmin_l_bfgs_b(func = self.func,
                            x0 = p,
                            fprime = self.gradfunc,
                            epsilon = self.eps,
                            # callback=callback,
                            pgtol=tol,
                            iprint=verbose-1)[0]

        else:
            raise ValueError, 'Method unknown.'

        self._set_stochastics(p)
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
        self._set_stochastics(p)
        try:
            return -1. * self.logp
        except ZeroProbability:
            return Inf

    def gradfunc(self, p):
        """
        The gradient-computing function that gets passed to the optimizers,
        if needed.
        """
        self._set_stochastics(p)
        for i in xrange(self.len):
            self.grad[i] = self.diff(i)

        return -1 * self.grad


    def _set_stochastics(self, p):
        for stochastic in self.stochastics:
            if self.stochastic_type_dict[stochastic] is int:
                stochastic.value = round_array(reshape(ravel(p)[self._slices[stochastic]],shape(stochastic.value)))
            else:
                stochastic.value = reshape(ravel(p)[self._slices[stochastic]],shape(stochastic.value))

    def __setitem__(self, index, value):
        p, i = self.stochastic_indices[index]
        val = ravel(p.value).copy()
        val[i] = value
        p.value = reshape(val, shape(p.value))

    def __getitem__(self, index):
        p, i = self.stochastic_indices[index]
        val = ravel(p.value)
        return val[i]

    def i_logp(self, index):
        """
        Evaluates the log-probability of the Markov blanket of
        a stochastic owning a particular index.
        """
        all_relevant_stochastics = set()
        p,i = self.stochastic_indices[index]
        try:
            return p.logp + sum([child.logp for child in p.extended_children])
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

        if not self.stochastic_indices[i][0] in self.stochastic_indices[j][0].moral_neighbors:
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
        self._set_stochastics(p)
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

        Sets all N's stochastics to their MAP values.
        """
        self._set_stochastics(self.mu)

class NormApprox(MAP, Sampler):
    """
    N = NormApprox(input, db='ram', eps=.001, diff_order = 5, **kwds)

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
    def __init__(self, input=None, db='ram', eps=.001, diff_order = 5, **kwds):
        if not scipy_imported:
            raise ImportError, 'Scipy must be installed to use NormApprox and MAP.'

        MAP.__init__(self, input, eps, diff_order)

        Sampler.__init__(self, input, db, reinit_model=False, **kwds)
        self.C = NormApproxC(self)

    def fit(self, *args, **kwargs):
        MAP.fit(self, *args, **kwargs)
        self.fitted = False
        self.grad_and_hess()

        self._C = -1. * self.hess.I
        self._sig = msqrt(self._C).T

        self.fitted = True

    def draw(self):
        """
        N.draw()

        Sets all N's stochastics to random values drawn from
        the normal approximation to the posterior.
        """
        devs = normal(size=self._sig.shape[1])
        p = inner(self._sig,devs) + self._mu
        self._set_stochastics(p)
