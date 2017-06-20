from numpy import dot
from numpy.random import normal
import scipy.linalg
import theano.tensor as tt
from theano.tensor import slinalg
from scipy.sparse import issparse
import theano

from pymc3.theanof import floatX

import numpy as np

__all__ = ['quad_potential', 'QuadPotentialDiag', 'QuadPotentialFull',
           'QuadPotentialFullInv', 'QuadPotentialDiagAdapt', 'isquadpotential']


def quad_potential(C, is_cov):
    """
    Parameters
    ----------
    C : arraylike, 0 <= ndim <= 2
        scaling matrix for the potential
        vector treated as diagonal matrix.
    is_cov : Boolean
        whether C is provided as a covariance matrix or hessian

    Returns
    -------
    q : Quadpotential
    """
    if issparse(C):
        if not chol_available:
            raise ImportError("Sparse mass matrices require scikits.sparse")
        if is_cov:
            return QuadPotentialSparse(C)
        else:
            raise ValueError("Sparse precission matrices are not supported")

    partial_check_positive_definite(C)
    if C.ndim == 1:
        if is_cov:
            return QuadPotentialDiag(C)
        else:
            return QuadPotentialDiag(1. / C)
    else:
        if is_cov:
            return QuadPotentialFull(C)
        else:
            return QuadPotentialFullInv(C)


def partial_check_positive_definite(C):
    """Simple but partial check for Positive Definiteness"""
    if C.ndim == 1:
        d = C
    else:
        d = np.diag(C)
    i, = np.nonzero(np.logical_or(np.isnan(d), d <= 0))

    if len(i):
        raise PositiveDefiniteError(
            "Simple check failed. Diagonal contains negatives", i)


class PositiveDefiniteError(ValueError):
    def __init__(self, msg, idx):
        self.idx = idx
        self.msg = msg

    def __str__(self):
        return ("Scaling is not positive definite: %s. Check indexes %s."
                % (self.msg, self.idx))


class QuadPotential(object):
    def velocity(self, x):
        raise NotImplementedError('Abstract method')

    def energy(self, x):
        raise NotImplementedError('Abstract method')

    def random(self, x):
        raise NotImplementedError('Abstract method')

    def adapt(self, sample, grad):
        pass


def isquadpotential(value):
    return isinstance(value, QuadPotential)


class QuadPotentialDiagAdapt(QuadPotential):
    def __init__(self, n, initial_mean, initial_diag=None, initial_weight=0):
        if initial_diag is not None and initial_diag.ndim != 1:
            raise ValueError('Initial diagonal must be one-dimensional.')
        if initial_mean.ndim != 1:
            raise ValueError('Initial mean must be one-dimensional.')
        if initial_diag is not None and len(initial_diag) != n:
            raise ValueError('Wrong shape for initial_diag: expected %s got %s'
                             % (n, len(initial_diag)))
        if len(initial_mean) != n:
            raise ValueError('Wrong shape for initial_mean: expected %s got %s'
                             % (n, len(initial_mean)))

        if initial_diag is None:
            initial_diag = np.ones(n, dtype=theano.config.floatX)
            initial_weight = 1

        self._n = n
        self._var = np.array(initial_diag, dtype=theano.config.floatX, copy=True)
        self._var_theano = theano.shared(self._var)
        self._stds = np.sqrt(initial_diag)
        self._inv_stds = floatX(1.) / self._stds
        self._foreground_var = WeightedVariance(
            self._n, initial_mean, initial_diag, initial_weight)
        self._background_var = WeightedVariance(self._n)
        self._n_samples = 0

    def velocity(self, x):
        return self._var_theano * x

    def energy(self, x):
        return 0.5 * x.dot(self._var_theano * x)

    def random(self):
        vals = floatX(normal(size=self._n))
        return self._inv_stds * vals

    def _update_from_weightvar(self, weightvar):
        weightvar.current_variance(out=self._var)
        np.sqrt(self._var, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)
        self._var_theano.set_value(self._var)

    def adapt(self, sample, grad):
        weight = 1
        window = 100

        self._foreground_var.add_sample(sample, weight)
        self._background_var.add_sample(sample, weight)
        self._update_from_weightvar(self._foreground_var)

        if self._n_samples > 0 and self._n_samples % window == 0:
            self._foreground_var = self._background_var
            self._background_var = WeightedVariance(self._n)

        self._n_samples += 1


class QuadPotentialDiagAdaptGrad(QuadPotentialDiagAdapt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grads1 = np.zeros(self._n)
        self._ngrads1 = 0
        self._grads2 = np.zeros(self._n)
        self._ngrads2 = 0

    def _update(self, var):
        self._var[:] = var
        np.sqrt(self._var, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)
        self._var_theano.set_value(self._var)

    def adapt(self, sample, grad):
        self._grads1[:] += grad ** 2
        self._grads2[:] += grad ** 2
        self._ngrads1 += 1
        self._ngrads2 += 1

        if self._n_samples < 100:
            super().adapt(sample, grad)
        else:
            self._update(self._ngrads1 / self._grads1)

        if self._n_samples > 100 and self._n_samples % 100 == 0:
            self._ngrads1 = self._ngrads2
            self._ngrads2 = 0
            self._grads1[:] = self._grads2
            self._grads2[:] = 0


class WeightedVariance(object):
    def __init__(self, nelem, initial_mean=None, initial_variance=None,
                 initial_weight=0):
        self.w_sum = initial_weight
        self.w_sum2 = initial_weight ** 2
        if initial_mean is None:
            self.mean = np.zeros(nelem, dtype='d')
        else:
            self.mean = np.array(initial_mean, dtype='d', copy=True)
        if initial_variance is None:
            self.raw_var = np.zeros(nelem, dtype='d')
        else:
            self.raw_var = np.array(initial_variance, dtype='d', copy=True)

        self.raw_var[:] *= self.w_sum

        if self.raw_var.shape != (nelem,):
            raise ValueError('Invalid shape for initial variance.')
        if self.mean.shape != (nelem,):
            raise ValueError('Invalid shape for initial mean.')

    def add_sample(self, x, weight):
        x = np.asarray(x)
        self.w_sum += weight
        self.w_sum2 += weight * weight
        prop = weight / self.w_sum
        old_diff = x - self.mean
        self.mean[:] += prop * old_diff
        new_diff = x - self.mean
        self.raw_var[:] += weight * old_diff * new_diff

    def current_variance(self, out=None):
        if self.w_sum == 0:
            raise ValueError('Can not compute variance for empty set of samples.')
        if out is not None:
            return np.divide(self.raw_var, self.w_sum, out=out)
        else:
            return floatX(self.raw_var / self.w_sum)

    def current_mean(self):
        return floatX(self.mean.copy())


class QuadPotentialDiag(QuadPotential):
    def __init__(self, v):
        v = floatX(v)
        s = v ** .5

        self.s = s
        self.inv_s = 1. / s
        self.v = v

    def velocity(self, x):
        return self.v * x

    def random(self):
        return floatX(normal(size=self.s.shape)) * self.inv_s

    def energy(self, x):
        return .5 * x.dot(self.v * x)


class QuadPotentialFullInv(QuadPotential):

    def __init__(self, A):
        self.L = floatX(scipy.linalg.cholesky(A, lower=True))

    def velocity(self, x):
        solve = slinalg.Solve(lower=True)
        y = solve(self.L, x)
        return solve(self.L.T, y)

    def random(self):
        n = floatX(normal(size=self.L.shape[0]))
        return dot(self.L, n)

    def energy(self, x):
        L1x = slinalg.Solve(lower=True)(self.L, x)
        return .5 * L1x.T.dot(L1x)


class QuadPotentialFull(QuadPotential):

    def __init__(self, A):
        self.A = floatX(A)
        self.L = scipy.linalg.cholesky(A, lower=True)

    def velocity(self, x):
        return tt.dot(self.A, x)

    def random(self):
        n = floatX(normal(size=self.L.shape[0]))
        return scipy.linalg.solve_triangular(self.L.T, n)

    def energy(self, x):
        return .5 * x.dot(self.A).dot(x)

    __call__ = random


try:
    import sksparse.cholmod as cholmod
    chol_available = True
except ImportError:
    chol_available = False

if chol_available:
    __all__ += ['QuadPotentialSparse']

    import theano.sparse

    class QuadPotentialSparse(QuadPotential):
        def __init__(self, A):
            self.A = A
            self.size = A.shape[0]
            self.factor = factor = cholmod.cholesky(A)
            self.d_sqrt = np.sqrt(factor.D())

        def velocity(self, x):
            A = theano.sparse.as_sparse(self.A)
            return theano.sparse.dot(A, x)

        def random(self):
            n = floatX(normal(size=self.size))
            n /= self.d_sqrt
            n = self.factor.solve_Lt(n)
            n = self.factor.apply_Pt(n)
            return n

        def energy(self, x):
            return 0.5 * x.T.dot(self.velocity(x))
