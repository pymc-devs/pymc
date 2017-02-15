from numpy import dot
from numpy.random import normal
import scipy.linalg
import theano.tensor as tt
from theano.tensor import slinalg
from scipy.sparse import issparse

from pymc3.theanof import floatX

import numpy as np

__all__ = ['quad_potential', 'ElemWiseQuadPotential', 'QuadPotential',
           'QuadPotential_Inv', 'isquadpotential']


def quad_potential(C, is_cov, as_cov):
    """
    Parameters
    ----------
    C : arraylike, 0 <= ndim <= 2
        scaling matrix for the potential
        vector treated as diagonal matrix.
    is_cov : Boolean
        whether C is provided as a covariance matrix or hessian
    as_cov : Boolean
        whether the random draws should come from the normal dist
        using the covariance matrix above or the inverse

    Returns
    -------
    q : Quadpotential
    """
    if issparse(C):
        if not chol_available:
            raise ImportError("Sparse mass matrices require scikits.sparse")
        if is_cov != as_cov:
            return QuadPotential_Sparse(C)
        else:
            raise ValueError("Sparse precission matrices are not supported")

    partial_check_positive_definite(C)
    if C.ndim == 1:
        if is_cov != as_cov:
            return ElemWiseQuadPotential(C)
        else:
            return ElemWiseQuadPotential(1. / C)
    else:
        if is_cov != as_cov:
            return QuadPotential(C)
        else:
            return QuadPotential_Inv(C)


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
        return "Scaling is not positive definite. " + self.msg + ". Check indexes " + str(self.idx)


def isquadpotential(o):
    return all(hasattr(o, attr) for attr in ["velocity", "random", "energy"])


class ElemWiseQuadPotential(object):

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


class QuadPotential_Inv(object):

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


class QuadPotential(object):

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
    __all__ += ['QuadPotential_Sparse']

    import theano
    import theano.sparse

    class QuadPotential_Sparse(object):
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
