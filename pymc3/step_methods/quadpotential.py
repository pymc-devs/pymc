from numpy import dot
from numpy.random import normal
from numpy.linalg import solve
from scipy.linalg import cholesky, cho_solve
from scipy.sparse import issparse

import numpy as np

__all__ = ['quad_potential', 'ElemWiseQuadPotential', 'QuadPotential',
           'QuadPotential_Inv', 'isquadpotential']


def quad_potential(C, is_cov, as_cov):
    """
    Parameters
    ----------
        C : arraylike, 0 <= ndim <= 2
            scaling matrix for the potential
            vector treated as diagonal matrix
        is_cov : Boolean
            whether C is provided as a covariance matrix or hessian
        as_cov : Boolean
            whether the random draws should come from the normal dist
            using the covariance matrix above or the inverse

    Returns
    -------
        q : Quadpotential
    """

    if issparse(C) and is_cov != as_cov:
        if not chol_available:
            raise ImportError("Requires scikits.sparse")
        return QuadPotential_SparseInv(C)

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
        s = v ** .5

        self.s = s
        self.inv_s = 1. / s
        self.v = v

    def velocity(self, x):
        return self.v * x

    def random(self):
        return normal(size=self.s.shape) * self.inv_s

    def energy(self, x):
        return .5 * x.dot(self.v * x)


class QuadPotential_Inv(object):

    def __init__(self, A):
        self.L = cholesky(A, lower=True)

    def velocity(self, x):
        return cho_solve((self.L, True), x)

    def random(self):
        n = normal(size=self.L.shape[0])
        return dot(self.L, n)

    def energy(self, x):
        L1x = solve(self.L, x)
        return .5 * dot(L1x.T, L1x)


class QuadPotential(object):

    def __init__(self, A):
        self.A = A
        self.L = cholesky(A, lower=True)

    def velocity(self, x):
        return x.T.dot(self.A.T)

    def random(self):
        n = normal(size=self.L.shape[0])
        return solve(self.L.T, n)

    def energy(self, x):
        return .5 * x.dot(self.A).dot(x)

    __call__ = random

try:
    import scikits.sparse.cholmod as cholmod
    chol_available = True
except ImportError:
    chol_available = False

if chol_available:
    __all__ += ['QuadPotential_SparseInv']

    class QuadPotential_SparseInv(object):

        def __init__(self, A):
            self.n = A.shape[0]
            self.factor = factor = cholmod.cholesky(A)
            self.L = factor.L()
            self.p = np.argsort(factor.P())

        def velocity(self, x):
            x = np.ones((x.shape[0], 2)) * x[:, np.newaxis]
            return self.factor(x)[:, 0]

        def Ldot(self, x):
            return (self.L * x)[self.p]

        def random(self):
            return self.Ldot(normal(size=self.n))

        def energy(self, x):
            return .5 * dot(x, self.velocity(x))
