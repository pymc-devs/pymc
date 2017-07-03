from numpy.random import normal
import scipy.linalg
import scipy.sparse
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
        return "Scaling is not positive definite. %s. Check indexes %s" % \
               (self.msg, str(self.idx))


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
        solve = slinalg.Solve(A_structure='lower_triangular')
        y = solve(self.L, x)
        solve = slinalg.Solve(A_structure='upper_triangular')
        return solve(self.L.T, y)

    def random(self):
        n = floatX(normal(size=self.L.shape[0]))
        return np.dot(self.L, n)

    def energy(self, x):
        #L1x = slinalg.Solve(lower=True)(self.L, x)
        #L1x = slinalg.Solve(A_structure='lower_triangular')(self.L, x)
        #return .5 * L1x.T.dot(L1x)
        return 0.5 * tt.dot(x.T, self.velocity(x))


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


class _DiagFactorization(object):
    def __init__(self, diag):
        self._diag = diag
        self._sqrt = np.sqrt(diag)
        self._tt_diag = tt.as_tensor_variable(floatX(self._diag))
        self._tt_sqrt = tt.as_tensor_variable(floatX(self._sqrt))

    def dot(self, x):
        return self._diag * x

    def tt_dot(self, x):
        x = tt.as_tensor_variable(x)
        return self._tt_diag * x

    def solve(self, x):
        return x.T / self._diag

    def tt_solve(self, x):
        x = tt.as_tensor_variable(x)
        return (x.T / self._tt_diag).T

    def dot_factor(self, x):
        return floatX(self._sqrt) * x

    def tt_dot_factor(self, x):
        x = tt.as_tensor_variable(x)
        return self._tt_sqrt * x

    def solve_factor(self, x):
        return (x.T / self._sqrt).T

    def tt_solve_factor(self, x):
        x = tt.as_tensor_variable(x)
        return (x.T / self._tt_sqrt).T

    def solve_factor_t(self, x):
        return (x.T / self._sqrt).T


class LowRankUpdate(object):
    def __init__(self, base, u, k):
        self._base = base
        self._u = u
        self._k = k

        u_prime = self._base.solve_factor(self._u)
        q, r = scipy.linalg.qr(u_prime, mode='economic')
        t = np.eye(len(r)) + np.dot(r, np.dot(np.diag(k), r.T))
        m = scipy.linalg.cholesky(t, lower=True)
        x = m - np.eye(len(m))

        z = k / (k * np.diag(np.dot(u.T, u)) + 1)

        self._z = floatX(z)
        self._q = floatX(q)
        self._x = floatX(x)

    def tt_dot(self, x):
        u, k = self._u, self._k
        u = tt.as_tensor_variable(floatX(u))
        k = tt.as_tensor_variable(floatX(k))

        update = tt.dot(u, k * tt.dot(u.T, x))
        return self._base.tt_dot(x) + update

    def solve(self, x):
        u, k = self._u, self._k
        x = self._base.solve(x)
        update = np.dot(u, np.dot(u.T, x) / (1 + k))
        return x + update

    def solve_factor_t(self, x):
        z, q = self._z, self._q
        x = self._base.solve_factor_t(x)
        return x - np.dot(q, np.dot(z, np.dot(q.T, x)))

    def tt_solve(self, x):
        u, k, z = self._u, self._k, self._z
        u = tt.as_tensor_variable(floatX(u))
        k = tt.as_tensor_variable(floatX(k))
        z = tt.as_tensor_variable(floatX(z))
        # z = (K^-1 + UU^T), diag

        x = self._base.tt_solve(x)
        return x - tt.dot(z * u, tt.dot(u.T, x))

    def dot_factor(self, x):
        q = self._q
        X = self._x

        x = np.dot(q, np.dot(X, np.dot(q.T, x))) + x
        return self._base.dot_factor(x)


class LowRankUpdatePotential(object):
    def __init__(self, stds, diag, u, k):
        self._stds = floatX(stds)
        self._n = len(stds)
        diag_factor = _DiagFactorization(diag)
        self._factor = LowRankUpdate(diag_factor, u, k)

    def velocity(self, x):
        return self._factor.tt_solve(x * self._stds) * self._stds

    def random(self):
        x = floatX(np.random.randn(self._n))
        return self._factor.dot_factor(x) / self._stds

    def energy(self, x):
        return floatX(0.5) * tt.dot(x.T, self.velocity(x))


class QuadPotentialTrace(object):
    def __init__(self, model, trace, n_eigs, lam):
        samples = np.array([model.dict_to_array(s) for s in trace]).T
        samples[:] -= samples.mean(axis=1)[:, None]
        self.stds = samples.std(axis=1)
        samples[:] /= self.stds[:, None]
        self.eigvals, self.eigvecs = self._find_eigs(samples, n_eigs, lam)
        self.diag = 1 - (self.eigvecs ** 2 * self.eigvals).sum(axis=1)
        self.n = len(samples)
        #unexplained = self.n - self.eigvecs.sum()
        #self.diag = np.ones(self.n) / self.n * unexplained

    def __find_eigs(self, samples, n_eigs, lam):
        n, m = samples.shape

        def matvec(x):
            prod = np.dot(samples, np.dot(samples.T, x))
            regularize = np.ones(n) * x
            return (1 - lam) * prod / m + lam * regularize

        mult = scipy.sparse.linalg.LinearOperator(matvec=matvec, shape=(n, n))
        return scipy.sparse.linalg.eigsh(mult, k=n_eigs)

    def velocity(self, x):
        u = floatX(self.eigvecs * np.sqrt(self.eigvals))
        stds = floatX(self.stds)
        diag = floatX(self.diag)
        return (diag * x * stds + tt.dot(u, tt.dot(u.T, x * stds))) * stds

    def random(self):
        v = floatX(np.random.randn(len(self.eigvecs)))
        u_scaled = self.eigvecs * np.sqrt(self.eigvals)
        u_prime = u_scaled / np.sqrt(self.diag)[:, None]
        q, r = scipy.linalg.qr(u_prime, mode='economic')
        t = np.eye(len(r)) + np.dot(r, r.T)
        m = scipy.linalg.cholesky(t, lower=True)
        # x = m - np.eye(len(r))

        qtv = np.dot(q.T, v)
        corr = v - np.dot(q, qtv) + np.dot(q, scipy.linalg.solve(m.T, qtv))
        return floatX(corr / (np.sqrt(self.diag) * self.stds))

    def energy(self, x):
        return x.dot(self.velocity(x)) / 2


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
