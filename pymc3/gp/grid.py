import theano.tensor as tt
import pymc3 as pm
from pymc3.gp.gp import Base
from pymc3.gp.cov import Covariance
from pymc3.gp.mean import Constant
from pymc3.gp.util import (conditioned_vars,
    infer_shape, stabilize, cholesky, solve, solve_lower, solve_upper)


__all__ = ["Grid2DLatent"]

def vec(A):
    return tt.flatten(tt.transpose(A))[:, None]


def devec(x, r, c):
    return x.reshape((c,r)).T
    #return tt.transpose(tt.reshape(x, (r, c)))


def grid_to_full(X1, X2, n1, n2):
    n1 = infer_shape(X1)
    n2 = infer_shape(X2)
    X1 = tt.as_tensor_variable(X1)
    X2 = tt.as_tensor_variable(X2)
    xx1 = tt.repeat(X1, n2, 0)
    xx2 = tt.tile(X2, (n1, 1))
    return tt.concatenate((xx1, xx2), 1)


def kronprod(A1, A2, v, veced=True):
    if veced:
        v = devec(v, A2.shape[0], A1.shape[0])
    tmp = tt.dot(v, A1)
    tmp = tt.dot(A2, tmp)
    return tt.dot(A2, tt.dot(v, A1))


def kronsolve(A1, A2, v, veced=True, chol=False):
    """ if chol is true, assume A1, A2 chol factors """
    if veced:
        v = devec(v, A2.shape[0], A1.shape[0])
    if chol:
        return vec(solve(A2, tt.transpose(solve_upper(tt.transpose(A1), tt.transpose(v)))))
    else:
        return vec(solve(A2, tt.transpose(solve(tt.transpose(A1), tt.transpose(v)))))


def make_gridcov_func(self, cov_func1, cov_func2):
    ndim1 = cov_func1.input_dim
    def gridcov_func(X, Xnew=None):
        if Xnew is not None:
            return cov_func1(X[:,:ndim1], Xnew[:,ndim1:]) * cov_func2(X[:,:ndim1], Xnew[:,ndim2:])
        else:
            return cov_func1(X[:,:ndim1]) * cov_func2(X[:,:ndim1])
    return gridcov_func


class Grid(Base):
    def __init__(self, cov_funcs, mean_func):
        self.cov_func1, self.cov_func2 = cov_funcs
        self.mean_func = mean_func
        # K1 is 5x5
        # K2 is 3x3
        # Y is 3x5

@conditioned_vars(["X", "n", "f"])
class Grid2DLatent(Grid):
    def _build_prior(self, name, X, n, reparameterize=True):
        mu = 0.0
        L1 = cholesky(stabilize(self.cov_func1(X[0])))
        L2 = cholesky(stabilize(self.cov_func2(X[1])))
        if reparameterize:
            v = pm.Normal(name + "_rotated_", mu=0.0, sd=1.0, shape=n[0] * n[1])
            f = pm.Deterministic(name, devec(kronprod(L1, L2, v[:,None]), n[0], n[1]))
        else:
            raise NotImplementedError
        return f

    def prior(self, name, X, n=None, reparameterize=True):
        if n is None:
            n = (None, None)
        if len(X) != 2 or len(n) != 2:
            raise ValueError("2d implemented")
        n = (infer_shape(X[0], n[0]), infer_shape(X[1], n[1]))
        f = self._build_prior(name, X, n, reparameterize)
        self.X = (tt.as_tensor_variable(X[0]), tt.as_tensor_variable(X[1]))
        self.n = n
        self.f = f
        return f

    def _get_given_vals(self, **given):
        if 'gp' in given:
            cov_total = given['gp'].cov_func
            mean_total = given['gp'].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ['X', 'n', 'f']):
            X, n, f = given['X'], given['n'], given['f']
            if not isinstance(X, tuple) and not isinstance(n, tuple):
                raise ValueError("must provide tuple with each element for dimension")
        else:
            X, n, f = self.X, self.n, self.f
        return X, n, f, cov_total, mean_total

    def conditional(self, name, Xnew, n_points=None, given=None):
        givens = self._get_given_vals(**given)
        mu, cov, n_points = self._build_conditional(Xnew, n_points, *givens)
        chol = cholesky(stabilize(cov))
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)

    def _build_conditional(self, Xnew, n_points, X, n, f, cov_total, mean_total):
        Kxx1 = cov_total1(X1)
        Kxx2 = cov_total2(X2)
        L1 = cholesky(stabilize(Kxx1))
        L2 = cholesky(stabilize(Kxx2))
        if isinstance(Xnew, tuple) and isinstance(n_points, tuple):
            # if X1new and X2new given sep, Xnew comes in as a tuple
            Kxs = tt.slinalg.kron(self.cov_func1(self.X[0], Xnew[0]),
                                  self.cov_func2(self.X[1], Xnew[1]))
            Kss = tt.slinalg.kron(self.cov_func1(Xnew[0]),
                                  self.cov_func2(Xnew[1]))
            Xnew = grid_to_full(Xnew[0], Xnew[1], n_points[0], n_points[1])
            n_points = np.prod(n_points)
        else:
            # predict given full
            gridcov_func = make_gridcov_func(self.cov_func1, self.cov_func2)
            X = grid_to_full(self.X[0], self.X[1], self.n[0], self.n[1])
            Kxs = gridcov_func(X, Xnew)
            Kss = gridcov_func(Xnew)
            n_points = infer_shape(Xnew, n_points)
        A = kronsolve(L1, L2, Kxs, veced=False, chol=True)
        v = kronsolve(L1, L2, f, veced=False, chol=True) # f - mean_total(X)
        #mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        mu = tt.dot(tt.transpose(A), v)
        cov = Kss - tt.dot(tt.transpose(A), A)
        return mu, cov, n_points


@conditioned_vars(["X", "n", "f"])
class Grid2DMarginal(Grid):

    def _build_marginal_likelihood_logp(self, X, sigma, n):
        mu = 0.0
        K1 = stabilize(self.cov_func1(X[0]))
        K2 = stabilize(self.cov_func2(X[1]))
        if sigma is None: # use cholesky
            L1 = cholesky(stabilize(self.cov_func1(X[0])))
            L2 = cholesky(stabilize(self.cov_func2(X[1])))
            const = n[0] * n[1] * tt.log(2 * np.pi)
            logdet = (n[1] * 2.0 * tt.sum(tt.log(tt.diag(L1))) +
                      n[0] * 2.0 * tt.sum(tt.log(tt.diag(L2))))
            tmp = kronsolve(L1, L2, Y, veced=False, chol=True)
            quad = tt.sum(tt.square(tmp))
            return -0.5 * (logdet + quad + const)
        else: # use eigh
            S1, Q1 = tt.nlinalg.eigh(K1)
            S2, Q2 = tt.nlinalg.eigh(K2)
            W = np.kron(S2, S1) + tt.square(sigma)
            Qinvy = kronsolve(Q1, Q2, Y, veced=False, chol=False)
            const = n[0] * n[1] * tt.log(2.0 * np.pi)
            logdet = tt.sum(tt.log(W))
            quad = tt.sum(tt.square(tt.sqrt(1.0 / W) * Qinvy))
            return -0.5 * (logdet + quad + const)


    def marginal_likelihood(self, name, X, sigma=None, n=None):
        if n is None:
            n = (None, None)
        if len(X) != 2 or len(n) != 2:
            raise ValueError("2d implemented")
        n = (infer_shape(X[0], n[0]), infer_shape(X[1], n[1]))
        logp = lambda Y: self._build_marginal_likelihood(X, Y, sigma, n)
        self.X = (tt.as_tensor_variable(X[0]), tt.as_tensor_variable(X[1]))
        self.n = n
        return f












