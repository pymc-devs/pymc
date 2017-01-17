from functools import reduce
import copy

import theano
import theano.tensor as tt
import numpy as np

__all__ = ["RBF", "White"]

"""
Modeled *very* closely after GPy and GPFlow so far
"""


class Covariance(object):
    def __init__(self, input_dim, active_dims=None):
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = np.arange(input_dim)
        else:
            self.active_dims = np.array(active_dims)
            assert len(active_dims) == input_dim

    def _slice(self, X, Z):
        X = X[:, self.active_dims]
        if Z is not None:
            Z = Z[:, self.active_dims]
        return X, Z

    def __add__(self, other):
        return Add([self, other])

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return Prod([self, other])

    def __rmul__(self, other):
        return self.__mul__(other)


class Combination(Covariance):
    def __init__(self, factor_list):
        input_dim = np.max([factor.input_dim for factor in
                            filter(lambda x: isinstance(x, Covariance), factor_list)])
        Covariance.__init__(self, input_dim=input_dim)
        self.factor_list = []
        for factor in factor_list:
            if isinstance(factor, self.__class__):
                self.factor_list.extend(factor.factor_list)
            else:
                self.factor_list.append(factor)

class Add(Combination):
    def K(self, X, Z=None):
        return reduce((lambda x, y: x + y),
                      [k.K(X, Z) if isinstance(k, Covariance) else k for k in self.factor_list])

class Prod(Combination):
    def K(self, X, Z=None):
        return reduce((lambda x, y: x * y),
                      [k.K(X, Z) if isinstance(k, Covariance) else k for k in self.factor_list])


class Stationary(Covariance):
    def __init__(self, input_dim, lengthscales, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        self.lengthscales = lengthscales

    def square_dist(self, X, Z):
        X = tt.mul(X, 1.0 / self.lengthscales)
        Xs = tt.sum(tt.square(X), 1)
        if Z is None:
            return -2.0 * tt.dot(X, tt.transpose(X)) +\
                   tt.reshape(Xs, (-1, 1)) + tt.reshape(Xs, (1, -1))
        else:
            Z = tt.mul(Z, 1.0 / self.lengthscales)
            Zs = tt.sum(tt.square(Z), 1)
            return -2.0 * tt.dot(X, tt.transpose(Z)) +\
                   tt.reshape(Xs, (-1, 1)) + tt.reshape(Zs, (1, -1))

    def euclidean_dist(self, X, Z):
        r2 = self.square_dist(X, Z)
        return tt.sqrt(r2 + 1e-12)


class ExpQuad(Stationary):
    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        return tt.exp( -0.5 * self.square_dist(X, Z))


class RatQuad(Stationary):
    def __init__(self, input_dim, lengthscales, alpha, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        self.lengthscales = lengthscales
        self.alpha = alpha

    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        return tt.power((1.0 + 0.5 * self.square_dist(X, Z) * (1.0 / self.alpha)), -1.0 * self.alpha)


class Matern52(Stationary):
    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        r = self.euclidean_dist(X, Z)
        return (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * tt.square(r)) * tt.exp(-1.0 * np.sqrt(5.0) * r)


class Matern32(Stationary):
    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        r = self.euclidean_dist(X, Z)
        return (1.0 + np.sqrt(3.0) * r) * tt.exp(-np.sqrt(3.0) * r)


class Exponential(Stationary):
    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        return tt.exp(-0.5 * self.euclidean_dist(X, Z))


class Cosine(Stationary):
    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        return tt.cos(self.euclidean_dist(X, Z))


class Linear(Covariance):
    def __init__(self, input_dim, slopes, centers, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        self.centers = centers
        self.slopes  = slopes

    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        # need to fix this with slope and center
        Xc = tt.sub(tt.mul(self.slopes, X), self.centers)
        if Z is None:
            return tt.dot(Xc, tt.transpose(Xc))
        else:
            Zc = tt.sub(Z, self.centers)
            return tt.dot(Xc, tt.transpose(Zc))

class Gibbs(Covariance):
    def __init__(self, input_dim, lengthscale_func, args=None, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        assert callable(lengthscale_func), "Must be a function"
        self.l = handle_args(lengthscale_func, args)
        self.args = args

    def square_dist(self, X, Z):
        # smallish? problem in here
        X = tt.mul(X, 1.0 / self.l(X, self.args))
        Xs = tt.sum(tt.square(X), 1)
        if Z is None:
            return -2.0 * tt.dot(X, tt.transpose(X)) +\
                   tt.reshape(Xs, (-1, 1)) + tt.reshape(Xs, (1, -1))
        else:
            Z = tt.mul(Z, 1.0 / self.l(Z, self.args))
            Zs = tt.sum(tt.square(Z), 1)
            return -2.0 * tt.dot(X, tt.transpose(Z)) +\
                   tt.reshape(Xs, (-1, 1)) + tt.reshape(Zs, (1, -1))

    def euclidean_dist(self, X, Z):
        r2 = self.square_dist(X, Z)
        return tt.sqrt(r2 + 1e-12)

    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        lx = self.l(X, self.args)
        if Z is None:
            lxp = lx
        else:
            lxp = self.l(Z, self.args)
        c = tt.prod(tt.sqrt( ((2.0 * lx * lxp) / (tt.square(lx) + tt.square(lxp))) + 1e-12))
        return c * tt.exp(-0.5 * self.square_dist(X, Z))


class WarpedInput(Covariance):
    def __init__(self, input_dim, cov_func, warp_func, args=None, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        assert callable(warp_func), "Must be a function"
        self.w = handle_args(warp_func, args)
        self.args = args
        self.cov_func = cov_func

    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        if Z is None:
            return self.cov_func.K(self.w(X, self.args), Z)
        else:
            return self.cov_func.K(self.w(X, self.args), self.w(Z, self.args))


class BasisFuncCov(Covariance):
    def __init__(self, input_dim, alpha, basis, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        # non functional, idealy user would suppy basis *function* from theano, so
        # that prediction could be done, not just a one off linear basis that can
        # only be used for fitting.  This requires e.g. spline functionality in theano
        self.alpha = alpha
        self.basis = basis

    def K(self, X, Z=None):
        X, Z = self._slice(X, Z)
        phi = tt.dot(self.basis, self.alpha)
        return tt.exp(tt.dot(phi, tt.transpose(phi)))

def handle_args(func, args):
    def func2(x, args):
        if args is None:
            return func(x)
        else:
            if not isinstance(args, tuple):
                args = (args,)
            return func(x, *args)
    return func2


