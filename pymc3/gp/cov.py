import theano.tensor as tt
import numpy as np
from functools import reduce

__all__ = ['ExpQuad',
           'RatQuad',
           'Exponential',
           'Matern52',
           'Matern32',
           'Linear',
           'Polynomial',
           'Cosine',
           'WarpedInput',
           'Gibbs']

class Covariance(object):
    R"""
    Base class for all kernels/covariance functions.

    Parameters
    ----------
    input_dim : integer
        The number of input dimensions, or columns of X (or Z) the kernel will operate on.
    active_dims : A list of booleans whose length equals input_dim.
	The booleans indicate whether or not the covariance function operates
        over that dimension or column of X.
    """

    def __init__(self, input_dim, active_dims=None):
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = np.arange(input_dim)
        else:
            self.active_dims = np.array(active_dims)
            if len(active_dims) != input_dim:
                raise ValueError("Length of active_dims must match input_dim")

    def __call__(self, X, Z):
        R"""
        Evaluate the kernel/covariance function.

        Parameters
        ----------
        X : The training inputs to the kernel.
        Z : The optional prediction set of inputs the kernel.  If Z is None, Z = X.
        """
        raise NotImplementedError

    def _slice(self, X, Z):
        X = X[:, self.active_dims]
        if Z is not None:
            Z = Z[:, self.active_dims]
        return X, Z

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __array_wrap__(self, result):
        """
        Required to allow radd/rmul by numpy arrays.
        """
        r,c = result.shape
        A = np.zeros((r,c))
        for i in range(r):
            for j in range(c):
                A[i,j] = result[i,j].factor_list[1]
        if isinstance(result[0][0], Add):
            return result[0][0].factor_list[0] + A
        elif isinstance(result[0][0], Prod):
            return result[0][0].factor_list[0] * A
        else:
            raise RuntimeError


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
    def __call__(self, X, Z=None):
        return reduce((lambda x, y: x + y),
                      [k(X, Z) if isinstance(k, Covariance) else k for k in self.factor_list])


class Prod(Combination):
    def __call__(self, X, Z=None):
        return reduce((lambda x, y: x * y),
                      [k(X, Z) if isinstance(k, Covariance) else k for k in self.factor_list])


class Stationary(Covariance):
    R"""
    Base class for stationary kernels/covariance functions.

    Parameters
    ----------
    lengthscales: If input_dim > 1, a list or array of scalars or PyMC3 random
    variables.  If input_dim == 1, a scalar or PyMC3 random variable.
    """

    def __init__(self, input_dim, lengthscales, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        if isinstance(lengthscales, (list, tuple)):
            lengthscales = np.array(lengthscales)
        self.lengthscales = lengthscales

    def square_dist(self, X, Z):
        X = tt.mul(X, 1.0 / self.lengthscales)
        Xs = tt.sum(tt.square(X), 1)
        if Z is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) +\
                  (tt.reshape(Xs, (-1, 1)) + tt.reshape(Xs, (1, -1)))
        else:
            Z = tt.mul(Z, 1.0 / self.lengthscales)
            Zs = tt.sum(tt.square(Z), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Z)) +\
                  (tt.reshape(Xs, (-1, 1)) + tt.reshape(Zs, (1, -1)))
        return tt.clip(sqd, 0.0, np.inf)

    def euclidean_dist(self, X, Z):
        r2 = self.square_dist(X, Z)
        return tt.sqrt(r2 + 1e-12)


class ExpQuad(Stationary):
    R"""
    The Exponentiated Quadratic kernel.  Also refered to as the Squared
    Exponential, or Radial Basis Function kernel.

    .. math::

       k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{2 \ell^2} \right]
    """

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        return tt.exp( -0.5 * self.square_dist(X, Z))


class RatQuad(Stationary):
    R"""
    The Rational Quadratic kernel.

    .. math::

       k(x, x') = \left(1 + \frac{(x - x')^2}{2\alpha\ell^2} \right)^{-\alpha}
    """

    def __init__(self, input_dim, lengthscales, alpha, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        self.lengthscales = lengthscales
        self.alpha = alpha

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        return tt.power((1.0 + 0.5 * self.square_dist(X, Z) * (1.0 / self.alpha)), -1.0 * self.alpha)


class Matern52(Stationary):
    R"""
    The Matern kernel with nu = 5/2.

    .. math::

       k(x, x') = \left(1 + \frac{\sqrt{5(x - x')^2}}{\ell} + \frac{5(x-x')^2}{3\ell^2}\right) \mathrm{exp}\left[ - \frac{\sqrt{5(x - x')^2}}{\ell} \right]
    """

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        r = self.euclidean_dist(X, Z)
        return (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * tt.square(r)) * tt.exp(-1.0 * np.sqrt(5.0) * r)


class Matern32(Stationary):
    R"""
    The Matern kernel with nu = 3/2.

    .. math::

       k(x, x') = \left(1 + \frac{\sqrt{3(x - x')^2}}{\ell}\right)\mathrm{exp}\left[ - \frac{\sqrt{3(x - x')^2}}{\ell} \right]
    """

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        r = self.euclidean_dist(X, Z)
        return (1.0 + np.sqrt(3.0) * r) * tt.exp(-np.sqrt(3.0) * r)


class Exponential(Stationary):
    R"""
    The Exponential kernel.

    .. math::

       k(x, x') = \mathrm{exp}\left[ -\frac{||x - x'||}{2\ell^2} \right]
    """

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        return tt.exp(-0.5 * self.euclidean_dist(X, Z))


class Cosine(Stationary):
    R"""
    The Cosine kernel.

    .. math::
       k(x, x') = \mathrm{cos}\left( \frac{||x - x'||}{ \ell^2} \right)
    """

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        return tt.cos(np.pi * self.euclidean_dist(X, Z))


class Linear(Covariance):
    R"""
    The Linear kernel.

    .. math::
       k(x, x') = (x - c)(x' - c)
    """

    def __init__(self, input_dim, c, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        self.c = c

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        Xc = tt.sub(X, self.c)
        if Z is None:
            return tt.dot(Xc, tt.transpose(Xc))
        else:
            Zc = tt.sub(Z, self.c)
            return tt.dot(Xc, tt.transpose(Zc))


class Polynomial(Linear):
    R"""
    The Polynomial kernel.

    .. math::
       k(x, x') = [(x - c)(x' - c) + \mathrm{offset}]^{d}
    """

    def __init__(self, input_dim, c, d, offset, active_dims=None):
        Linear.__init__(self, input_dim, c, active_dims)
        self.d = d
        self.offset = offset

    def __call__(self, X, Z=None):
        linear = super(Polynomial, self).__call__(X, Z)
        return tt.power(linear + self.offset, self.d)


class WarpedInput(Covariance):
    R"""
    Warp the inputs of any kernel using an arbitrary function
    defined using Theano.

    .. math::
       k(x, x') = k(w(x), w(x'))

    Parameters
    ----------
    cov_func : Covariance
    warp_func : callable
        Theano function of X and additional optional arguments.
    args : optional, tuple or list of scalars or PyMC3 variables
        Additional inputs (besides X or Z) to warp_func.
    """

    def __init__(self, input_dim, cov_func, warp_func, args=None, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        if not callable(warp_func):
            raise TypeError("warp_func must be callable")
        if not isinstance(cov_func, Covariance):
            raise TypeError("Must be or inherit from the Covariance class")
        self.w = handle_args(warp_func, args)
        self.args = args
        self.cov_func = cov_func

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        if Z is None:
            return self.cov_func(self.w(X, self.args), Z)
        else:
            return self.cov_func(self.w(X, self.args), self.w(Z, self.args))


class Gibbs(Covariance):
    R"""
    The Gibbs kernel.  Use an arbitrary lengthscale function defined
    using Theano.  Only tested in one dimension.

    .. math::
       k(x, x') = \sqrt{\frac{2\ell(x)\ell(x')}{\ell^2(x) + \ell^2(x')}}
                  \mathrm{exp}\left[ -\frac{(x - x')^2}{\ell^2(x) + \ell^2(x')} \right]

    Parameters
    ----------
    lengthscale_func : callable
        Theano function of X and additional optional arguments.
    args : optional, tuple or list of scalars or PyMC3 variables
        Additional inputs (besides X or Z) to lengthscale_func.
    """
    def __init__(self, input_dim, lengthscale_func, args=None, active_dims=None):
        Covariance.__init__(self, input_dim, active_dims)
        if active_dims is not None:
            if input_dim != 1 or sum(active_dims) == 1:
                raise NotImplementedError("Higher dimensional inputs are untested")
        else:
            if input_dim != 1:
                raise NotImplementedError("Higher dimensional inputs are untested")
        if not callable(lengthscale_func):
            raise TypeError("lengthscale_func must be callable")
        self.ell = handle_args(lengthscale_func, args)
        self.args = args

    def square_dist(self, X, Z):
        X = tt.as_tensor_variable(X)
        Xs = tt.sum(tt.square(X), 1)
        if Z is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) +\
                  (tt.reshape(Xs, (-1, 1)) + tt.reshape(Xs, (1, -1)))
        else:
            Z = tt.as_tensor_variable(Z)
            Zs = tt.sum(tt.square(Z), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Z)) +\
                  (tt.reshape(Xs, (-1, 1)) + tt.reshape(Zs, (1, -1)))
        return tt.clip(sqd, 0.0, np.inf)

    def __call__(self, X, Z=None):
        X, Z = self._slice(X, Z)
        rx = self.ell(X, self.args)
        rx2 = tt.reshape(tt.square(rx), (-1, 1))
        if Z is None:
            r2 = self.square_dist(X,X)
            rz = self.ell(X, self.args)
        else:
            r2 = self.square_dist(X,Z)
            rz = self.ell(Z, self.args)
        rz2 = tt.reshape(tt.square(rz), (1, -1))
        return tt.sqrt((2.0 * tt.dot(rx, tt.transpose(rz))) / (rx2 + rz2)) *\
               tt.exp(-1.0 * r2 / (rx2 + rz2))


def handle_args(func, args):
    def f(x, args):
        if args is None:
            return func(x)
        else:
            if not isinstance(args, tuple):
                args = (args,)
            return func(x, *args)
    return f
