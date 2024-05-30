#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import warnings

from functools import reduce
from numbers import Number
from operator import add, mul

import numpy as np
import theano
import theano.tensor as tt

__all__ = [
    "Constant",
    "WhiteNoise",
    "ExpQuad",
    "RatQuad",
    "Exponential",
    "Matern52",
    "Matern32",
    "Linear",
    "Polynomial",
    "Cosine",
    "Periodic",
    "WarpedInput",
    "Gibbs",
    "Coregion",
    "ScaledCov",
    "Kron",
]


class Covariance:
    r"""
    Base class for all kernels/covariance functions.

    Parameters
    ----------
    input_dim: integer
        The number of input dimensions, or columns of X (or Xs)
        the kernel will operate on.
    active_dims: List of integers
        Indicate which dimension or column of X the covariance
        function operates on.
    """

    def __init__(self, input_dim, active_dims=None):
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = np.arange(input_dim)
        else:
            self.active_dims = np.asarray(active_dims, int)

    def __call__(self, X, Xs=None, diag=False):
        r"""
        Evaluate the kernel/covariance function.

        Parameters
        ----------
        X: The training inputs to the kernel.
        Xs: The optional prediction set of inputs the kernel.
            If Xs is None, Xs = X.
        diag: bool
            Return only the diagonal of the covariance function.
            Default is False.
        """
        if diag:
            return self.diag(X)
        else:
            return self.full(X, Xs)

    def diag(self, X):
        raise NotImplementedError

    def full(self, X, Xs):
        raise NotImplementedError

    def _slice(self, X, Xs):
        if self.input_dim != X.shape[-1]:
            warnings.warn(
                f"Only {self.input_dim} column(s) out of {X.shape[-1]} are"
                " being used to compute the covariance function. If this"
                " is not intended, increase 'input_dim' parameter to"
                " the number of columns to use. Ignore otherwise.",
                UserWarning,
            )
        X = tt.as_tensor_variable(X[:, self.active_dims])
        if Xs is not None:
            Xs = tt.as_tensor_variable(Xs[:, self.active_dims])
        return X, Xs

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        if (
            isinstance(other, theano.compile.SharedVariable)
            and other.get_value().squeeze().shape == ()
        ):
            other = tt.squeeze(other)
            return Exponentiated(self, other)
        elif isinstance(other, Number):
            return Exponentiated(self, other)
        elif np.asarray(other).squeeze().shape == ():
            other = np.squeeze(other)
            return Exponentiated(self, other)

        raise ValueError("A covariance function can only be exponentiated by a scalar value")

    def __array_wrap__(self, result):
        """
        Required to allow radd/rmul by numpy arrays.
        """
        result = np.squeeze(result)
        if len(result.shape) <= 1:
            result = result.reshape(1, 1)
        elif len(result.shape) > 2:
            raise ValueError(
                f"cannot combine a covariance function with array of shape {result.shape}"
            )
        r, c = result.shape
        A = np.zeros((r, c))
        for i in range(r):
            for j in range(c):
                A[i, j] = result[i, j].factor_list[1]
        if isinstance(result[0][0], Add):
            return result[0][0].factor_list[0] + A
        elif isinstance(result[0][0], Prod):
            return result[0][0].factor_list[0] * A
        else:
            raise RuntimeError


class Combination(Covariance):
    def __init__(self, factor_list):
        input_dim = max(
            factor.input_dim for factor in factor_list if isinstance(factor, Covariance)
        )
        super().__init__(input_dim=input_dim)
        self.factor_list = []
        for factor in factor_list:
            if isinstance(factor, self.__class__):
                self.factor_list.extend(factor.factor_list)
            else:
                self.factor_list.append(factor)

    def merge_factors(self, X, Xs=None, diag=False):
        factor_list = []
        for factor in self.factor_list:
            # make sure diag=True is handled properly
            if isinstance(factor, Covariance):
                factor_list.append(factor(X, Xs, diag))
            elif isinstance(factor, np.ndarray):
                if np.ndim(factor) == 2 and diag:
                    factor_list.append(np.diag(factor))
                else:
                    factor_list.append(factor)
            elif isinstance(
                factor,
                (
                    tt.TensorConstant,
                    tt.TensorVariable,
                    tt.sharedvar.TensorSharedVariable,
                ),
            ):
                if factor.ndim == 2 and diag:
                    factor_list.append(tt.diag(factor))
                else:
                    factor_list.append(factor)
            else:
                factor_list.append(factor)
        return factor_list


class Add(Combination):
    def __call__(self, X, Xs=None, diag=False):
        return reduce(add, self.merge_factors(X, Xs, diag))


class Prod(Combination):
    def __call__(self, X, Xs=None, diag=False):
        return reduce(mul, self.merge_factors(X, Xs, diag))


class Exponentiated(Covariance):
    def __init__(self, kernel, power):
        self.kernel = kernel
        self.power = power
        super().__init__(input_dim=self.kernel.input_dim, active_dims=self.kernel.active_dims)

    def __call__(self, X, Xs=None, diag=False):
        return self.kernel(X, Xs, diag=diag) ** self.power


class Kron(Covariance):
    r"""Form a covariance object that is the kronecker product of other covariances.

    In contrast to standard multiplication, where each covariance is given the
    same inputs X and Xs, kronecker product covariances first split the inputs
    into their respective spaces (inferred from the input_dim of each object)
    before forming their product. Kronecker covariances have a larger
    input dimension than any of its factors since the inputs are the
    concatenated columns of its components.

    Factors must be covariances or their combinations, arrays will not work.

    Generally utilized by the `gp.MarginalKron` and gp.LatentKron`
    implementations.
    """

    def __init__(self, factor_list):
        self.input_dims = [factor.input_dim for factor in factor_list]
        input_dim = sum(self.input_dims)
        super().__init__(input_dim=input_dim)
        self.factor_list = factor_list

    def _split(self, X, Xs):
        indices = np.cumsum(self.input_dims)
        X_split = np.hsplit(X, indices)
        if Xs is not None:
            Xs_split = np.hsplit(Xs, indices)
        else:
            Xs_split = [None] * len(X_split)
        return X_split, Xs_split

    def __call__(self, X, Xs=None, diag=False):
        X_split, Xs_split = self._split(X, Xs)
        covs = [cov(x, xs, diag) for cov, x, xs in zip(self.factor_list, X_split, Xs_split)]
        return reduce(mul, covs)


class Constant(Covariance):
    r"""
    Constant valued covariance function.

    .. math::

       k(x, x') = c
    """

    def __init__(self, c):
        super().__init__(1, None)
        self.c = c

    def diag(self, X):
        return tt.alloc(self.c, X.shape[0])

    def full(self, X, Xs=None):
        if Xs is None:
            return tt.alloc(self.c, X.shape[0], X.shape[0])
        else:
            return tt.alloc(self.c, X.shape[0], Xs.shape[0])


class WhiteNoise(Covariance):
    r"""
    White noise covariance function.

    .. math::

       k(x, x') = \sigma^2 \mathrm{I}
    """

    def __init__(self, sigma):
        super().__init__(1, None)
        self.sigma = sigma

    def diag(self, X):
        return tt.alloc(tt.square(self.sigma), X.shape[0])

    def full(self, X, Xs=None):
        if Xs is None:
            return tt.diag(self.diag(X))
        else:
            return tt.alloc(0.0, X.shape[0], Xs.shape[0])


class Circular(Covariance):
    R"""
    Circular Kernel.

    .. math::

        k_g(x, y) = W_\pi(\operatorname{dist}_{\mathit{geo}}(x, y)),

    with

    .. math::

        W_c(t) = \left(1 + \tau \frac{t}{c}\right)\left(1-\frac{t}{c}\right)^\tau_+

    where :math:`c` is maximum value for :math:`t` and :math:`\tau\ge 4`.
    :math:`\tau` controls for correlation strength, larger :math:`\tau` leads to less smooth functions
    See [1]_ for more explanations and use cases.

    Parameters
    ----------
    period : scalar
        defines the circular interval :math:`[0, \mathit{bound})`
    tau : scalar
        :math:`\tau\ge 4` defines correlation strength, the larger,
        the smaller correlation is. Minimum value is :math:`4`

    References
    ----------
    .. [1] Espéran Padonou, O Roustant, "Polar Gaussian Processes for Predicting on Circular Domains"
    https://hal.archives-ouvertes.fr/hal-01119942v1/document
    """

    def __init__(self, input_dim, period, tau=4, active_dims=None):
        super().__init__(input_dim, active_dims)
        self.c = tt.as_tensor_variable(period / 2)
        self.tau = tau

    def dist(self, X, Xs):
        if Xs is None:
            Xs = tt.transpose(X)
        else:
            Xs = tt.transpose(Xs)
        return tt.abs_((X - Xs + self.c) % (self.c * 2) - self.c)

    def weinland(self, t):
        return (1 + self.tau * t / self.c) * tt.clip(1 - t / self.c, 0, np.inf) ** self.tau

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return self.weinland(self.dist(X, Xs))

    def diag(self, X):
        return tt.alloc(1.0, X.shape[0])


class Stationary(Covariance):
    r"""
    Base class for stationary kernels/covariance functions.

    Parameters
    ----------
    ls: Lengthscale.  If input_dim > 1, a list or array of scalars or PyMC3 random
    variables.  If input_dim == 1, a scalar or PyMC3 random variable.
    ls_inv: Inverse lengthscale.  1 / ls.  One of ls or ls_inv must be provided.
    """

    def __init__(self, input_dim, ls=None, ls_inv=None, active_dims=None):
        super().__init__(input_dim, active_dims)
        if (ls is None and ls_inv is None) or (ls is not None and ls_inv is not None):
            raise ValueError("Only one of 'ls' or 'ls_inv' must be provided")
        elif ls_inv is not None:
            if isinstance(ls_inv, (list, tuple)):
                ls = 1.0 / np.asarray(ls_inv)
            else:
                ls = 1.0 / ls_inv
        self.ls = tt.as_tensor_variable(ls)

    def square_dist(self, X, Xs):
        X = tt.mul(X, 1.0 / self.ls)
        X2 = tt.sum(tt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(X2, (1, -1))
            )
        else:
            Xs = tt.mul(Xs, 1.0 / self.ls)
            Xs2 = tt.sum(tt.square(Xs), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Xs)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(Xs2, (1, -1))
            )
        return tt.clip(sqd, 0.0, np.inf)

    def euclidean_dist(self, X, Xs):
        r2 = self.square_dist(X, Xs)
        return tt.sqrt(r2 + 1e-12)

    def diag(self, X):
        return tt.alloc(1.0, X.shape[0])

    def full(self, X, Xs=None):
        raise NotImplementedError


class Periodic(Stationary):
    r"""
    The Periodic kernel.

    .. math::
       k(x, x') = \mathrm{exp}\left( -\frac{\mathrm{sin}^2(\pi |x-x'| \frac{1}{T})}{2\ell^2} \right)

    Notes
    -----
    Note that the scaling factor for this kernel is different compared to the more common
    definition (see [1]_). Here, 0.5 is in the exponent instead of the more common value, 2.
    Divide the length-scale by 2 when initializing the kernel to recover the standard definition.

    References
    ----------
    .. [1] David Duvenaud, "The Kernel Cookbook"
       https://www.cs.toronto.edu/~duvenaud/cookbook/
    """

    def __init__(self, input_dim, period, ls=None, ls_inv=None, active_dims=None):
        super().__init__(input_dim, ls, ls_inv, active_dims)
        self.period = period

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        if Xs is None:
            Xs = X
        f1 = X.dimshuffle(0, "x", 1)
        f2 = Xs.dimshuffle("x", 0, 1)
        r = np.pi * (f1 - f2) / self.period
        r = tt.sum(tt.square(tt.sin(r) / self.ls), 2)
        return tt.exp(-0.5 * r)


class ExpQuad(Stationary):
    r"""
    The Exponentiated Quadratic kernel.  Also refered to as the Squared
    Exponential, or Radial Basis Function kernel.

    .. math::

       k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{2 \ell^2} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return tt.exp(-0.5 * self.square_dist(X, Xs))


class RatQuad(Stationary):
    r"""
    The Rational Quadratic kernel.

    .. math::

       k(x, x') = \left(1 + \frac{(x - x')^2}{2\alpha\ell^2} \right)^{-\alpha}
    """

    def __init__(self, input_dim, alpha, ls=None, ls_inv=None, active_dims=None):
        super().__init__(input_dim, ls, ls_inv, active_dims)
        self.alpha = alpha

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return tt.power(
            (1.0 + 0.5 * self.square_dist(X, Xs) * (1.0 / self.alpha)),
            -1.0 * self.alpha,
        )


class Matern52(Stationary):
    r"""
    The Matern kernel with nu = 5/2.

    .. math::

       k(x, x') = \left(1 + \frac{\sqrt{5(x - x')^2}}{\ell} +
                   \frac{5(x-x')^2}{3\ell^2}\right)
                   \mathrm{exp}\left[ - \frac{\sqrt{5(x - x')^2}}{\ell} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.euclidean_dist(X, Xs)
        return (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * tt.square(r)) * tt.exp(-1.0 * np.sqrt(5.0) * r)


class Matern32(Stationary):
    r"""
    The Matern kernel with nu = 3/2.

    .. math::

       k(x, x') = \left(1 + \frac{\sqrt{3(x - x')^2}}{\ell}\right)
                  \mathrm{exp}\left[ - \frac{\sqrt{3(x - x')^2}}{\ell} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.euclidean_dist(X, Xs)
        return (1.0 + np.sqrt(3.0) * r) * tt.exp(-np.sqrt(3.0) * r)


class Matern12(Stationary):
    r"""
    The Matern kernel with nu = 1/2

    k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{\ell} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        r = self.euclidean_dist(X, Xs)
        return tt.exp(-r)


class Exponential(Stationary):
    r"""
    The Exponential kernel.

    .. math::

       k(x, x') = \mathrm{exp}\left[ -\frac{||x - x'||}{2\ell^2} \right]
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return tt.exp(-0.5 * self.euclidean_dist(X, Xs))


class Cosine(Stationary):
    r"""
    The Cosine kernel.

    .. math::
       k(x, x') = \mathrm{cos}\left( 2 \pi \frac{||x - x'||}{ \ell^2} \right)
    """

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        return tt.cos(2.0 * np.pi * self.euclidean_dist(X, Xs))


class Linear(Covariance):
    r"""
    The Linear kernel.

    .. math::
       k(x, x') = (x - c)(x' - c)
    """

    def __init__(self, input_dim, c, active_dims=None):
        super().__init__(input_dim, active_dims)
        self.c = c

    def _common(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        Xc = tt.sub(X, self.c)
        return X, Xc, Xs

    def full(self, X, Xs=None):
        X, Xc, Xs = self._common(X, Xs)
        if Xs is None:
            return tt.dot(Xc, tt.transpose(Xc))
        else:
            Xsc = tt.sub(Xs, self.c)
            return tt.dot(Xc, tt.transpose(Xsc))

    def diag(self, X):
        X, Xc, _ = self._common(X, None)
        return tt.sum(tt.square(Xc), 1)


class Polynomial(Linear):
    r"""
    The Polynomial kernel.

    .. math::
       k(x, x') = [(x - c)(x' - c) + \mathrm{offset}]^{d}
    """

    def __init__(self, input_dim, c, d, offset, active_dims=None):
        super().__init__(input_dim, c, active_dims)
        self.d = d
        self.offset = offset

    def full(self, X, Xs=None):
        linear = super().full(X, Xs)
        return tt.power(linear + self.offset, self.d)

    def diag(self, X):
        linear = super().diag(X)
        return tt.power(linear + self.offset, self.d)


class WarpedInput(Covariance):
    r"""
    Warp the inputs of any kernel using an arbitrary function
    defined using Theano.

    .. math::
       k(x, x') = k(w(x), w(x'))

    Parameters
    ----------
    cov_func: Covariance
    warp_func: callable
        Theano function of X and additional optional arguments.
    args: optional, tuple or list of scalars or PyMC3 variables
        Additional inputs (besides X or Xs) to warp_func.
    """

    def __init__(self, input_dim, cov_func, warp_func, args=None, active_dims=None):
        super().__init__(input_dim, active_dims)
        if not callable(warp_func):
            raise TypeError("warp_func must be callable")
        if not isinstance(cov_func, Covariance):
            raise TypeError("Must be or inherit from the Covariance class")
        self.w = handle_args(warp_func, args)
        self.args = args
        self.cov_func = cov_func

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        if Xs is None:
            return self.cov_func(self.w(X, self.args), Xs)
        else:
            return self.cov_func(self.w(X, self.args), self.w(Xs, self.args))

    def diag(self, X):
        X, _ = self._slice(X, None)
        return self.cov_func(self.w(X, self.args), diag=True)


class Gibbs(Covariance):
    r"""
    The Gibbs kernel.  Use an arbitrary lengthscale function defined
    using Theano.  Only tested in one dimension.

    .. math::
       k(x, x') = \sqrt{\frac{2\ell(x)\ell(x')}{\ell^2(x) + \ell^2(x')}}
                  \mathrm{exp}\left[ -\frac{(x - x')^2}
                                           {\ell^2(x) + \ell^2(x')} \right]

    Parameters
    ----------
    lengthscale_func: callable
        Theano function of X and additional optional arguments.
    args: optional, tuple or list of scalars or PyMC3 variables
        Additional inputs (besides X or Xs) to lengthscale_func.
    """

    def __init__(self, input_dim, lengthscale_func, args=None, active_dims=None):
        super().__init__(input_dim, active_dims)
        if active_dims is not None:
            if len(active_dims) > 1:
                raise NotImplementedError(("Higher dimensional inputs ", "are untested"))
        else:
            if input_dim != 1:
                raise NotImplementedError(("Higher dimensional inputs ", "are untested"))
        if not callable(lengthscale_func):
            raise TypeError("lengthscale_func must be callable")
        self.lfunc = handle_args(lengthscale_func, args)
        self.args = args

    def square_dist(self, X, Xs=None):
        X2 = tt.sum(tt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * tt.dot(X, tt.transpose(X)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(X2, (1, -1))
            )
        else:
            Xs2 = tt.sum(tt.square(Xs), 1)
            sqd = -2.0 * tt.dot(X, tt.transpose(Xs)) + (
                tt.reshape(X2, (-1, 1)) + tt.reshape(Xs2, (1, -1))
            )
        return tt.clip(sqd, 0.0, np.inf)

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        rx = self.lfunc(tt.as_tensor_variable(X), self.args)
        if Xs is None:
            rz = self.lfunc(tt.as_tensor_variable(X), self.args)
            r2 = self.square_dist(X, X)
        else:
            rz = self.lfunc(tt.as_tensor_variable(Xs), self.args)
            r2 = self.square_dist(X, Xs)
        rx2 = tt.reshape(tt.square(rx), (-1, 1))
        rz2 = tt.reshape(tt.square(rz), (1, -1))
        return tt.sqrt((2.0 * tt.outer(rx, rz)) / (rx2 + rz2)) * tt.exp(-1.0 * r2 / (rx2 + rz2))

    def diag(self, X):
        return tt.alloc(1.0, X.shape[0])


class ScaledCov(Covariance):
    r"""
    Construct a kernel by multiplying a base kernel with a scaling
    function defined using Theano.  The scaling function is
    non-negative, and can be parameterized.

    .. math::
       k(x, x') = \phi(x) k_{\text{base}}(x, x') \phi(x')

    Parameters
    ----------
    cov_func: Covariance
        Base kernel or covariance function
    scaling_func: callable
        Theano function of X and additional optional arguments.
    args: optional, tuple or list of scalars or PyMC3 variables
        Additional inputs (besides X or Xs) to lengthscale_func.
    """

    def __init__(self, input_dim, cov_func, scaling_func, args=None, active_dims=None):
        super().__init__(input_dim, active_dims)
        if not callable(scaling_func):
            raise TypeError("scaling_func must be callable")
        if not isinstance(cov_func, Covariance):
            raise TypeError("Must be or inherit from the Covariance class")
        self.cov_func = cov_func
        self.scaling_func = handle_args(scaling_func, args)
        self.args = args

    def diag(self, X):
        X, _ = self._slice(X, None)
        cov_diag = self.cov_func(X, diag=True)
        scf_diag = tt.square(tt.flatten(self.scaling_func(X, self.args)))
        return cov_diag * scf_diag

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        scf_x = self.scaling_func(X, self.args)
        if Xs is None:
            return tt.outer(scf_x, scf_x) * self.cov_func(X)
        else:
            scf_xs = self.scaling_func(Xs, self.args)
            return tt.outer(scf_x, scf_xs) * self.cov_func(X, Xs)


class Coregion(Covariance):
    r"""Covariance function for intrinsic/linear coregionalization models.
    Adapted from GPy http://gpy.readthedocs.io/en/deploy/GPy.kern.src.html#GPy.kern.src.coregionalize.Coregionalize.

    This covariance has the form:

    .. math::

       \mathbf{B} = \mathbf{W}\mathbf{W}^\top + \text{diag}(\kappa)

    and calls must use integers associated with the index of the matrix.
    This allows the api to remain consistent with other covariance objects:

    .. math::

        k(x, x') = \mathbf{B}[x, x'^\top]

    Parameters
    ----------
    W: 2D array of shape (num_outputs, rank)
        a low rank matrix that determines the correlations between
        the different outputs (rows)
    kappa: 1D array of shape (num_outputs, )
        a vector which allows the outputs to behave independently
    B: 2D array of shape (num_outputs, rank)
        the total matrix, exactly one of (W, kappa) and B must be provided

    Notes
    -----
    Exactly one dimension must be active for this kernel. Thus, if
    `input_dim != 1`, then `active_dims` must have a length of one.
    """

    def __init__(self, input_dim, W=None, kappa=None, B=None, active_dims=None):
        super().__init__(input_dim, active_dims)
        if len(self.active_dims) != 1:
            raise ValueError("Coregion requires exactly one dimension to be active")
        make_B = W is not None or kappa is not None
        if make_B and B is not None:
            raise ValueError("Exactly one of (W, kappa) and B must be provided to Coregion")
        if make_B:
            self.W = tt.as_tensor_variable(W)
            self.kappa = tt.as_tensor_variable(kappa)
            self.B = tt.dot(self.W, self.W.T) + tt.diag(self.kappa)
        elif B is not None:
            self.B = tt.as_tensor_variable(B)
        else:
            raise ValueError("Exactly one of (W, kappa) and B must be provided to Coregion")

    def full(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        index = tt.cast(X, "int32")
        if Xs is None:
            index2 = index.T
        else:
            index2 = tt.cast(Xs, "int32").T
        return self.B[index, index2]

    def diag(self, X):
        X, _ = self._slice(X, None)
        index = tt.cast(X, "int32")
        return tt.diag(self.B)[index.ravel()]


def handle_args(func, args):
    def f(x, args):
        if args is None:
            return func(x)
        else:
            if not isinstance(args, tuple):
                args = (args,)
            return func(x, *args)

    return f
