#   Copyright 2024 - present The PyMC Developers
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

import numbers
import warnings

from collections import Counter
from collections.abc import Callable, Sequence
from functools import reduce
from operator import add, mul
from typing import Any

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import Variable
from pytensor.tensor.sharedvar import TensorSharedVariable
from pytensor.tensor.variable import TensorConstant, TensorVariable

__all__ = [
    "Constant",
    "Coregion",
    "Cosine",
    "ExpQuad",
    "Exponential",
    "Gibbs",
    "Kron",
    "Linear",
    "Matern12",
    "Matern32",
    "Matern52",
    "Periodic",
    "Polynomial",
    "RatQuad",
    "ScaledCov",
    "WarpedInput",
    "WhiteNoise",
    "WrappedPeriodic",
]

from pymc.pytensorf import constant_fold

TensorLike = np.ndarray | TensorVariable
IntSequence = np.ndarray | Sequence[int]


class BaseCovariance:
    """Base class for kernels/covariance functions."""

    def __call__(
        self,
        X: TensorLike,
        Xs: TensorLike | None = None,
        diag: bool = False,
    ) -> TensorVariable:
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

    def diag(self, X: TensorLike) -> TensorVariable:
        raise NotImplementedError

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        raise NotImplementedError

    def __add__(self, other) -> "Add":
        # If it's a scalar, cast as Constant covariance.  This allows validation for power spectral
        # density calc.
        if isinstance(other, numbers.Real):
            other = Constant(c=other)
        return Add([self, other])

    def __mul__(self, other) -> "Prod":
        return Prod([self, other])

    def __radd__(self, other) -> "Add":
        return self.__add__(other)

    def __rmul__(self, other) -> "Prod":
        return self.__mul__(other)

    def __pow__(self, other) -> "Exponentiated":
        other = pt.as_tensor_variable(other).squeeze()
        if not other.ndim == 0:
            raise ValueError("A covariance function can only be exponentiated by a scalar value")
        if not isinstance(self, Covariance):
            raise TypeError(
                "Can only exponentiate covariance functions which inherit from `Covariance`"
            )
        return Exponentiated(self, other)

    def __array_wrap__(self, result):
        """Allow radd/rmul by numpy arrays."""
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
                r = result[i, j]._factor_list[1]
                if isinstance(r, Constant):
                    # Counteract the elemwise Add edgecase
                    r = r.c
                A[i, j] = r
        if isinstance(result[0][0], Add):
            return result[0][0]._factor_list[0] + A
        elif isinstance(result[0][0], Prod):
            return result[0][0]._factor_list[0] * A
        else:
            raise TypeError(
                f"Unknown Covariance combination type {result[0][0]}.  "
                "Known types are `Add` or `Prod`."
            )

    @staticmethod
    def _alloc(X, *shape: int) -> TensorVariable:
        return pt.alloc(X, *shape)  # type: ignore[return-value]


class Covariance(BaseCovariance):
    """
    Base class for kernels/covariance functions with input_dim and active_dims.

    This excludes kernels like `Constant` and `WhiteNoise`.

    Parameters
    ----------
    input_dim: integer
        The number of input dimensions, or columns of X (or Xs)
        the kernel will operate on.
    active_dims: List of integers
        Indicate which dimension or column of X the covariance
        function operates on.
    """

    def __init__(self, input_dim: int, active_dims: IntSequence | None = None):
        self.input_dim = input_dim
        self.active_dims: np.ndarray[Any, np.dtype[np.int_]]
        if active_dims is None:
            self.active_dims = np.arange(input_dim)
        else:
            self.active_dims = np.asarray(active_dims, int)

        if self.active_dims.max() > self.input_dim:
            raise ValueError("Values in `active_dims` can't be larger than `input_dim`.")

    @property
    def n_dims(self) -> int:
        """The dimensionality of the input, as taken from the `active_dims`."""
        # Evaluate lazily in case this changes.
        return len(self.active_dims)

    def _slice(self, X, Xs=None):
        xdims = X.shape[-1]
        if isinstance(xdims, Variable):
            [xdims] = constant_fold([xdims], raise_not_constant=False)
        # If xdims can't be fully constant folded, it will be a TensorVariable
        if isinstance(xdims, int) and self.input_dim != xdims:
            warnings.warn(
                f"Only {self.input_dim} column(s) out of {xdims} are"
                " being used to compute the covariance function. If this"
                " is not intended, increase 'input_dim' parameter to"
                " the number of columns to use. Ignore otherwise.",
                UserWarning,
            )
        X = pt.as_tensor_variable(X[:, self.active_dims])
        if Xs is not None:
            Xs = pt.as_tensor_variable(Xs[:, self.active_dims])
        return X, Xs


class Combination(Covariance):
    def __init__(self, factor_list: Sequence):
        """Use constituent factors to get input_dim and active_dims for the Combination covariance."""
        # Check if all input_dim are the same in factor_list
        input_dims = {factor.input_dim for factor in factor_list if isinstance(factor, Covariance)}

        if len(input_dims) != 1:
            raise ValueError("All covariances must have the same `input_dim`.")
        input_dim = input_dims.pop()

        # Union all active_dims sets in factor_list for the combination covariance
        active_dims = np.sort(
            np.asarray(
                list(
                    set.union(
                        *[
                            set(factor.active_dims)
                            for factor in factor_list
                            if isinstance(factor, Covariance)
                        ]
                    )
                ),
                dtype=int,
            )
        )
        super().__init__(input_dim=input_dim, active_dims=active_dims)

        # Set up combination kernel, flatten out factor_list so that
        self._factor_list: list[Any] = []
        for factor in factor_list:
            if isinstance(factor, self.__class__):
                self._factor_list.extend(factor._factor_list)
            else:
                self._factor_list.append(factor)

    def _merge_factors_cov(self, X, Xs=None, diag=False):
        """Evaluate either all the sums or all the products of kernels that are possible to evaluate."""
        factor_list = []
        for factor in self._factor_list:
            # make sure diag=True is handled properly
            if isinstance(factor, BaseCovariance):
                factor_list.append(factor(X, Xs, diag))

            elif isinstance(factor, np.ndarray):
                if np.ndim(factor) == 2 and diag:
                    factor_list.append(np.diag(factor))
                else:
                    factor_list.append(factor)

            elif isinstance(
                factor,
                TensorConstant | TensorVariable | TensorSharedVariable,
            ):
                if factor.ndim == 2 and diag:
                    factor_list.append(pt.diag(factor))
                else:
                    factor_list.append(factor)

            else:
                factor_list.append(factor)

        return factor_list

    def _merge_factors_psd(self, omega):
        """Evaluate spectral densities of combination kernels when possible.

        Implements a more restricted set of rules than `_merge_factors_cov` --
        just additivity of stationary covariances with defined power spectral
        densities and multiplication by scalars.  Also, the active_dims for all
        covariances in the sum must be the same.
        """
        factor_list = []
        for factor in self._factor_list:
            if isinstance(factor, Covariance):
                # Allow merging covariances for psd only if active_dims are the same
                if set(self.active_dims) != set(factor.active_dims):
                    raise ValueError(
                        "For power spectral density calculations `active_dims` must be the same "
                        "for all covariances in the sum."
                    )

                # If it's a covariance try to calculate the psd
                try:
                    factor_list.append(factor.power_spectral_density(omega))

                except (AttributeError, NotImplementedError) as e:
                    if isinstance(factor, Stationary):
                        raise NotImplementedError(
                            f"No power spectral density method has been implemented for {factor}."
                        ) from e

                    else:
                        raise ValueError(
                            "Power spectral densities, `.power_spectral_density(omega)`, can only "
                            f"be calculated for `Stationary` covariance functions.  {factor} is "
                            "non-stationary."
                        ) from e

            else:
                # Otherwise defer the reduction to later
                factor_list.append(factor)

        return factor_list


class Add(Combination):
    def __call__(
        self,
        X: TensorLike,
        Xs: TensorLike | None = None,
        diag: bool = False,
    ) -> TensorVariable:
        return reduce(add, self._merge_factors_cov(X, Xs, diag))

    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        return reduce(add, self._merge_factors_psd(omega))


class Prod(Combination):
    def __call__(
        self,
        X: TensorLike,
        Xs: TensorLike | None = None,
        diag: bool = False,
    ) -> TensorVariable:
        return reduce(mul, self._merge_factors_cov(X, Xs, diag))

    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        check = Counter([isinstance(factor, Covariance) for factor in self._factor_list])
        if check.get(True, 0) >= 2:
            raise NotImplementedError(
                "The power spectral density of products of covariance functions is not implemented."
            )
        return reduce(mul, self._merge_factors_psd(omega))


class Exponentiated(Covariance):
    def __init__(self, kernel: Covariance, power):
        self.kernel = kernel
        self.power = power
        super().__init__(input_dim=self.kernel.input_dim, active_dims=self.kernel.active_dims)

    def __call__(
        self, X: TensorLike, Xs: TensorLike | None = None, diag: bool = False
    ) -> TensorVariable:
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

    def __init__(self, factor_list: Sequence[Covariance]):
        self.input_dims = [factor.input_dim for factor in factor_list]
        input_dim = sum(self.input_dims)
        super().__init__(input_dim=input_dim)
        self._factor_list = factor_list

    def _split(self, X, Xs):
        indices = np.cumsum(self.input_dims)
        X_split = np.hsplit(X, indices)
        if Xs is not None:
            Xs_split = np.hsplit(Xs, indices)
        else:
            Xs_split = [None] * len(X_split)
        return X_split, Xs_split

    def __call__(
        self, X: TensorLike, Xs: TensorLike | None = None, diag: bool = False
    ) -> TensorVariable:
        X_split, Xs_split = self._split(X, Xs)
        covs = [cov(x, xs, diag) for cov, x, xs in zip(self._factor_list, X_split, Xs_split)]
        return reduce(mul, covs)


class Constant(BaseCovariance):
    r"""
    Constant valued covariance function.

    .. math::

       k(x, x') = c
    """

    def __init__(self, c):
        self.c = c

    def diag(self, X: TensorLike) -> TensorVariable:
        return self._alloc(self.c, X.shape[0])

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        if Xs is None:
            return self._alloc(self.c, X.shape[0], X.shape[0])
        else:
            return self._alloc(self.c, X.shape[0], Xs.shape[0])


class WhiteNoise(BaseCovariance):
    r"""
    White noise covariance function.

    .. math::

       k(x, x') = \sigma^2 \mathrm{I}
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def diag(self, X: TensorLike) -> TensorVariable:
        return self._alloc(pt.square(self.sigma), X.shape[0])

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        if Xs is None:
            return pt.diag(self.diag(X))
        else:
            return self._alloc(0.0, X.shape[0], Xs.shape[0])


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

    def __init__(
        self,
        input_dim: int,
        period,
        tau=4,
        active_dims: IntSequence | None = None,
    ):
        super().__init__(input_dim, active_dims)
        self.c = pt.as_tensor_variable(period / 2)
        self.tau = tau

    def dist(self, X, Xs):
        if Xs is None:
            Xs = pt.transpose(X)
        else:
            Xs = pt.transpose(Xs)
        return pt.abs((X - Xs + self.c) % (self.c * 2) - self.c)

    def weinland(self, t):
        return (1 + self.tau * t / self.c) * pt.clip(1 - t / self.c, 0, np.inf) ** self.tau

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xs = self._slice(X, Xs)
        return self.weinland(self.dist(X, Xs))

    def diag(self, X: TensorLike) -> TensorVariable:
        return self._alloc(1.0, X.shape[0])


class Stationary(Covariance):
    r"""
    Base class for stationary kernels/covariance functions.

    Parameters
    ----------
    ls: Lengthscale.  If input_dim > 1, a list or array of scalars or PyMC random
        variables.  If input_dim == 1, a scalar or PyMC random variable.
    ls_inv: Inverse lengthscale.  1 / ls.  One of ls or ls_inv must be provided.
    """

    def __init__(
        self,
        input_dim: int,
        ls=None,
        ls_inv=None,
        active_dims: IntSequence | None = None,
    ):
        super().__init__(input_dim, active_dims)
        if (ls is None and ls_inv is None) or (ls is not None and ls_inv is not None):
            raise ValueError("Only one of 'ls' or 'ls_inv' must be provided")
        elif ls_inv is not None:
            if isinstance(ls_inv, list | tuple):
                ls = 1.0 / np.asarray(ls_inv)
            else:
                ls = 1.0 / ls_inv
        self.ls = pt.as_tensor_variable(ls)

    def square_dist(self, X, Xs):
        X = pt.mul(X, 1.0 / self.ls)
        X2 = pt.sum(pt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * pt.dot(X, pt.transpose(X)) + (
                pt.reshape(X2, (-1, 1)) + pt.reshape(X2, (1, -1))
            )
        else:
            Xs = pt.mul(Xs, 1.0 / self.ls)
            Xs2 = pt.sum(pt.square(Xs), 1)
            sqd = -2.0 * pt.dot(X, pt.transpose(Xs)) + (
                pt.reshape(X2, (-1, 1)) + pt.reshape(Xs2, (1, -1))
            )
        return pt.clip(sqd, 0.0, np.inf)

    def euclidean_dist(self, X, Xs):
        r2 = self.square_dist(X, Xs)
        return self._sqrt(r2)

    def _sqrt(self, r2):
        return pt.sqrt(r2 + 1e-12)

    def diag(self, X: TensorLike) -> TensorVariable:
        return self._alloc(1.0, X.shape[0])

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xs = self._slice(X, Xs)
        r2 = self.square_dist(X, Xs)
        return self.full_from_distance(r2, squared=True)

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        raise NotImplementedError

    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        raise NotImplementedError


class ExpQuad(Stationary):
    r"""
    The Exponentiated Quadratic kernel.

    Also referred to as the Squared Exponential, or Radial Basis Function kernel.

    .. math::

       k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{2 \ell^2} \right]

    """

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        r2 = dist if squared else dist**2
        return pt.exp(-0.5 * r2)

    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        r"""
        Power spectral density for the ExpQuad kernel.

        .. math::

           S(\boldsymbol\omega) =
               (\sqrt(2 \pi)^D \prod_{i}^{D}\ell_i
                \exp\left( -\frac{1}{2} \sum_{i}^{D}\ell_i^2 \omega_i^{2} \right)
        """
        ls = pt.ones(self.n_dims) * self.ls
        c = pt.power(pt.sqrt(2.0 * np.pi), self.n_dims)
        exp = pt.exp(-0.5 * pt.dot(pt.square(omega), pt.square(ls)))
        return c * pt.prod(ls) * exp


class RatQuad(Stationary):
    r"""
    The Rational Quadratic kernel.

    .. math::

       k(x, x') = \left(1 + \frac{(x - x')^2}{2\alpha\ell^2} \right)^{-\alpha}
    """

    def __init__(
        self,
        input_dim: int,
        alpha,
        ls=None,
        ls_inv=None,
        active_dims: IntSequence | None = None,
    ):
        super().__init__(input_dim, ls, ls_inv, active_dims)
        self.alpha = alpha

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        r2 = dist if squared else dist**2
        return pt.power(
            (1.0 + 0.5 * r2 * (1.0 / self.alpha)),
            -1.0 * self.alpha,
        )

    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        r"""
        Power spectral density for the Rational Quadratic kernel.

        .. math::
           S(\boldsymbol\omega) = \frac{2 (2\pi\alpha)^{D/2} \prod_{i=1}^D \ell_i}{\Gamma(\alpha)}
                                  \left(\frac{z}{2}\right)^{\nu}
                                  K_{\nu}(z)
        where :math:`z = \sqrt{2\alpha} \sqrt{\sum \ell_i^2 \omega_i^2}` and :math:`\nu = \alpha - D/2`.

        Derivation
        ----------
        The Rational Quadratic kernel can be expressed as a scale mixture of Squared Exponential kernels:

        .. math::
            k_{RQ}(r) = \int_0^\infty k_{SE}(r; \lambda) p(\lambda) d\lambda

        where :math:`k_{SE}(r; \lambda) = \exp\left(-\frac{\lambda r^2}{2}\right)` and the mixing distribution
        on the precision parameter :math:`\lambda` is :math:`\lambda \sim \text{Gamma}(\alpha, \beta)`
        with rate parameter :math:`\beta = \alpha \ell^2`.

        By the linearity of the Fourier transform, the PSD of the Rational Quadratic kernel is the expectation
        of the PSD of the Squared Exponential kernel with respect to the mixing distribution:

        .. math::
            S_{RQ}(\omega) = \int_0^\infty S_{SE}(\omega; \lambda) p(\lambda) d\lambda

        Substituting the known PSD of the Squared Exponential kernel and evaluating the integral yields
        the expression involving the modified Bessel function of the second kind, :math:`K_{\nu}(z)`.
        """
        ls = pt.ones(self.n_dims) * self.ls
        alpha = self.alpha
        D = self.n_dims
        nu = alpha - D / 2.0

        z = pt.sqrt(2 * alpha) * pt.sqrt(pt.dot(pt.square(omega), pt.square(ls)))
        coeff = 2.0 * pt.power(2.0 * np.pi * alpha, D / 2.0) * pt.prod(ls) / pt.gamma(alpha)

        # Handle singularity at z=0
        term_z = pt.switch(pt.eq(z, 0), pt.gamma(nu) / 2.0, pt.power(z / 2.0, nu) * pt.kv(nu, z))

        return coeff * term_z


class Matern52(Stationary):
    r"""
    The Matérn kernel with :math:`\nu = \frac{5}{2}`.

    .. math::

       k(x, x') = \left(1 + \frac{\sqrt{5(x - x')^2}}{\ell} +
                   \frac{5(x-x')^2}{3\ell^2}\right)
                   \mathrm{exp}\left[ - \frac{\sqrt{5(x - x')^2}}{\ell} \right]

    Read more `here <https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function>`_.

    Parameters
    ----------
    input_dim : int
        The number of input dimensions
    ls : scalar or array, optional
        Lengthscale parameter :math:`\ell`; if `input_dim` > 1, a list or array of scalars.
        If `input_dim` == 1, a scalar.
    ls_inv : scalar or array, optional
        Inverse lengthscale :math:`1 / \ell`. One of `ls` or `ls_inv` must be provided.
    active_dims : list of int, optional
        The dimension(s) the covariance function operates on.
    """

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        r = self._sqrt(dist) if squared else dist
        return (1.0 + np.sqrt(5.0) * r + 5.0 / 3.0 * pt.square(r)) * pt.exp(-1.0 * np.sqrt(5.0) * r)

    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        r"""
        Power spectral density for the Matern52 kernel.

        .. math::

           S(\boldsymbol\omega) =
               \frac{2^D \pi^{\frac{D}{2}} \Gamma(\frac{D+5}{2}) 5^{5/2}}
                    {\frac{3}{4}\sqrt{\pi}}
               \prod_{i=1}^{D}\ell_{i}
               \left(5 + \sum_{i=1}^{D}\ell_{i}^2 \boldsymbol\omega_{i}^{2}\right)^{-\frac{D+5}{2}}
        """
        ls = pt.ones(self.n_dims) * self.ls
        D52 = (self.n_dims + 5) / 2
        num = (
            pt.power(2, self.n_dims)
            * pt.power(np.pi, self.n_dims / 2)
            * pt.gamma(D52)
            * pt.power(5, 5 / 2)
        )
        den = 0.75 * pt.sqrt(np.pi)
        pow = pt.power(5.0 + pt.dot(pt.square(omega), pt.square(ls)), -1 * D52)
        return (num / den) * pt.prod(ls) * pow


class Matern32(Stationary):
    r"""
    The Matérn kernel with :math:`\nu = \frac{3}{2}`.

    .. math::

       k(x, x') = \left(1 + \frac{\sqrt{3(x - x')^2}}{\ell}\right)
                  \mathrm{exp}\left[ - \frac{\sqrt{3(x - x')^2}}{\ell} \right]

    Read more `here <https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function>`_.

    Parameters
    ----------
    input_dim : int
        The number of input dimensions
    ls : scalar or array, optional
        Lengthscale parameter :math:`\ell`; if `input_dim` > 1, a list or array of scalars.
        If `input_dim` == 1, a scalar.
    ls_inv : scalar or array, optional
        Inverse lengthscale :math:`1 / \ell`. One of `ls` or `ls_inv` must be provided.
    active_dims : list of int, optional
        The dimension(s) the covariance function operates on.
    """

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        r = self._sqrt(dist) if squared else dist
        return (1.0 + np.sqrt(3.0) * r) * pt.exp(-np.sqrt(3.0) * r)

    def power_spectral_density(self, omega: TensorLike) -> TensorVariable:
        r"""
        Power spectral density for the Matern32 kernel.

        .. math::

            S(\boldsymbol\omega) =
                \frac{2^D \pi^{D/2} \Gamma\left(\frac{D+3}{2}\right) 3^{3/2}}
                     {\frac{1}{2}\sqrt{\pi}}
               \prod_{i=1}^{D}\ell_{i}
               \left(3 + \sum_{i=1}^{D}\ell_{i}^2 \boldsymbol\omega_{i}^{2}\right)^{-\frac{D+3}{2}}
        """
        ls = pt.ones(self.n_dims) * self.ls
        D32 = (self.n_dims + 3) / 2
        num = (
            pt.power(2, self.n_dims)
            * pt.power(np.pi, self.n_dims / 2)
            * pt.gamma(D32)
            * pt.power(3, 3 / 2)
        )
        den = 0.5 * pt.sqrt(np.pi)
        pow = pt.power(3.0 + pt.dot(pt.square(omega), pt.square(ls)), -1 * D32)
        return (num / den) * pt.prod(ls) * pow


class Matern12(Stationary):
    r"""
    The Matern kernel with nu = 1/2.

    .. math::

        k(x, x') = \mathrm{exp}\left[ -\frac{(x - x')^2}{\ell} \right]
    """

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        r = self._sqrt(dist) if squared else dist
        return pt.exp(-r)


class Exponential(Stationary):
    r"""
    The Exponential kernel.

    .. math::

       k(x, x') = \mathrm{exp}\left[ -\frac{||x - x'||}{2\ell} \right]
    """

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        r = self._sqrt(dist) if squared else dist
        return pt.exp(-0.5 * r)


class Cosine(Stationary):
    r"""
    The Cosine kernel.

    .. math::
       k(x, x') = \mathrm{cos}\left( 2 \pi \frac{||x - x'||}{ \ell^2} \right)
    """

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        r = self._sqrt(dist) if squared else dist
        return pt.cos(2.0 * np.pi * r)


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

    def __init__(
        self,
        input_dim: int,
        period,
        ls=None,
        ls_inv=None,
        active_dims: IntSequence | None = None,
    ):
        super().__init__(input_dim, ls, ls_inv, active_dims)
        self.period = period

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xs = self._slice(X, Xs)
        if Xs is None:
            Xs = X
        f1 = pt.expand_dims(X, axis=(1,))
        f2 = pt.expand_dims(Xs, axis=(0,))
        r = np.pi * (f1 - f2) / self.period
        r2 = pt.sum(pt.square(pt.sin(r) / self.ls), 2)
        return self.full_from_distance(r2, squared=True)

    def full_from_distance(self, dist: TensorLike, squared: bool = False) -> TensorVariable:
        # NOTE: This is the same as the ExpQuad as we assume the periodicity
        # has already been accounted for in the distance
        r2 = dist if squared else dist**2
        return pt.exp(-0.5 * r2)

    def power_spectral_density_approx(self, J: TensorLike) -> TensorVariable:
        r"""Power spectral density approximation.

        Technically, this is not a spectral density but these are the first `m` coefficients of
        the low rank approximation for the periodic kernel, which are used in the same way.
        `J` is a vector of `np.arange(m)`.

        The coefficients of the HSGP approximation for the `Periodic` kernel are given by:

        .. math::

            \tilde{q}_j^2 = \frac{2 \text{I}_j (\\ell^{-2})}{\\exp(\\ell^{-2})} \\
            \tilde{q}_0^2 = \frac{\text{I}_0 (\\ell^{-2})}{\\exp(\\ell^{-2})}

        where $\text{I}_j$ is the modified Bessel function of the first kind.
        """
        a = 1 / pt.square(self.ls)
        c = pt.where(J > 0, 2, 1)

        q2 = c * pt.ive(J, a)

        return q2


class Linear(Covariance):
    r"""
    The Linear kernel.

    .. math::
       k(x, x') = (x - c)(x' - c)
    """

    def __init__(self, input_dim: int, c, active_dims: IntSequence | None = None):
        super().__init__(input_dim, active_dims)
        self.c = c

    def _common(self, X, Xs=None):
        X, Xs = self._slice(X, Xs)
        Xc = pt.sub(X, self.c)
        return X, Xc, Xs

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xc, Xs = self._common(X, Xs)
        if Xs is None:
            return pt.dot(Xc, pt.transpose(Xc))
        else:
            Xsc = pt.sub(Xs, self.c)
            return pt.dot(Xc, pt.transpose(Xsc))

    def diag(self, X: TensorLike) -> TensorVariable:
        X, Xc, _ = self._common(X, None)
        return pt.sum(pt.square(Xc), 1)


class Polynomial(Linear):
    r"""
    The Polynomial kernel.

    .. math::
       k(x, x') = [(x - c)(x' - c) + \mathrm{offset}]^{d}
    """

    def __init__(self, input_dim: int, c, d, offset, active_dims: IntSequence | None = None):
        super().__init__(input_dim, c, active_dims)
        self.d = d
        self.offset = offset

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        linear = super().full(X, Xs)
        return pt.power(linear + self.offset, self.d)

    def diag(self, X: TensorLike) -> TensorVariable:
        linear = super().diag(X)
        return pt.power(linear + self.offset, self.d)


class WarpedInput(Covariance):
    r"""
    Warp the inputs of any kernel using an arbitrary function defined using PyTensor.

    .. math::
       k(x, x') = k(w(x), w(x'))

    Parameters
    ----------
    cov_func: Covariance
    warp_func: callable
        PyTensor function of X and additional optional arguments.
    args: optional, tuple or list of scalars or PyMC variables
        Additional inputs (besides X or Xs) to warp_func.
    """

    def __init__(
        self,
        input_dim: int,
        cov_func: Covariance,
        warp_func: Callable,
        args=None,
        active_dims: IntSequence | None = None,
    ):
        super().__init__(input_dim, active_dims)
        if not callable(warp_func):
            raise TypeError("warp_func must be callable")
        if not isinstance(cov_func, Covariance):
            raise TypeError("Must be or inherit from the Covariance class")
        self.w = handle_args(warp_func)
        self.args = args
        self.cov_func = cov_func

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xs = self._slice(X, Xs)
        if Xs is None:
            return self.cov_func(self.w(X, self.args), Xs)
        else:
            return self.cov_func(self.w(X, self.args), self.w(Xs, self.args))

    def diag(self, X: TensorLike) -> TensorVariable:
        X, _ = self._slice(X, None)
        return self.cov_func(self.w(X, self.args), diag=True)


class WrappedPeriodic(Covariance):
    r"""
    The `WrappedPeriodic` kernel constructs periodic kernels from any `Stationary` kernel.

    This is done by warping the input with the function

    .. math::
        \mathbf{u}(x) = \left(
            \mathrm{sin} \left( \frac{2\pi x}{T} \right),
            \mathrm{cos} \left( \frac{2\pi x}{T} \right)
        \right)

    The original derivation as applied to the squared exponential kernel can
    be found in [1] or referenced in Chapter 4, page 92 of [2].

    Notes
    -----
    Note that we drop the resulting scaling by 4 found in the original derivation
    so that the interpretation of the length scales is consistent between the
    underlying kernel and the periodic version.

    Examples
    --------
    In order to construct a kernel equivalent to the `Periodic` kernel you
    can do the following (though using `Periodic` will likely be a bit faster):

    .. code-block:: python

        exp_quad = pm.gp.cov.ExpQuad(1, ls=0.5)
        cov = pm.gp.cov.WrappedPeriodic(exp_quad, period=5)

    References
    ----------
    .. [1] MacKay, D. J. C. (1998). Introduction to Gaussian Processes.
       In Bishop, C. M., editor, Neural Networks and Machine Learning. Springer-Verlag
    .. [2] Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes for machine learning. MIT Press.
       http://gaussianprocess.org/gpml/chapters/

    Parameters
    ----------
    cov_func: Stationary
        Base kernel or covariance function
    period: Period
    """

    def __init__(self, cov_func: Stationary, period):
        super().__init__(cov_func.input_dim, cov_func.active_dims)
        if not isinstance(cov_func, Stationary):
            raise TypeError("Must inherit from the Stationary class")
        self.cov_func = cov_func
        self.period = period

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xs = self._slice(X, Xs)
        if Xs is None:
            Xs = X
        f1 = pt.expand_dims(X, axis=(1,))
        f2 = pt.expand_dims(Xs, axis=(0,))
        r = np.pi * (f1 - f2) / self.period
        r2 = pt.sum(pt.square(pt.sin(r) / self.cov_func.ls), 2)
        return self.cov_func.full_from_distance(r2, squared=True)

    def diag(self, X: TensorLike) -> TensorVariable:
        return self._alloc(1.0, X.shape[0])


class Gibbs(Covariance):
    r"""
    The Gibbs kernel.

    Use an arbitrary lengthscale function defined using PyTensor.
    Only tested in one dimension.

    .. math::
       k(x, x') = \sqrt{\frac{2\ell(x)\ell(x')}{\ell^2(x) + \ell^2(x')}}
                  \mathrm{exp}\left[ -\frac{(x - x')^2}
                                           {\ell^2(x) + \ell^2(x')} \right]

    Parameters
    ----------
    lengthscale_func: callable
        PyTensor function of X and additional optional arguments.
    args: optional, tuple or list of scalars or PyMC variables
        Additional inputs (besides X or Xs) to lengthscale_func.
    """

    def __init__(
        self,
        input_dim: int,
        lengthscale_func: Callable,
        args=None,
        active_dims: IntSequence | None = None,
    ):
        super().__init__(input_dim, active_dims)
        if active_dims is not None:
            if len(active_dims) > 1:
                raise NotImplementedError(("Higher dimensional inputs ", "are untested"))
        else:
            if input_dim != 1:
                raise NotImplementedError(("Higher dimensional inputs ", "are untested"))
        if not callable(lengthscale_func):
            raise TypeError("lengthscale_func must be callable")
        self.lfunc = handle_args(lengthscale_func)
        self.args = args

    def square_dist(self, X, Xs=None):
        X2 = pt.sum(pt.square(X), 1)
        if Xs is None:
            sqd = -2.0 * pt.dot(X, pt.transpose(X)) + (
                pt.reshape(X2, (-1, 1)) + pt.reshape(X2, (1, -1))
            )
        else:
            Xs2 = pt.sum(pt.square(Xs), 1)
            sqd = -2.0 * pt.dot(X, pt.transpose(Xs)) + (
                pt.reshape(X2, (-1, 1)) + pt.reshape(Xs2, (1, -1))
            )
        return pt.clip(sqd, 0.0, np.inf)

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xs = self._slice(X, Xs)
        rx = self.lfunc(pt.as_tensor_variable(X), self.args)
        if Xs is None:
            rz = self.lfunc(pt.as_tensor_variable(X), self.args)
            r2 = self.square_dist(X, X)
        else:
            rz = self.lfunc(pt.as_tensor_variable(Xs), self.args)
            r2 = self.square_dist(X, Xs)
        rx2 = pt.reshape(pt.square(rx), (-1, 1))
        rz2 = pt.reshape(pt.square(rz), (1, -1))
        return pt.sqrt((2.0 * pt.outer(rx, rz)) / (rx2 + rz2)) * pt.exp(-1.0 * r2 / (rx2 + rz2))

    def diag(self, X: TensorLike) -> TensorVariable:
        return self._alloc(1.0, X.shape[0])


class ScaledCov(Covariance):
    r"""
    Construct a kernel by multiplying a base kernel with a scaling function defined using PyTensor.

    The scaling function is non-negative, and can be parameterized.

    .. math::
       k(x, x') = \phi(x) k_{\text{base}}(x, x') \phi(x')

    Parameters
    ----------
    cov_func: Covariance
        Base kernel or covariance function
    scaling_func: callable
        PyTensor function of X and additional optional arguments.
    args: optional, tuple or list of scalars or PyMC variables
        Additional inputs (besides X or Xs) to lengthscale_func.
    """

    def __init__(
        self,
        input_dim: int,
        cov_func: Covariance,
        scaling_func: Callable,
        args=None,
        active_dims: IntSequence | None = None,
    ):
        super().__init__(input_dim, active_dims)
        if not callable(scaling_func):
            raise TypeError("scaling_func must be callable")
        if not isinstance(cov_func, Covariance):
            raise TypeError("Must be or inherit from the Covariance class")
        self.cov_func = cov_func
        self.scaling_func = handle_args(scaling_func)
        self.args = args

    def diag(self, X: TensorLike) -> TensorVariable:
        X, _ = self._slice(X, None)
        cov_diag = self.cov_func(X, diag=True)
        scf_diag = pt.square(pt.flatten(self.scaling_func(X, self.args)))
        return cov_diag * scf_diag

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xs = self._slice(X, Xs)
        scf_x = self.scaling_func(X, self.args)
        if Xs is None:
            return pt.outer(scf_x, scf_x) * self.cov_func(X)
        else:
            scf_xs = self.scaling_func(Xs, self.args)
            return pt.outer(scf_x, scf_xs) * self.cov_func(X, Xs)


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

    def __init__(
        self,
        input_dim: int,
        W=None,
        kappa=None,
        B=None,
        active_dims: IntSequence | None = None,
    ):
        super().__init__(input_dim, active_dims)
        if len(self.active_dims) != 1:
            raise ValueError("Coregion requires exactly one dimension to be active")
        make_B = W is not None or kappa is not None
        if make_B and B is not None:
            raise ValueError("Exactly one of (W, kappa) and B must be provided to Coregion")
        if make_B:
            self.W = pt.as_tensor_variable(W)
            self.kappa = pt.as_tensor_variable(kappa)
            self.B = pt.dot(self.W, self.W.T) + pt.diag(self.kappa)
        elif B is not None:
            self.B = pt.as_tensor_variable(B)
        else:
            raise ValueError("Exactly one of (W, kappa) and B must be provided to Coregion")

    def full(self, X: TensorLike, Xs: TensorLike | None = None) -> TensorVariable:
        X, Xs = self._slice(X, Xs)
        index = pt.cast(X, "int32")
        if Xs is None:
            index2 = index.T
        else:
            index2 = pt.cast(Xs, "int32").T
        return self.B[index, index2]

    def diag(self, X: TensorLike) -> TensorVariable:
        X, _ = self._slice(X, None)
        index = pt.cast(X, "int32")
        return pt.diag(self.B)[index.ravel()]


def handle_args(func: Callable) -> Callable:
    def f(x, args):
        if args is None:
            return func(x)
        else:
            if not isinstance(args, tuple):
                args = (args,)
            return func(x, *args)

    return f
