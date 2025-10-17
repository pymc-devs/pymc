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

from collections.abc import Sequence
from types import ModuleType

import numpy as np
import pytensor.tensor as pt

import pymc as pm

from pymc.gp.cov import Covariance, Periodic
from pymc.gp.gp import Base
from pymc.gp.mean import Mean, Zero

TensorLike = np.ndarray | pt.TensorVariable


def set_boundary(X: TensorLike, c: numbers.Real | TensorLike) -> np.ndarray:
    """Set the boundary using `X` and `c`.

    `X` can be centered around zero but doesn't have to be, and `c` is usually
    a scalar multiplier greater than 1.0, but it may also be one value per
    dimension or column of `X`.
    """
    # compute radius. Works whether X is 0-centered or not
    S = (pt.max(X, axis=0) - pt.min(X, axis=0)) / 2.0

    L = (c * S).eval()  # eval() makes sure L is not changed with out-of-sample preds
    return L


def calc_eigenvalues(L: TensorLike, m: Sequence[int]):
    """Calculate eigenvalues of the Laplacian."""
    S = np.meshgrid(*[np.arange(1, 1 + m[d]) for d in range(len(m))])
    S_arr = np.vstack([s.flatten() for s in S]).T

    return np.square((np.pi * S_arr) / (2 * L))


def calc_eigenvectors(
    Xs: TensorLike,
    L: TensorLike,
    eigvals: TensorLike,
    m: Sequence[int],
):
    """Calculate eigenvectors of the Laplacian.

    These are used as basis vectors in the HSGP approximation.
    """
    m_star = int(np.prod(m))

    phi = pt.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / pt.sqrt(L[d])
        term1 = pt.sqrt(eigvals[:, d])
        term2 = pt.tile(Xs[:, d][:, None], m_star) + L[d]
        phi *= c * pt.sin(term1 * term2)

    return phi


def calc_basis_periodic(
    Xs: TensorLike,
    period: TensorLike,
    m: int,
    tl: ModuleType = np,
):
    """
    Calculate basis vectors for the cosine series expansion of the periodic covariance function.

    These are derived from the Taylor series representation of the covariance.
    """
    w0 = (2 * np.pi) / period  # angular frequency defining the periodicity
    m1 = tl.tile(w0 * Xs, m)
    m2 = tl.diag(tl.arange(0, m, 1))
    mw0x = m1 @ m2
    phi_cos = tl.cos(mw0x)
    phi_sin = tl.sin(mw0x)
    return phi_cos, phi_sin


def approx_hsgp_hyperparams(
    x_range: list[float], lengthscale_range: list[float], cov_func: str
) -> tuple[int, float]:
    """Use heuristics to recommend minimum `m` and `c` values, based on recommendations from Ruitort-Mayol et. al.

    In practice, you need to choose `c` large enough to handle the largest lengthscales,
    and `m` large enough to accommodate the smallest lengthscales.  Use your prior on the
    lengthscale as guidance for setting the prior range.  For example, if you believe
    that 95% of the prior mass of the lengthscale is between 1 and 5, set the
    `lengthscale_range` to be [1, 5], or maybe a touch wider.

    Also, be sure to pass in an `x_range` that is exemplary of the domain not just of your
    training data, but also where you intend to make predictions.  For instance, if your
    training x values are from [0, 10], and you intend to predict from [7, 15], the narrowest
    `x_range` you should pass in would be `x_range = [0, 15]`.

    NB: These recommendations are based on a one-dimensional GP.

    Parameters
    ----------
    x_range : list[float]
        The range of the x values you intend to both train and predict over.  Should be a list with
        two elements, [x_min, x_max].
    lengthscale_range : List[float]
        The range of the lengthscales. Should be a list with two elements, [lengthscale_min, lengthscale_max].
    cov_func : str
        The covariance function to use. Supported options are "expquad", "matern52", and "matern32".

    Returns
    -------
    - `m` : int
        Number of basis vectors. Increasing it helps approximate smaller lengthscales, but increases computational cost.
    - `c` : float
        Scaling factor such that L = c * S, where L is the boundary of the approximation.
        Increasing it helps approximate larger lengthscales, but may require increasing m.

    Raises
    ------
    ValueError
        If either `x_range` or `lengthscale_range` is not in the correct order.

    References
    ----------
    - Ruitort-Mayol, G., Anderson, M., Solin, A., Vehtari, A. (2022).
    Practical Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming
    """
    if lengthscale_range[0] >= lengthscale_range[1]:
        raise ValueError("One of the `lengthscale_range` boundaries is out of order.")

    if x_range[0] >= x_range[1]:
        raise ValueError("One of the `x_range` boundaries is out of order.")

    S = (x_range[1] - x_range[0]) / 2.0

    if cov_func.lower() == "expquad":
        a1, a2 = 3.2, 1.75

    elif cov_func.lower() == "matern52":
        a1, a2 = 4.1, 2.65

    elif cov_func.lower() == "matern32":
        a1, a2 = 4.5, 3.42

    else:
        raise ValueError(
            "Unsupported covariance function. Supported options are 'expquad', 'matern52', and 'matern32'."
        )

    c = max(a1 * (lengthscale_range[1] / S), 1.2)
    m = int(a2 * c / (lengthscale_range[0] / S))

    return m, c


class HSGP(Base):
    R"""
    Hilbert Space Gaussian process approximation.

    The `gp.HSGP` class is an implementation of the Hilbert Space Gaussian process.  It is a
    reduced rank GP approximation that uses a fixed set of basis vectors whose coefficients are
    random functions of a stationary covariance function's power spectral density.  Its usage
    is largely similar to `gp.Latent`.  Like `gp.Latent`, it does not assume a Gaussian noise model
    and can be used with any likelihood, or as a component anywhere within a model.  Also like
    `gp.Latent`, it has `prior` and `conditional` methods.  It supports any sum of covariance
    functions that implement a `power_spectral_density` method. (Note, this excludes the
    `Periodic` covariance function, which uses a different set of basis functions for a
    low rank approximation, as described in `HSGPPeriodic`.).

    For information on choosing appropriate `m`, `L`, and `c`, refer to Ruitort-Mayol et al. or to
    the PyMC examples that use HSGP.

    To work with the HSGP in its "linearized" form, as a matrix of basis vectors and a vector of
    coefficients, see the method `prior_linearized`.

    Parameters
    ----------
    m: list
        The number of basis vectors to use for each active dimension (covariance parameter
        `active_dim`).
    L: list
        The boundary of the space for each `active_dim`.  It is called the boundary condition.
        Choose L such that the domain `[-L, L]` contains all points in the column of X given by the
        `active_dim`.
    c: float
        The proportion extension factor.  Used to construct L from X.  Defined as `S = max|X|` such
        that `X` is in `[-S, S]`.  `L` is calculated as `c * S`.  One of `c` or `L` must be
        provided.  Further information can be found in Ruitort-Mayol et al.
    drop_first: bool
        Default `False`. Sometimes the first basis vector is quite "flat" and very similar to
        the intercept term.  When there is an intercept in the model, ignoring the first basis
        vector may improve sampling. This argument will be deprecated in future versions.
    parametrization: str
        Whether to use the `centered` or `noncentered` parametrization when multiplying the
        basis by the coefficients.
    cov_func: Covariance function, must be an instance of `Stationary` and implement a
        `power_spectral_density` method.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code-block:: python

        # A three dimensional column vector of inputs.
        X = np.random.rand(100, 3)

        with pm.Model() as model:
            # Specify the covariance function.
            # Three input dimensions, but we only want to use the last two.
            cov_func = pm.gp.cov.ExpQuad(3, ls=0.1, active_dims=[1, 2])

            # Specify the HSGP.
            # Use 25 basis vectors across each active dimension for a total of 25 * 25 = 625.
            # The value `c = 4` means the boundary of the approximation
            # lies at four times the half width of the data.
            # In this example the data lie between zero and one,
            # so the boundaries occur at -1.5 and 2.5.  The data, both for
            # training and prediction should reside well within that boundary..
            gp = pm.gp.HSGP(m=[25, 25], c=4.0, cov_func=cov_func)

            # Place a GP prior over the function f.
            f = gp.prior("f", X=X)

        ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)

    References
    ----------
    -   Ruitort-Mayol, G., and Anderson, M., and Solin, A., and Vehtari, A. (2022). Practical
        Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming

    -   Solin, A., Sarkka, S. (2019) Hilbert Space Methods for Reduced-Rank Gaussian Process
        Regression.
    """

    def __init__(
        self,
        m: Sequence[int],
        L: Sequence[float] | None = None,
        c: numbers.Real | None = None,
        drop_first: bool = False,
        parametrization: str | None = "noncentered",
        *,
        mean_func: Mean = Zero(),
        cov_func: Covariance,
    ):
        arg_err_msg = (
            "`m` and `L`, if provided, must be sequences with one element per active "
            "dimension of the kernel or covariance function."
        )

        if not isinstance(m, Sequence):
            raise ValueError(arg_err_msg)

        if len(m) != cov_func.n_dims:
            raise ValueError(arg_err_msg)
        m = tuple(m)

        if (L is None and c is None) or (L is not None and c is not None):
            raise ValueError("Provide one of `c` or `L`")

        if L is not None and (not isinstance(L, Sequence) or len(L) != cov_func.n_dims):
            raise ValueError(arg_err_msg)

        if L is None and c is not None and c < 1.2:
            warnings.warn("For an adequate approximation `c >= 1.2` is recommended.")

        if parametrization is not None:
            parametrization = parametrization.lower().replace("-", "").replace("_", "")

        if parametrization not in ["centered", "noncentered"]:
            raise ValueError("`parametrization` must be either 'centered' or 'noncentered'.")

        if drop_first:
            warnings.warn(
                "The drop_first argument will be deprecated in future versions."
                " See https://github.com/pymc-devs/pymc/pull/6877",
                DeprecationWarning,
            )

        self._drop_first = drop_first
        self._m = m
        self._m_star = self.n_basis_vectors = int(np.prod(self._m))
        self._L: pt.TensorVariable | None = None
        if L is not None:
            self._L = pt.as_tensor(L).eval()  # make sure L cannot be changed
        self._c = c
        self._parametrization = parametrization
        self._X_center = None

        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        """Add two HSGPs."""
        raise NotImplementedError("Additive HSGPs aren't supported.")

    @property
    def L(self) -> pt.TensorVariable:
        if self._L is None:
            raise RuntimeError("Boundaries `L` required but still unset.")
        return self._L

    @L.setter
    def L(self, value: TensorLike):
        self._L = pt.as_tensor_variable(value)

    def prior_linearized(self, X: TensorLike):
        """Linearized version of the HSGP.

        Returns the Laplace eigenfunctions and the square root
        of the power spectral density needed to create the GP.

        This function allows the user to bypass the GP interface and work with
        the basis and coefficients directly.  This format allows the user to
        create predictions using `pm.set_data` similarly to a linear model.  It
        also enables computational speed ups in multi-GP models, since they may
        share the same basis.  The return values are the Laplace eigenfunctions
        `phi`, and the square root of the power spectral density.

        An example is given below.

        Parameters
        ----------
        X: array-like
            Function input values.

        Returns
        -------
        phi: array-like
            Either Numpy or PyTensor 2D array of the fixed basis vectors.  There are n rows, one
            per row of `Xs` and `prod(m)` columns, one for each basis vector.
        sqrt_psd: array-like
            Either a Numpy or PyTensor 1D array of the square roots of the power spectral
            densities.

        Examples
        --------
        .. code-block:: python

            # A one dimensional column vector of inputs.
            X = np.linspace(0, 10, 100)[:, None]

            with pm.Model() as model:
                eta = pm.Exponential("eta", lam=1.0)
                ell = pm.InverseGamma("ell", mu=5.0, sigma=5.0)
                cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)

                # m = [200] means 200 basis vectors for the first dimension
                # L = [10] means the approximation is valid from Xs = [-10, 10]
                gp = pm.gp.HSGP(m=[200], L=[10], cov_func=cov_func)

                # Set X as Data so it can be mutated later, and then pass it to the GP
                X = pm.Data("X", X)
                phi, sqrt_psd = gp.prior_linearized(X=X)

                # Specify standard normal prior in the coefficients, the number of which
                # is given by the number of basis vectors, saved in `n_basis_vectors`.
                beta = pm.Normal("beta", size=gp.n_basis_vectors)

                # The (non-centered) GP approximation is given by:
                f = pm.Deterministic("f", phi @ (beta * sqrt_psd))

                # The centered approximation can be more efficient when
                # the GP is stronger than the noise
                # beta = pm.Normal("beta", sigma=sqrt_psd, size=gp.n_basis_vectors)
                # f = pm.Deterministic("f", phi @ beta)

                ...


            # Then it works just like a linear regression to predict on new data.
            # First mutate the data X,
            x_new = np.linspace(-10, 10, 100)
            with model:
                pm.set_data({"X": x_new[:, None]})

            # and then make predictions for the GP using posterior predictive sampling.
            with model:
                ppc = pm.sample_posterior_predictive(idata, var_names=["f"])
        """
        # Important: fix the computation of the midpoint of X.
        # If X is mutated later, the training midpoint will be subtracted, not the testing one.
        if self._X_center is None:
            self._X_center = (pt.max(X, axis=0) + pt.min(X, axis=0)).eval() / 2
        Xs = X - self._X_center  # center for accurate computation

        # Index Xs using input_dim and active_dims of covariance function
        Xs, _ = self.cov_func._slice(Xs)

        # If not provided, use Xs and c to set L
        if self._L is None:
            assert isinstance(self._c, numbers.Real | np.ndarray | pt.TensorVariable)
            self.L = pt.as_tensor(set_boundary(Xs, self._c))  # Xs should be 0-centered
        else:
            self.L = self._L

        eigvals = calc_eigenvalues(self.L, self._m)
        phi = calc_eigenvectors(Xs, self.L, eigvals, self._m)
        omega = pt.sqrt(eigvals)
        psd = self.cov_func.power_spectral_density(omega)

        i = int(self._drop_first is True)
        return phi[:, i:], pt.sqrt(psd[i:])

    def prior(
        self,
        name: str,
        X: TensorLike,
        dims: str | None = None,
        hsgp_coeffs_dims: str | None = None,
        *args,
        **kwargs,
    ):
        R"""
        Return the (approximate) GP prior distribution evaluated over the input locations `X`.

        For usage examples, refer to `pm.gp.Latent`.

        Parameters
        ----------
        name: str
            Name of the random variable
        X: array-like
            Function input values.
        dims: str, default None
            Dimension name for the GP random variable.
        hsgp_coeffs_dims: str, default None
            Dimension name for the HSGP basis vectors.

        """
        phi, sqrt_psd = self.prior_linearized(X)
        self._sqrt_psd = sqrt_psd

        if self._parametrization == "noncentered":
            self._beta = pm.Normal(
                f"{name}_hsgp_coeffs",
                size=self.n_basis_vectors - int(self._drop_first),
                dims=hsgp_coeffs_dims,
            )
            f = self.mean_func(X) + phi @ (self._beta * self._sqrt_psd)

        elif self._parametrization == "centered":
            self._beta = pm.Normal(
                f"{name}_hsgp_coeffs",
                sigma=sqrt_psd,
                size=self.n_basis_vectors - int(self._drop_first),
                dims=hsgp_coeffs_dims,
            )
            f = self.mean_func(X) + phi @ self._beta

        self.f = pm.Deterministic(name, f, dims=dims)
        return self.f

    def _build_conditional(self, Xnew):
        try:
            beta, X_center = self._beta, self._X_center

            if self._parametrization == "noncentered":
                sqrt_psd = self._sqrt_psd

        except AttributeError:
            raise ValueError(
                "Prior is not set, can't create a conditional.  Call `.prior(name, X)` first."
            )

        Xnew, _ = self.cov_func._slice(Xnew)

        eigvals = calc_eigenvalues(self.L, self._m)
        phi = calc_eigenvectors(Xnew - X_center, self.L, eigvals, self._m)
        i = int(self._drop_first is True)

        if self._parametrization == "noncentered":
            return self.mean_func(Xnew) + phi[:, i:] @ (beta * sqrt_psd)

        elif self._parametrization == "centered":
            return self.mean_func(Xnew) + phi[:, i:] @ beta

    def conditional(self, name: str, Xnew: TensorLike, dims: str | None = None):  # type: ignore[override]
        R"""
        Return the (approximate) conditional distribution evaluated over new input locations `Xnew`.

        Parameters
        ----------
        name
            Name of the random variable
        Xnew : array-like
            Function input values.
        dims: None
            Dimension name for the GP random variable.
        """
        fnew = self._build_conditional(Xnew)
        return pm.Deterministic(name, fnew, dims=dims)


class HSGPPeriodic(Base):
    R"""
    Hilbert Space Gaussian process approximation for the Periodic covariance function.

    Note, this is not actually a Hilbert space approximation, but it comes from the same
    paper (Ruitort-Mayol et al., 2022. See Appendix B) and follows the same spirit: using a basis
    approximation to a Gaussian process. In this case, the approximation is based on a series of
    stochastic resonators.

    For these reasons, we have followed the same API as `gp.HSGP`, and can be used as a drop-in
    replacement for `gp.Latent`. Like `gp.Latent`, it has `prior` and `conditional` methods.

    For information on choosing appropriate `m`, refer to Ruitort-Mayol et al.. Note, this approximation
    is only implemented for the 1-D case.

    To work with the approximation in its "linearized" form, as a matrix of basis vectors and a
    vector of coefficients, see the method `prior_linearized`.

    Parameters
    ----------
    m: int
        The number of basis vectors to use. Must be a positive integer.
    scale: TensorLike
        The standard deviation (square root of the variance) of the GP effect. Defaults to 1.0.
    cov_func: Must be an instance of instance of `Periodic` covariance
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code-block:: python

        # A three dimensional column vector of inputs.
        X = np.random.rand(100, 3)

        with pm.Model() as model:
            # Specify the covariance function, only for the 1-D case
            scale = pm.HalfNormal("scale", 10)
            cov_func = pm.gp.cov.Periodic(1, period=1, ls=0.1)

            # Specify the approximation with 25 basis vectors
            gp = pm.gp.HSGPPeriodic(m=25, scale=scale, cov_func=cov_func)

            # Place a GP prior over the function f.
            f = gp.prior("f", X=X)

        ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)

    References
    ----------
    -   Ruitort-Mayol, G., and Anderson, M., and Solin, A., and Vehtari, A. (2022). Practical
        Hilbert Space Approximate Bayesian Gaussian Processes for Probabilistic Programming
    """

    def __init__(
        self,
        m: int,
        scale: float | TensorLike | None = 1.0,
        *,
        mean_func: Mean = Zero(),
        cov_func: Periodic,
    ):
        arg_err_msg = "`m` must be a positive integer as the `Periodic` kernel approximation is only implemented for 1-dimensional case."

        if not isinstance(m, int):
            raise ValueError(arg_err_msg)

        if m <= 0:
            raise ValueError(arg_err_msg)

        if not isinstance(cov_func, Periodic):
            raise ValueError(
                "`cov_func` must be an instance of a `Periodic` kernel only. Use the `scale` parameter to control the variance."
            )

        if cov_func.n_dims > 1:
            raise ValueError(
                "HSGP approximation for `Periodic` kernel only implemented for 1-dimensional case."
            )

        self._m = m
        self.scale = scale
        self._X_center = None

        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def prior_linearized(self, X: TensorLike):
        """Linearized version of the approximation.

        Returns the cosine and sine bases and coefficients
        of the expansion needed to create the GP.

        This function allows the user to bypass the GP interface and work
        directly with the basis and coefficients directly.  This format allows
        the user to create predictions using `pm.set_data` similarly to a linear
        model.  It also enables computational speed ups in multi-GP models since
        they may share the same basis.

        Correct results when using `prior_linearized` in tandem with
        `pm.set_data` and `pm.Data` require that the `Xs` are
        zero-centered, so its mean must be subtracted.

        An example is given below.

        Parameters
        ----------
        X: array-like
            Function input values.

        Returns
        -------
        (phi_cos, phi_sin): Tuple[array-like]
            List of either Numpy or PyTensor 2D array of the cosine and sine fixed basis vectors.
            There are n rows, one per row of `Xs` and `m` columns, one for each basis vector.
        psd: array-like
            Either a Numpy or PyTensor 1D array of the coefficients of the expansion.

        Examples
        --------
        .. code-block:: python

            # A one dimensional column vector of inputs.
            X = np.linspace(0, 10, 100)[:, None]

            with pm.Model() as model:
                scale = pm.HalfNormal("scale", 10)
                cov_func = pm.gp.cov.Periodic(1, period=1.0, ls=2.0)

                # m=200 means 200 basis vectors
                gp = pm.gp.HSGPPeriodic(m=200, scale=scale, cov_func=cov_func)

                # Set X as Data so it can be mutated later, and then pass it to the GP
                X = pm.Data("X", X)
                (phi_cos, phi_sin), psd = gp.prior_linearized(X=X)

                # Specify standard normal prior in the coefficients.  The number of which
                # is twice the number of basis vectors minus one.
                # This is so that each cosine term has a `beta` and all but one of the
                # sine terms, as first eigenfunction for the sine component is zero
                m = gp._m
                beta = pm.Normal("beta", size=(m * 2 - 1))

                # The (non-centered) GP approximation is given by
                f = pm.Deterministic(
                    "f", phi_cos @ (psd * beta[:m]) + phi_sin[..., 1:] @ (psd[1:] * beta[m:])
                )
                ...


            # Then it works just like a linear regression to predict on new data.
            # First mutate the data X,
            x_new = np.linspace(-10, 10, 100)
            with model:
                model.set_data("X", x_new[:, None])

            # and then make predictions for the GP using posterior predictive sampling.
            with model:
                ppc = pm.sample_posterior_predictive(idata, var_names=["f"])
        """
        # Important: fix the computation of the midpoint of X.
        # If X is mutated later, the training midpoint will be subtracted, not the testing one.
        if self._X_center is None:
            self._X_center = (pt.max(X, axis=0) + pt.min(X, axis=0)).eval() / 2
        Xs = X - self._X_center  # center for accurate computation

        # Index Xs using input_dim and active_dims of covariance function
        Xs, _ = self.cov_func._slice(Xs)

        phi_cos, phi_sin = calc_basis_periodic(Xs, self.cov_func.period, self._m, tl=pt)
        J = pt.arange(0, self._m, 1)
        # rescale basis coefficients by the sqrt variance term
        psd = self.scale * self.cov_func.power_spectral_density_approx(J)
        return (phi_cos, phi_sin), psd

    def prior(  # type: ignore[override]
        self, name: str, X: TensorLike, dims: str | None = None, hsgp_coeffs_dims: str | None = None
    ):
        R"""
        Return the (approximate) GP prior distribution evaluated over the input locations `X`.

        For usage examples, refer to `pm.gp.Latent`.

        Parameters
        ----------
        name: str
            Name of the random variable
        X: array-like
            Function input values.
        dims: None
            Dimension name for the GP random variable.
        hsgp_coeffs_dims: str | None = None
            Dimension name for the HSGPPeriodic basis vectors.
        """
        (phi_cos, phi_sin), psd = self.prior_linearized(X)

        m = self._m
        self._beta = pm.Normal(f"{name}_hsgp_coeffs_", size=(m * 2 - 1), dims=hsgp_coeffs_dims)
        # The first eigenfunction for the sine component is zero
        # and so does not contribute to the approximation.
        f = (
            self.mean_func(X)
            + phi_cos @ (psd * self._beta[:m])  # type: ignore[index]
            + phi_sin[..., 1:] @ (psd[1:] * self._beta[m:])  # type: ignore[index]
        )

        self.f = pm.Deterministic(name, f, dims=dims)
        return self.f

    def _build_conditional(self, Xnew):
        try:
            beta, X_center = self._beta, self._X_center

        except AttributeError:
            raise ValueError(
                "Prior is not set, can't create a conditional.  Call `.prior(name, X)` first."
            )

        Xnew, _ = self.cov_func._slice(Xnew)

        phi_cos, phi_sin = calc_basis_periodic(
            Xnew - X_center, self.cov_func.period, self._m, tl=pt
        )
        m = self._m
        J = pt.arange(0, m, 1)
        # rescale basis coefficients by the sqrt variance term
        psd = self.scale * self.cov_func.power_spectral_density_approx(J)

        phi = phi_cos @ (psd * beta[:m]) + phi_sin[..., 1:] @ (psd[1:] * beta[m:])
        return self.mean_func(Xnew) + phi

    def conditional(self, name: str, Xnew: TensorLike, dims: str | None = None):  # type: ignore[override]
        R"""
        Return the (approximate) conditional distribution evaluated over new input locations `Xnew`.

        Parameters
        ----------
        name
            Name of the random variable
        Xnew : array-like
            Function input values.
        dims: None
            Dimension name for the GP random variable.
        """
        fnew = self._build_conditional(Xnew)
        return pm.Deterministic(name, fnew, dims=dims)
