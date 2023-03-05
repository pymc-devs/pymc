#   Copyright 2023 The PyMC Developers
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

from types import ModuleType
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pytensor.tensor as pt

import pymc as pm

from pymc.gp.cov import Covariance
from pymc.gp.gp import Base
from pymc.gp.mean import Mean, Zero

TensorVariable = Union[np.ndarray, pt.TensorVariable]
TensorConstant = Union[np.ndarray, pt.TensorConstant]


def set_boundaries(
    Xs: TensorVariable,
    L: Optional[Sequence] = None,
    c: Optional[Union[float, Sequence]] = None,
    tl: ModuleType = np,
) -> Tuple[TensorVariable, TensorVariable, Union[TensorConstant, float]]:
    """R Compute the boundary over which the approximation is accurate using the centered input data
    locations, `Xs`.  It is requried that `Xs` has a mean at zero for each dimension, such that
    `np.mean(Xs, axis=0) == [0, ..., 0]`.  If `L` is provided, c is calculated instead.  `tl` stands
    for "tensor library", so the user can choose whether basic calculations are done with pytensor
    or numpy.

    Parameters
    ----------
    Xs: array-like
        Function input values.  Assumes they have been mean subtracted or centered at zero.
    """

    if tl.__name__ not in ("numpy", "pytensor.tensor"):
        raise ValueError("tl must be either numpy or pytensor.tensor.")

    S = tl.max(tl.abs(Xs), axis=0)

    if L is None and c is not None:
        L = c * S
    elif c is None and L is not None:
        c = L / S
    elif c is None and L is None:
        raise ValueError("At least one of `c` or `L` must be supplied.")
    # if both are passed, L takes precedent.

    if tl.__name__ == "numpy":
        L = tl.asarray(L)
    elif tl.__name__ == "pytensor.tensor":
        L = tl.as_tensor_variable(L)

    return S, L, c


def calc_eigenvalues(L: TensorConstant, m: Sequence[int], tl: ModuleType = np):
    """R Calculate eigenvalues of the Laplacian."""
    S = np.meshgrid(*[np.arange(1, 1 + m[d]) for d in range(len(m))])
    S = np.vstack([s.flatten() for s in S]).T
    top = np.pi * S
    bot = 2 * L
    return tl.square((np.pi * S) / (2 * L))


def calc_eigenvectors(
    Xs: TensorVariable,
    L: TensorConstant,
    eigvals: TensorConstant,
    m: Sequence[int],
    tl: ModuleType = np,
):
    """R Calculate eigenvectors of the Laplacian.  These are used as basis vectors in the HSGP
    approximation.
    """
    m_star = int(np.prod(m))
    phi = tl.ones((Xs.shape[0], m_star))
    for d in range(len(m)):
        c = 1.0 / tl.sqrt(L[d])
        term1 = tl.sqrt(eigvals[:, d])
        term2 = tl.tile(Xs[:, d][:, None], m_star) + L[d]
        phi *= c * tl.sin(term1 * term2)
    return phi


class HSGP(Base):
    R"""
    Hilbert Space Gaussian process

    The `gp.HSGP` class is an implementation of the Hilbert Space Gaussian process.  It is a
    reduced rank GP approximation that uses a fixed set of basis vectors whose coefficients are
    random functions of a stationary covariance function's power spectral density.  It's usage
    is largely similar to `gp.Latent`.  Like `gp.Latent`, it does not assume a Gaussian noise model
    and can be used with any likelihood, or as a component anywhere within a model.  Also like
    `gp.Latent`, it has `prior` and `conditional` methods.  It supports a limited subset of
    additive covariances.

    For information on choosing appropriate `m`, `L`, and `c`, refer Ruitort-Mayol et. al. or to
    the pymc examples that use HSGP.

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
        that `X` is in `[-S, S]`.  `L` is the calculated as `c * S`.  One of `c` or `L` must be
        provided.  Further information can be found in Ruitort-Mayol et. al.
    drop_first: bool
        Default `False`. Sometimes the first basis vector is quite "flat" and very similar to
        the intercept term.  When there is an intercept in the model, ignoring the first basis
        vector may improve sampling.
    cov_func: None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

        # A three dimensional column vector of inputs.
        X = np.random.rand(100, 3)

        with pm.Model() as model:
            # Specify the covariance function.  Three input dimensions, but we only want to use the
            # last two.
            cov_func = pm.gp.cov.ExpQuad(3, ls=0.1, active_dims=[1, 2])

            # Specify the HSGP.  Use 25 basis vectors across each active dimension, [1, 2]  for a
            # total of 25 * 25 = 625.  The value `c = 4` means the boundary of the approximation
            # lies at four times the half width of the data.  In this example the data lie between
            # zero and  one, so the boundaries occur at -1.5 and 2.5.  The data, both for training
            # and prediction should reside well within that boundary..
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
        L: Optional[Sequence[float]] = None,
        c: Optional[float] = None,
        drop_first: bool = False,
        parameterization="noncentered",
        *,
        mean_func: Mean = Zero(),
        cov_func: Covariance,
    ):
        arg_err_msg = (
            "`m` and L, if provided, must be sequences with one element per active "
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
        elif L is not None:
            L = tuple(L)

        if L is None and c is not None and c < 1.2:
            warnings.warn(
                "Most applications will require `c >= 1.2` for accuracy at the boundaries of the "
                "domain."
            )

        parameterization = parameterization.lower().replace("-", "")
        if parameterization not in ["centered", "noncentered"]:
            raise ValueError("`parameterization` must be either 'centered' or 'noncentered'.")
        else:
            self.parameterization = parameterization

        self.drop_first = drop_first
        self.L = L
        self.m = m
        self.c = c
        self.n_dims = cov_func.n_dims
        self.m_star = int(np.prod(self.m))
        self._boundary_set = False

        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise NotImplementedError("Additive HSGPs aren't supported ")

    def prior_linearized(self, Xs: TensorVariable):
        """Returns the linearized version of the HSGP, the Laplace eigenfunctions and the square
        root of the power spectral density needed to create the GP.  This function allows the user
        to bypass the GP interface and work directly with the basis and coefficients directly.
        This format allows the user to create predictions using `pm.set_data` similarly to a
        linear model.  It also enables computational speed ups in multi-GP models since they may
        share the same basis.  The return values are the Laplace eigenfunctions `phi`, and the
        square root of the power spectral density.

        Correct results when using `prior_linearized` in tandem with `pm.set_data` and
        `pm.MutableData` require two conditions.  First, one must specify `L` instead of `c` when
        the GP is constructed.  If not, a RuntimeError is raised.  Second, the `Xs` needs to be
        zero centered, so it's mean must be subtracted.  An example is given below.

        Parameters
        ----------
        Xs: array-like
            Function input values.  Assumes they have been mean subtracted or centered at zero.

        Examples
        --------
        .. code:: python

            # A one dimensional column vector of inputs.
            X = np.linspace(0, 10, 100)[:, None]

            with pm.Model() as model:
                eta = pm.Exponential("eta", lam=1.0)
                ell = pm.InverseGamma("ell", mu=5.0, sigma=5.0)
                cov_func = eta**2 * pm.gp.cov.ExpQuad(1, ls=ell)

                # m = [200] means 200 basis vectors for the first dimenison
                # L = [10] means the approximation is valid from Xs = [-10, 10]
                gp = pm.gp.HSGP(m=[200], L=[10], cov_func=cov_func)

                # Order is important.  First calculate the mean, then make X a shared variable,
                # then subtract the mean.  When X is mutated later, the correct mean will be
                # subtracted.
                X_mu = np.mean(X, axis=0)
                X = pm.MutableData("X", X)
                Xs = X - X_mu

                # Pass the zero-subtracted Xs in to the GP
                phi, sqrt_psd = gp.prior_linearized(Xs=Xs)

                # Specify standard normal prior in the coefficients.  The number of which
                # is given by the number of basis vectors, which is also saved in the GP object
                # as m_star.
                beta = pm.Normal("beta", size=gp.m_star)

                # The (non-centered) GP approximation is given by
                f = pm.Deterministic("f", phi @ (beta * sqrt_psd))

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

        # Index Xs using input_dim and active_dims of covariance function
        Xs, _ = self.cov_func._slice(Xs)

        # If not provided, use Xs and c to set L
        S, self.L, self.c = set_boundaries(Xs, self.L, self.c, tl=pt)
        self._boundary_set = True

        eigvals = calc_eigenvalues(self.L, self.m, tl=pt)
        phi = calc_eigenvectors(Xs, self.L, eigvals, self.m, tl=pt)
        omega = pt.sqrt(eigvals)
        psd = self.cov_func.power_spectral_density(omega)

        i = int(self.drop_first == True)
        return phi[:, i:], pt.sqrt(psd[i:])

    def prior(self, name: str, X: TensorVariable, *args, **kwargs):
        R"""
        Returns the (approximate) GP prior distribution evaluated over the input locations `X`.

        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.
        dims: None
            Dimension name for the GP random variable.
        """
        self.X_mu = pt.mean(X, axis=0)
        phi, sqrt_psd = self.prior_linearized(X - self.X_mu)

        if self.parameterization == "noncentered":
            self.beta = pm.Normal(f"{name}_hsgp_coeffs_", size=self.m_star - int(self.drop_first))
            self.sqrt_psd = sqrt_psd
            f = self.mean_func(X) + phi @ (self.beta * self.sqrt_psd)

        elif self.parameterization == "centered":
            self.beta = pm.Normal(f"{name}_hsgp_coeffs_", sigma=sqrt_psd)
            f = self.mean_func(X) + phi @ self.beta

        self.f = pm.Deterministic(name, f, dims=kwargs.get("dims"))
        return self.f

    def _build_conditional(self, Xnew):
        try:
            beta, X_mu = self.beta, self.X_mu
        except AttributeError:
            raise ValueError(
                "Prior is not set, can't create a conditional.  Call `.prior(name, X)` first."
            )

        Xnew, _ = self.cov_func._slice(Xnew)
        eigvals = calc_eigenvalues(self.L, self.m, tl=pt)
        phi = calc_eigenvectors(Xnew - X_mu, self.L, eigvals, self.m, tl=pt)
        i = int(self.drop_first == True)

        if self.parameterization == "noncentered":
            return self.mean_func(Xnew) + phi[:, i:] @ (beta * self.sqrt_psd)

        elif self.parameterization == "centered":
            return self.mean_func(Xnew) + phi[:, i:] @ beta

    def conditional(self, name: str, Xnew: TensorVariable, *args, **kwargs):
        R"""
        Returns the (approximate) conditional distribution evaluated over new input locations
        `Xnew`.  If using the

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.
        kwargs: dict-like
            Optional arguments such as `dims`.
        """
        fnew = self._build_conditional(Xnew)
        return pm.Deterministic(name, fnew, dims=kwargs.get("dims"))
