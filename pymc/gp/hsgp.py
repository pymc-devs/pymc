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

from typing import Optional, Sequence, Union

import numpy as np
import pytensor.tensor as pt

import pymc as pm

from pymc.gp.cov import Covariance
from pymc.gp.gp import Base
from pymc.gp.mean import Mean, Zero


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
    the pymc examples documentation.

    Parameters
    ----------
    m: list
        The number of basis vectors to use for each active dimension (covariance parameter
        `active_dim`).
    L: list
        The boundary of the space for each `active_dim`.  It is called the boundary condition.
        Choose L such that the domain `[-L, L]` contains all points in the column of X given by the
        `active_dim`.
    c: 1.5
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
        X = np.random.randn(100, 3)

        with pm.Model() as model:
            # Specify the covariance function.  Three input dimensions, but we only want to use the
            # last two.
            cov_func = pm.gp.cov.ExpQuad(3, ls=0.1, active_dims=[1, 2])

            # Specify the HSGP.  Use 50 basis vectors across each active dimension, [1, 2]  for a
            # total of 50 * 50 = 2500.  The range of the data is inferred from X, and the boundary
            # condition multiplier `c` uses 4 * half range of the data, `L`.
            gp = pm.gp.HSGP(m=[50, 50], c=4.0, cov_func=cov_func)

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
        self._boundary_set = False

        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise NotImplementedError("Additive HSGPs aren't supported ")

    def _set_boundary(self, X):
        """Make L from X and c if L is not passed in."""
        if not self._boundary_set:
            if self.L is None:
                # Define new L based on c and X range
                La = pt.abs(pt.min(X, axis=0))
                Lb = pt.abs(pt.max(X, axis=0))
                self.L = self.c * pt.max(pt.stack((La, Lb)), axis=0)
            else:
                self.L = pt.as_tensor_variable(self.L)

            self._boundary_set = True

    def prior_components(self, X: Union[np.ndarray, pt.TensorVariable]):
        """Return the basis and coefficient priors, whose product is the HSGP.  It can be useful
        to bypass the GP API and work with the basis and coefficients directly for models with i.e.
        multiple GPs.  It returns the basis `phi` and the power spectral density, `psd`.  The GP
        `f` can be formed by `f = phi @ (pm.Normal("hsgp_coeffs", size=psd.size) * psd)`.

        Parameters
        ----------
        X: array-like
            Function input values.
        """
        X, _ = self.cov_func._slice(X)
        self._set_boundary(X)

        omega, phi, m_star = self._eigendecomposition(X, self.L, self.m, self.n_dims)
        psd = self.cov_func.power_spectral_density(omega)

        i = int(self.drop_first == True)
        return phi[:, i:], pt.sqrt(psd[i:])

    @staticmethod
    def _eigendecomposition(X, L, m, n_dims):
        """Construct the eigenvalues and eigenfunctions of the Laplace operator."""
        m_star = pt.prod(m)
        S = np.meshgrid(*[np.arange(1, 1 + m[d]) for d in range(n_dims)])
        S = np.vstack([s.flatten() for s in S]).T
        eigvals = pt.square((np.pi * S) / (2 * L))
        phi = pt.ones((X.shape[0], m_star))
        for d in range(n_dims):
            c = 1.0 / np.sqrt(L[d])
            phi *= c * pt.sin(pt.sqrt(eigvals[:, d]) * (pt.tile(X[:, d][:, None], m_star) + L[d]))
        omega = pt.sqrt(eigvals)
        return omega, phi, m_star

    def prior(self, name: str, X: Union[np.ndarray, pt.TensorVariable], *args, **kwargs):
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

        phi, sqrt_psd = self.prior_components(X)

        if self.parameterization == "noncentered":
            self.beta = pm.Normal(f"{name}_hsgp_coeffs_", size=sqrt_psd.size)
            f = self.mean_func(X) + phi @ (self.beta * sqrt_psd)

        elif self.parameterization == "centered":
            self.beta = pm.Normal(f"{name}_hsgp_coeffs_", sigma=sqrt_psd, size=sqrt_psd.size)
            f = self.mean_func(X) + phi @ self.beta

        self.f = pm.Deterministic(name, f, dims=kwargs.get("dims"))
        return self.f

    def _build_conditional(self, Xnew):
        try:
            beta = self.beta
        except AttributeError:
            raise ValueError(
                "Prior is not set, can't create a conditional.  Call `.prior(name, X)` first."
            )

        Xnew, _ = self.cov_func._slice(Xnew)
        omega, phi, _ = self._eigendecomposition(Xnew, self.L, self.m, self.n_dims)
        i = int(self.drop_first == True)

        if self.parameterization == "noncentered":
            psd = self.cov_func.power_spectral_density(omega)
            return self.mean_func(Xnew) + phi[:, i:] @ (beta * pt.sqrt(psd[i:]))

        elif self.parameterization == "centered":
            return self.mean_func(Xnew) + phi[:, i:] @ beta

    def conditional(self, name: str, Xnew: Union[np.ndarray, pt.TensorVariable], *args, **kwargs):
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
