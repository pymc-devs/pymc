#   Copyright 2022 The PyMC Developers
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

from typing import Optional, Sequence

import numpy as np
import pytensor.tensor as at

import pymc as pm

from pymc.gp.cov import Covariance
from pymc.gp.gp import Base
from pymc.gp.mean import Mean, Zero


class HSGP(Base):
    R"""
    Hilbert Space Gaussian process

    The `gp.HSGP` class is an implementation of the Hilbert Space Gaussian process.  This
    approximation is a linear model that uses a fixed set of basis vectors, whose coeficients are
    random functions of a stationary covariance function's power spectral density.  Like
    `gp.Latent`, it does not assume a Gaussian noise model and can be used with any likelihood or as
    a component anywhere within a model.  Also like `gp.Latent`, it has `prior` and `conditional`
    methods.  It additonally has an `approx_K` method which returns the approximate covariance
    matrix.  It supports a limited subset of additive covariances.

    For information on choosing appropriate `m`, `L`, and `c`, refer Ruitort-Mayol et. al. or to the
    pymc examples documentation.

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
            # condition multiplier `c` uses 4 * half range.
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
        c: float = 1.5,
        drop_first=False,
        *,
        mean_func: Mean = Zero(),
        cov_func: Optional[Covariance] = None,
    ):
        arg_err_msg = (
            "`m` and L, if provided, must be lists or tuples, with one element per active "
            "dimension of the kernel or covariance function."
        )
        try:
            if len(m) != cov_func.D:
                raise ValueError(arg_err_msg)
        except TypeError as e:
            raise ValueError(arg_err_msg) from e

        if L is not None and len(L) != cov_func.D:
            raise ValueError(arg_err_msg)

        if L is None and c < 1.2:
            warnings.warn(
                "Most applications will require a `c >= 1.2` for accuracy at the boundaries of the "
                "domain."
            )

        self.drop_first = drop_first
        self.m = m
        self.L = L
        self.c = c
        self.D = cov_func.D

        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise NotImplementedError("Additive HSGPs aren't supported ")

    def _set_boundary(self, X):
        """Make L from X and c if L is not passed in."""
        if self.L is None:
            # Define new L based on c and X range
            La = at.abs(at.min(X, axis=0))
            Lb = at.abs(at.max(X, axis=0))
            self.L = self.c * at.max(at.stack((La, Lb)), axis=0)
        else:
            self.L = at.as_tensor_variable(self.L)

    @staticmethod
    def _eigendecomposition(X, L, m, D):
        """Construct the eigenvalues and eigenfunctions of the Laplace operator."""
        m_star = at.prod(m)
        S = np.meshgrid(*[np.arange(1, 1 + m[d]) for d in range(D)])
        S = np.vstack([s.flatten() for s in S]).T
        eigvals = at.square((np.pi * S) / (2 * L))
        phi = at.ones((X.shape[0], m_star))
        for d in range(D):
            c = 1.0 / np.sqrt(L[d])
            phi *= c * at.sin(at.sqrt(eigvals[:, d]) * (at.tile(X[:, d][:, None], m_star) + L[d]))
        omega = at.sqrt(eigvals)
        return omega, phi, m_star

    def approx_K(self, X, L, m):
        """A helper function which gives the approximate kernel or covariance matrix K. This can be
        helpful when trying to see how well an approximation may work.
        """
        X, _ = self.cov_func._slice(X)
        omega, phi, _ = self._eigendecomposition(X, self.L, self.m, self.cov_func.D)
        psd = self.cov_func.psd(omega)
        return at.dot(phi * psd, at.transpose(phi))

    def prior(self, name, X, dims=None):
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

        X, _ = self.cov_func._slice(X)
        self._set_boundary(X)
        omega, phi, m_star = self._eigendecomposition(X, self.L, self.m, self.D)
        psd = self.cov_func.psd(omega)

        if self.drop_first:
            self.beta = pm.Normal(f"{name}_coeffs_", size=m_star - 1)
            self.f = pm.Deterministic(
                name,
                self.mean_func(X) + at.squeeze(at.dot(phi[:, 1:], self.beta * psd[1:])),
                dims=dims,
            )
        else:
            self.beta = pm.Normal(f"{name}_coeffs_", size=m_star)
            self.f = pm.Deterministic(
                name,
                self.mean_func(X) + at.squeeze(at.dot(phi, self.beta * at.sqrt(psd))),
                dims=dims,
            )
        return self.f

    def _build_conditional(self, name, Xnew):
        Xnew, _ = self.cov_func._slice(Xnew)
        omega, phi, _ = self._eigendecomposition(Xnew, self.L, self.m, self.D)
        psd = self.cov_func.psd(omega)
        return self.mean_func(Xnew) + at.squeeze(at.dot(phi, self.beta * psd))

    def conditional(self, name, Xnew, dims=None):
        R"""
        Returns the (approximate) conditional distribution evaluated over new input locations
        `Xnew`.

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.
        dims: None
            Dimension name for the GP random variable.
        """
        fnew = self._build_conditional(name, Xnew)
        return pm.Deterministic(name, fnew, dims=dims)
