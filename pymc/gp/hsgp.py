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

        # Record whether or not L instead of c was passed on construction.
        self.L_given = self.L is not None
        
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise NotImplementedError("Additive HSGPs aren't supported ")

    def _set_boundary(self, Xs):
        """Make L from X and c if L is not passed in."""
        if not self._boundary_set:
            
            self.S = pt.max(pt.abs(Xs), axis=0)
            
            if self.L is None:
                self.L = self.c * self.S
            
            else:
                self.c = self.L / self.S
                self.L = pt.as_tensor_variable(self.L)

            self._boundary_set = True

    @property
    def _eigenvalues(self):
        S = np.meshgrid(*[np.arange(1, 1 + self.m[d]) for d in range(self.n_dims)])
        S = np.vstack([s.flatten() for s in S]).T
        return pt.square((np.pi * S) / (2 * self.L))

    def _eigenvectors(self, X):
        phi = pt.ones((X.shape[0], self.m_star))
        for d in range(self.n_dims):
            c = 1.0 / np.sqrt(self.L[d])
            term1 = pt.sqrt(self._eigenvalues[:, d])
            term2 = pt.tile(X[:, d][:, None], self.m_star) + self.L[d]
            phi *= c * pt.sin(term1 * term2)
        return phi

    def prior_linearized(self, X: Union[np.ndarray, pt.TensorVariable]):
        """ Returns the linearized version of the HSGP, the Laplace eigenfunctions and the square
        root of the power spectral density needed to create the GP.  This function allows the user
        to bypass the GP interface and work directly with the basis and coefficients directly.  
        This format allows the user to create predictions using `pm.set_data` similarly to a 
        linear model.  It also enables computational speed ups in multi-GP models since they may
        share the same basis.  The return values are the Laplace eigenfunctions `phi`, and the 
        square root of the power spectral density.

        Correct results when using `prior_linearized` in tandem with `pm.set_data` and 
        `pm.MutableData` require two conditions.  First, one must specify `L` instead of `c` when
        the GP is constructed.  If not, a RuntimeError is raised.  Second, the `X` needs to be 
        zero centered, so it's mean must be subtracted.  An example is given below. 

        Parameters
        ----------
        X: array-like
            Function input values.
    
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
                phi, sqrt_psd = gp.prior_linearized(Xs)

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
        if not self.L_given:
            warnings.warn(
                "Since `c` was given instead of `L` on construction, usage of `model.set_data` "
                "will result in an incorrect posterior predictive.  Ignore this warning if you "
                "dont plan on creating predictions this way.  To remove this warning pass `L` "
                "instead of `c` to the HSGP constructor."
            )

        X, _ = self.cov_func._slice(X)
        
        self._set_boundary(X)

        phi = self._eigenvectors(X)
        omega = pt.sqrt(self._eigenvalues)
        psd = self.cov_func.power_spectral_density(omega)
        
        i = int(self.drop_first == True)
        return phi[:, i:], pt.sqrt(psd[i:])

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
        self.X_mu = pt.mean(X, axis=0)
        phi, sqrt_psd = self.prior_linearized(X - self.X_mu)

        if self.parameterization == "noncentered":
            self.beta = pm.Normal(f"{name}_hsgp_coeffs_", size=self.m_star)
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
        phi = self._eigenvectors(Xnew - X_mu)
        i = int(self.drop_first == True)

        if self.parameterization == "noncentered":
            return self.mean_func(Xnew) + phi[:, i:] @ (beta * self.sqrt_psd)

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
