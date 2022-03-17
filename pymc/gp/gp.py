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

import aesara.tensor as at
import numpy as np

from aesara.tensor.nlinalg import eigh

import pymc as pm

from pymc.gp.cov import Constant, Covariance
from pymc.gp.mean import Zero
from pymc.gp.util import (
    JITTER_DEFAULT,
    cholesky,
    conditioned_vars,
    infer_size,
    replace_with_values,
    solve_lower,
    solve_upper,
    stabilize,
)
from pymc.math import cartesian, kron_diag, kron_dot, kron_solve_lower, kron_solve_upper

__all__ = ["Latent", "Marginal", "TP", "MarginalApprox", "LatentKron", "MarginalKron"]


class Base:
    R"""
    Base class.
    """

    def __init__(self, *, mean_func=Zero(), cov_func=Constant(0.0)):
        self.mean_func = mean_func
        self.cov_func = cov_func

    def __add__(self, other):
        same_attrs = set(self.__dict__.keys()) == set(other.__dict__.keys())
        if not isinstance(self, type(other)) or not same_attrs:
            raise TypeError("Cannot add different GP types")
        mean_total = self.mean_func + other.mean_func
        cov_total = self.cov_func + other.cov_func
        return self.__class__(mean_func=mean_total, cov_func=cov_total)

    def prior(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def conditional(self, name, Xnew, *args, **kwargs):
        raise NotImplementedError

    def predict(self, Xnew, point=None, given=None, diag=False, model=None):
        raise NotImplementedError


@conditioned_vars(["X", "f"])
class Latent(Base):
    R"""
    Latent Gaussian process.

    The `gp.Latent` class is a direct implementation of a GP.  No additive
    noise is assumed.  It is called "Latent" because the underlying function
    values are treated as latent variables.  It has a `prior` method and a
    `conditional` method.  Given a mean and covariance function the
    function :math:`f(x)` is modeled as,

    .. math::

       f(x) \sim \mathcal{GP}\left(\mu(x), k(x, x')\right)

    Use the `prior` and `conditional` methods to actually construct random
    variables representing the unknown, or latent, function whose
    distribution is the GP prior or GP conditional.  This GP implementation
    can be used to implement regression on data that is not normally
    distributed.  For more information on the `prior` and `conditional` methods,
    see their docstrings.

    Parameters
    ----------
    cov_func: None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

        # A one dimensional column vector of inputs.
        X = np.linspace(0, 1, 10)[:, None]

        with pm.Model() as model:
            # Specify the covariance function.
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.Latent(cov_func=cov_func)

            # Place a GP prior over the function f.
            f = gp.prior("f", X=X)

        ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def __init__(self, *, mean_func=Zero(), cov_func=Constant(0.0)):
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def _build_prior(self, name, X, reparameterize=True, jitter=JITTER_DEFAULT, **kwargs):
        mu = self.mean_func(X)
        cov = stabilize(self.cov_func(X), jitter)
        if reparameterize:
            size = infer_size(X, kwargs.pop("size", None))
            v = pm.Normal(name + "_rotated_", mu=0.0, sigma=1.0, size=size, **kwargs)
            f = pm.Deterministic(name, mu + cholesky(cov).dot(v))
        else:
            f = pm.MvNormal(name, mu=mu, cov=cov, **kwargs)
        return f

    def prior(self, name, X, reparameterize=True, jitter=JITTER_DEFAULT, **kwargs):
        R"""
        Returns the GP prior distribution evaluated over the input
        locations `X`.

        This is the prior probability over the space
        of functions described by its mean and covariance function.

        .. math::

           f \mid X \sim \text{MvNormal}\left( \mu(X), k(X, X') \right)

        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.
        reparameterize: bool
            Reparameterize the distribution by rotating the random
            variable by the Cholesky factor of the covariance matrix.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  Default value is 1e-6.
        **kwargs
            Extra keyword arguments that are passed to distribution constructor.
        """

        f = self._build_prior(name, X, reparameterize, jitter, **kwargs)
        self.X = X
        self.f = f
        return f

    def _get_given_vals(self, given):
        if given is None:
            given = {}
        if "gp" in given:
            cov_total = given["gp"].cov_func
            mean_total = given["gp"].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ["X", "f"]):
            X, f = given["X"], given["f"]
        else:
            X, f = self.X, self.f
        return X, f, cov_total, mean_total

    def _build_conditional(self, Xnew, X, f, cov_total, mean_total, jitter):
        Kxx = cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        L = cholesky(stabilize(Kxx, jitter))
        A = solve_lower(L, Kxs)
        v = solve_lower(L, f - mean_total(X))
        mu = self.mean_func(Xnew) + at.dot(at.transpose(A), v)
        Kss = self.cov_func(Xnew)
        cov = Kss - at.dot(at.transpose(A), A)
        return mu, cov

    def conditional(self, name, Xnew, given=None, jitter=0.0, **kwargs):
        R"""
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.

        Given a set of function values `f` that
        the GP prior was over, the conditional distribution over a
        set of new points, `f_*` is

        .. math::

           f_* \mid f, X, X_* \sim \mathcal{GP}\left(
               K(X_*, X) K(X, X)^{-1} f \,,
               K(X_*, X_*) - K(X_*, X) K(X, X)^{-1} K(X, X_*) \right)

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.
        given: dict
            Can optionally take as key value pairs: `X`, `y`, `noise`,
            and `gp`.  See the section in the documentation on additive GP
            models in PyMC for more information.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  For conditionals
            the default value is 0.0.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """
        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, *givens, jitter)
        return pm.MvNormal(name, mu=mu, cov=cov, **kwargs)


@conditioned_vars(["X", "f", "nu"])
class TP(Latent):
    r"""
    Student's T process prior.

    The usage is nearly identical to that of `gp.Latent`.  The differences
    are that it must be initialized with a degrees of freedom parameter, and
    TP is not additive.  Given a mean and covariance function, and a degrees of
    freedom parameter, the function :math:`f(x)` is modeled as,

    .. math::

       f(X) \sim \mathcal{TP}\left( \mu(X), k(X, X'), \\nu \\right)


    Parameters
    ----------
    cov_func : None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func : None, instance of Mean
        The mean function.  Defaults to zero.
    nu : float
        The degrees of freedom

    References
    ----------
    -   Shah, A., Wilson, A. G., and Ghahramani, Z. (2014).  Student-t
        Processes as Alternatives to Gaussian Processes.  arXiv preprint arXiv:1402.4306.
    """

    def __init__(self, *, mean_func=Zero(), cov_func=Constant(0.0), nu=None):
        if nu is None:
            raise ValueError("Student's T process requires a degrees of freedom parameter, 'nu'")
        self.nu = nu
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise TypeError("Student's T processes aren't additive")

    def _build_prior(self, name, X, reparameterize=True, jitter=JITTER_DEFAULT, **kwargs):
        mu = self.mean_func(X)
        cov = stabilize(self.cov_func(X), jitter)
        if reparameterize:
            size = infer_size(X, kwargs.pop("size", None))
            v = pm.StudentT(name + "_rotated_", mu=0.0, sigma=1.0, nu=self.nu, size=size, **kwargs)
            f = pm.Deterministic(name, mu + cholesky(cov).dot(v))
        else:
            f = pm.MvStudentT(name, nu=self.nu, mu=mu, cov=cov, **kwargs)
        return f

    def prior(self, name, X, reparameterize=True, jitter=JITTER_DEFAULT, **kwargs):
        R"""
        Returns the TP prior distribution evaluated over the input
        locations `X`.

        This is the prior probability over the space
        of functions described by its mean and covariance function.

        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.
        reparameterize: bool
            Reparameterize the distribution by rotating the random
            variable by the Cholesky factor of the covariance matrix.
        **kwargs
            Extra keyword arguments that are passed to distribution constructor.
        """

        f = self._build_prior(name, X, reparameterize, jitter, **kwargs)
        self.X = X
        self.f = f
        return f

    def _build_conditional(self, Xnew, X, f, jitter):
        Kxx = self.cov_func(X)
        Kxs = self.cov_func(X, Xnew)
        Kss = self.cov_func(Xnew)
        L = cholesky(stabilize(Kxx, jitter))
        A = solve_lower(L, Kxs)
        cov = Kss - at.dot(at.transpose(A), A)
        v = solve_lower(L, f - self.mean_func(X))
        mu = self.mean_func(Xnew) + at.dot(at.transpose(A), v)
        beta = at.dot(v, v)
        nu2 = self.nu + X.shape[0]
        covT = (self.nu + beta - 2) / (nu2 - 2) * cov
        return nu2, mu, covT

    def conditional(self, name, Xnew, jitter=0.0, **kwargs):
        R"""
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.

        Given a set of function values `f` that
        the TP prior was over, the conditional distribution over a
        set of new points, `f_*` is

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  For conditionals
            the default value is 0.0.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        X = self.X
        f = self.f
        nu2, mu, cov = self._build_conditional(Xnew, X, f, jitter)
        return pm.MvStudentT(name, nu=nu2, mu=mu, cov=cov, **kwargs)


@conditioned_vars(["X", "y", "noise"])
class Marginal(Base):
    R"""
    Marginal Gaussian process.

    The `gp.Marginal` class is an implementation of the sum of a GP
    prior and additive noise.  It has `marginal_likelihood`, `conditional`
    and `predict` methods.  This GP implementation can be used to
    implement regression on data that is normally distributed.  For more
    information on the `prior` and `conditional` methods, see their docstrings.

    Parameters
    ----------
    cov_func: None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

        # A one dimensional column vector of inputs.
        X = np.linspace(0, 1, 10)[:, None]

        with pm.Model() as model:
            # Specify the covariance function.
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.Marginal(cov_func=cov_func)

            # Place a GP prior over the function f.
            sigma = pm.HalfCauchy("sigma", beta=3)
            y_ = gp.marginal_likelihood("y", X=X, y=y, noise=sigma)

        ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def _build_marginal_likelihood(self, X, noise, jitter):
        mu = self.mean_func(X)
        Kxx = self.cov_func(X)
        Knx = noise(X)
        cov = Kxx + Knx
        return mu, stabilize(cov, jitter)

    def marginal_likelihood(self, name, X, y, noise, jitter=0.0, is_observed=True, **kwargs):
        R"""
        Returns the marginal likelihood distribution, given the input
        locations `X` and the data `y`.

        This is integral over the product of the GP prior and a normal likelihood.

        .. math::

           y \mid X,\theta \sim \int p(y \mid f,\, X,\, \theta) \, p(f \mid X,\, \theta) \, df

        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        y: array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        noise: scalar, Variable, or Covariance
            Standard deviation of the Gaussian noise.  Can also be a Covariance for
            non-white noise.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  Default value is 0.0.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        if not isinstance(noise, Covariance):
            noise = pm.gp.cov.WhiteNoise(noise)
        mu, cov = self._build_marginal_likelihood(X, noise, jitter)
        self.X = X
        self.y = y
        self.noise = noise
        if is_observed:
            return pm.MvNormal(name, mu=mu, cov=cov, observed=y, **kwargs)
        else:
            warnings.warn(
                "The 'is_observed' argument has been deprecated.  If the GP is "
                "unobserved use gp.Latent instead.",
                FutureWarning,
            )
            return pm.MvNormal(name, mu=mu, cov=cov, **kwargs)

    def _get_given_vals(self, given):
        if given is None:
            given = {}

        if "gp" in given:
            cov_total = given["gp"].cov_func
            mean_total = given["gp"].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ["X", "y", "noise"]):
            X, y, noise = given["X"], given["y"], given["noise"]
            if not isinstance(noise, Covariance):
                noise = pm.gp.cov.WhiteNoise(noise)
        else:
            X, y, noise = self.X, self.y, self.noise
        return X, y, noise, cov_total, mean_total

    def _build_conditional(
        self, Xnew, pred_noise, diag, X, y, noise, cov_total, mean_total, jitter
    ):
        Kxx = cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Knx = noise(X)
        rxx = y - mean_total(X)
        L = cholesky(stabilize(Kxx, jitter) + Knx)
        A = solve_lower(L, Kxs)
        v = solve_lower(L, rxx)
        mu = self.mean_func(Xnew) + at.dot(at.transpose(A), v)
        if diag:
            Kss = self.cov_func(Xnew, diag=True)
            var = Kss - at.sum(at.square(A), 0)
            if pred_noise:
                var += noise(Xnew, diag=True)
            return mu, var
        else:
            Kss = self.cov_func(Xnew)
            cov = Kss - at.dot(at.transpose(A), A)
            if pred_noise:
                cov += noise(Xnew)
            return mu, cov if pred_noise else stabilize(cov, jitter)

    def conditional(self, name, Xnew, pred_noise=False, given=None, jitter=0.0, **kwargs):
        R"""
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.

        Given a set of function values `f` that the GP prior was over, the
        conditional distribution over a set of new points, `f_*` is:

        .. math::

           f_* \mid f, X, X_* \sim \mathcal{GP}\left(
               K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} f \,,
               K(X_*, X_*) - K(X_*, X) [K(X, X) + K_{n}(X, X)]^{-1} K(X, X_*) \right)

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Can optionally take as key value pairs: `X`, `y`, `noise`,
            and `gp`.  See the section in the documentation on additive GP
            models in PyMC for more information.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  For conditionals
            the default value is 0.0.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, pred_noise, False, *givens, jitter)
        return pm.MvNormal(name, mu=mu, cov=cov, **kwargs)

    def predict(
        self, Xnew, point=None, diag=False, pred_noise=False, given=None, jitter=0.0, model=None
    ):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as numpy arrays, given a `point`, such as the MAP
        estimate or a sample from a `trace`.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        point: pymc.model.Point
            A specific point to condition on.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  For conditionals
            the default value is 0.0.
        """
        if given is None:
            given = {}
        mu, cov = self._predict_at(Xnew, diag, pred_noise, given, jitter)
        return replace_with_values([mu, cov], replacements=point, model=model)

    def _predict_at(self, Xnew, diag=False, pred_noise=False, given=None, jitter=0.0):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as symbolic variables.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Same as `conditional` method.
        """
        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, pred_noise, diag, *givens, jitter)
        return mu, cov


@conditioned_vars(["X", "Xu", "y", "sigma"])
class MarginalApprox(Marginal):
    R"""
    Approximate marginal Gaussian process.

    The `gp.MarginalApprox` class is an implementation of the sum of a GP
    prior and additive noise.  It has `marginal_likelihood`, `conditional`
    and `predict` methods.  This GP implementation can be used to
    implement regression on data that is normally distributed.  The
    available approximations are:

    - DTC: Deterministic Training Conditional
    - FITC: Fully independent Training Conditional
    - VFE: Variational Free Energy

    Parameters
    ----------
    cov_func: None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.
    approx: string
        The approximation to use.  Must be one of `VFE`, `FITC` or `DTC`.
        Default is VFE.

    Examples
    --------
    .. code:: python

        # A one dimensional column vector of inputs.
        X = np.linspace(0, 1, 10)[:, None]

        # A smaller set of inducing inputs
        Xu = np.linspace(0, 1, 5)[:, None]

        with pm.Model() as model:
            # Specify the covariance function.
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.MarginalApprox(cov_func=cov_func, approx="FITC")

            # Place a GP prior over the function f.
            sigma = pm.HalfCauchy("sigma", beta=3)
            y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, y=y, sigma=sigma)

        ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        Xnew = np.linspace(-1, 2, 50)[:, None]

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)

    References
    ----------
    -   Quinonero-Candela, J., and Rasmussen, C. (2005). A Unifying View of
        Sparse Approximate Gaussian Process Regression.

    -   Titsias, M. (2009). Variational Learning of Inducing Variables in
        Sparse Gaussian Processes.

    -   Bauer, M., van der Wilk, M., and Rasmussen, C. E. (2016). Understanding
        Probabilistic Sparse Gaussian Process Approximations.
    """

    _available_approx = ("FITC", "VFE", "DTC")

    def __init__(self, approx="VFE", *, mean_func=Zero(), cov_func=Constant(0.0)):
        if approx not in self._available_approx:
            raise NotImplementedError(approx)
        self.approx = approx
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        # new_gp will default to FITC approx
        new_gp = super().__add__(other)
        # make sure new gp has correct approx
        if not self.approx == other.approx:
            raise TypeError("Cannot add GPs with different approximations")
        new_gp.approx = self.approx
        return new_gp

    # Use y as first argument, so that we can use functools.partial
    # in marginal_likelihood instead of lambda. This makes pickling
    # possible.
    def _build_marginal_likelihood_logp(self, y, X, Xu, sigma, jitter):
        sigma2 = at.square(sigma)
        Kuu = self.cov_func(Xu)
        Kuf = self.cov_func(Xu, X)
        Luu = cholesky(stabilize(Kuu, jitter))
        A = solve_lower(Luu, Kuf)
        Qffd = at.sum(A * A, 0)
        if self.approx == "FITC":
            Kffd = self.cov_func(X, diag=True)
            Lamd = at.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
            trace = 0.0
        elif self.approx == "VFE":
            Lamd = at.ones_like(Qffd) * sigma2
            trace = (1.0 / (2.0 * sigma2)) * (
                at.sum(self.cov_func(X, diag=True)) - at.sum(at.sum(A * A, 0))
            )
        else:  # DTC
            Lamd = at.ones_like(Qffd) * sigma2
            trace = 0.0
        A_l = A / Lamd
        L_B = cholesky(at.eye(Xu.shape[0]) + at.dot(A_l, at.transpose(A)))
        r = y - self.mean_func(X)
        r_l = r / Lamd
        c = solve_lower(L_B, at.dot(A, r_l))
        constant = 0.5 * X.shape[0] * at.log(2.0 * np.pi)
        logdet = 0.5 * at.sum(at.log(Lamd)) + at.sum(at.log(at.diag(L_B)))
        quadratic = 0.5 * (at.dot(r, r_l) - at.dot(c, c))
        return -1.0 * (constant + logdet + quadratic + trace)

    def marginal_likelihood(
        self, name, X, Xu, y, noise=None, is_observed=True, jitter=0.0, **kwargs
    ):
        R"""
        Returns the approximate marginal likelihood distribution, given the input
        locations `X`, inducing point locations `Xu`, data `y`, and white noise
        standard deviations `sigma`.

        Parameters
        ----------
        name: string
            Name of the random variable
        X: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        Xu: array-like
            The inducing points.  Must have the same number of columns as `X`.
        y: array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        noise: scalar, Variable
            Standard deviation of the Gaussian noise.
        is_observed: bool
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  Default value is 0.0.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        self.X = X
        self.Xu = Xu
        self.y = y

        if noise is None:
            raise ValueError("noise argument must be specified")
        else:
            self.sigma = noise

        if is_observed:
            return pm.DensityDist(
                name,
                X,
                Xu,
                self.sigma,
                jitter,
                logp=self._build_marginal_likelihood_logp,
                observed=y,
                ndims_params=[2, 2, 0],
                size=X.shape[0],
                **kwargs,
            )
        else:
            warnings.warn(
                "The 'is_observed' argument has been deprecated.  If the GP is "
                "unobserved use gp.Latent instead.",
                FutureWarning,
            )
            return pm.DensityDist(
                name,
                X,
                Xu,
                self.sigma,
                jitter,
                logp=self._build_marginal_likelihood_logp,
                observed=y,
                ndims_params=[2, 2, 0],
                # ndim_supp=1,
                size=X.shape[0],
                **kwargs,
            )

    def _build_conditional(
        self, Xnew, pred_noise, diag, X, Xu, y, sigma, cov_total, mean_total, jitter
    ):
        sigma2 = at.square(sigma)
        Kuu = cov_total(Xu)
        Kuf = cov_total(Xu, X)
        Luu = cholesky(stabilize(Kuu, jitter))
        A = solve_lower(Luu, Kuf)
        Qffd = at.sum(A * A, 0)
        if self.approx == "FITC":
            Kffd = cov_total(X, diag=True)
            Lamd = at.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
        else:  # VFE or DTC
            Lamd = at.ones_like(Qffd) * sigma2
        A_l = A / Lamd
        L_B = cholesky(at.eye(Xu.shape[0]) + at.dot(A_l, at.transpose(A)))
        r = y - mean_total(X)
        r_l = r / Lamd
        c = solve_lower(L_B, at.dot(A, r_l))
        Kus = self.cov_func(Xu, Xnew)
        As = solve_lower(Luu, Kus)
        mu = self.mean_func(Xnew) + at.dot(at.transpose(As), solve_upper(at.transpose(L_B), c))
        C = solve_lower(L_B, As)
        if diag:
            Kss = self.cov_func(Xnew, diag=True)
            var = Kss - at.sum(at.square(As), 0) + at.sum(at.square(C), 0)
            if pred_noise:
                var += sigma2
            return mu, var
        else:
            cov = self.cov_func(Xnew) - at.dot(at.transpose(As), As) + at.dot(at.transpose(C), C)
            if pred_noise:
                cov += sigma2 * at.identity_like(cov)
            return mu, cov if pred_noise else stabilize(cov, jitter)

    def _get_given_vals(self, given):
        if given is None:
            given = {}
        if "gp" in given:
            cov_total = given["gp"].cov_func
            mean_total = given["gp"].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ["X", "Xu", "y", "sigma"]):
            X, Xu, y, sigma = given["X"], given["Xu"], given["y"], given["sigma"]
        else:
            X, Xu, y, sigma = self.X, self.Xu, self.y, self.sigma
        return X, Xu, y, sigma, cov_total, mean_total

    def conditional(self, name, Xnew, pred_noise=False, given=None, jitter=0.0, **kwargs):
        R"""
        Returns the approximate conditional distribution of the GP evaluated over
        new input locations `Xnew`.

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given: dict
            Can optionally take as key value pairs: `X`, `Xu`, `y`, `noise`,
            and `gp`.  See the section in the documentation on additive GP
            models in PyMC for more information.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  For conditionals
            the default value is 0.0.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """

        givens = self._get_given_vals(given)
        mu, cov = self._build_conditional(Xnew, pred_noise, False, *givens, jitter)
        return pm.MvNormal(name, mu=mu, cov=cov, **kwargs)


@conditioned_vars(["X", "Xu", "y", "sigma"])
class MarginalSparse(MarginalApprox):
    def __init__(self, approx="VFE", *, mean_func=Zero(), cov_func=Constant(0.0)):
        warnings.warn(
            "gp.MarginalSparse has been renamed to gp.MarginalApprox.",
            FutureWarning,
        )
        super().__init__(mean_func=mean_func, cov_func=cov_func, approx=approx)


@conditioned_vars(["Xs", "f"])
class LatentKron(Base):
    R"""
    Latent Gaussian process whose covariance is a tensor product kernel.

    The `gp.LatentKron` class is a direct implementation of a GP with a
    Kronecker structured covariance, without reference to any noise or
    specific likelihood.  The GP is constructed with the `prior` method,
    and the conditional GP over new input locations is constructed with
    the `conditional` method.  `conditional` and method.  For more
    information on these methods, see their docstrings.  This GP
    implementation can be used to model a Gaussian process whose inputs
    cover evenly spaced grids on more than one dimension.  `LatentKron`
    is relies on the `KroneckerNormal` distribution, see its docstring
    for more information.

    Parameters
    ----------
    cov_funcs: list of Covariance objects
        The covariance functions that compose the tensor (Kronecker) product.
        Defaults to [zero].
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

        # One dimensional column vectors of inputs
        X1 = np.linspace(0, 1, 10)[:, None]
        X2 = np.linspace(0, 2, 5)[:, None]
        Xs = [X1, X2]
        with pm.Model() as model:
            # Specify the covariance functions for each Xi
            cov_func1 = pm.gp.cov.ExpQuad(1, ls=0.1)  # Must accept X1 without error
            cov_func2 = pm.gp.cov.ExpQuad(1, ls=0.3)  # Must accept X2 without error

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.LatentKron(cov_funcs=[cov_func1, cov_func2])

            # ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        # Xnew need not be on a full grid
        Xnew1 = np.linspace(-1, 2, 10)[:, None]
        Xnew2 = np.linspace(0, 3, 10)[:, None]
        Xnew = np.concatenate((Xnew1, Xnew2), axis=1)  # Not full grid, works
        Xnew = pm.math.cartesian(Xnew1, Xnew2)  # Full grid, also works

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def __init__(self, *, mean_func=Zero(), cov_funcs=(Constant(0.0))):
        try:
            self.cov_funcs = list(cov_funcs)
        except TypeError:
            self.cov_funcs = [cov_funcs]
        cov_func = pm.gp.cov.Kron(self.cov_funcs)
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise TypeError("Additive, Kronecker-structured processes not implemented")

    def _build_prior(self, name, Xs, jitter, **kwargs):
        self.N = int(np.prod([len(X) for X in Xs]))
        mu = self.mean_func(cartesian(*Xs))
        chols = [cholesky(stabilize(cov(X), jitter)) for cov, X in zip(self.cov_funcs, Xs)]
        v = pm.Normal(name + "_rotated_", mu=0.0, sigma=1.0, size=self.N, **kwargs)
        f = pm.Deterministic(name, mu + at.flatten(kron_dot(chols, v)))
        return f

    def prior(self, name, Xs, jitter=JITTER_DEFAULT, **kwargs):
        """
        Returns the prior distribution evaluated over the input
        locations `Xs`.

        Parameters
        ----------
        name: string
            Name of the random variable
        Xs: list of array-like
            Function input values for each covariance function. Each entry
            must be passable to its respective covariance without error. The
            total covariance function is measured on the full grid
            `cartesian(*Xs)`.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  Default value is 1e-6.
        **kwargs
            Extra keyword arguments that are passed to the `KroneckerNormal`
            distribution constructor.
        """
        if len(Xs) != len(self.cov_funcs):
            raise ValueError("Must provide a covariance function for each X")

        f = self._build_prior(name, Xs, jitter, **kwargs)
        self.Xs = Xs
        self.f = f
        return f

    def _build_conditional(self, Xnew, jitter):
        Xs, f = self.Xs, self.f
        X = cartesian(*Xs)
        delta = f - self.mean_func(X)
        covs = [stabilize(cov(Xi), jitter) for cov, Xi in zip(self.cov_funcs, Xs)]
        chols = [cholesky(cov) for cov in covs]
        cholTs = [at.transpose(chol) for chol in chols]
        Kss = self.cov_func(Xnew)
        Kxs = self.cov_func(X, Xnew)
        Ksx = at.transpose(Kxs)
        alpha = kron_solve_lower(chols, delta)
        alpha = kron_solve_upper(cholTs, alpha)
        mu = at.dot(Ksx, alpha).ravel() + self.mean_func(Xnew)
        A = kron_solve_lower(chols, Kxs)
        cov = stabilize(Kss - at.dot(at.transpose(A), A), jitter)
        return mu, cov

    def conditional(self, name, Xnew, jitter=0.0, **kwargs):
        """
        Returns the conditional distribution evaluated over new input
        locations `Xnew`.

        `Xnew` will be split by columns and fed to the relevant
        covariance functions based on their `input_dim`. For example, if
        `cov_func1`, `cov_func2`, and `cov_func3` have `input_dim` of 2,
        1, and 4, respectively, then `Xnew` must have 7 columns and a
        covariance between the prediction points

        .. code:: python

            cov_func(Xnew) = cov_func1(Xnew[:, :2]) * cov_func1(Xnew[:, 2:3]) * cov_func1(Xnew[:, 3:])

        The distribution returned by `conditional` does not have a
        Kronecker structure regardless of whether the input points lie
        on a full grid.  Therefore, `Xnew` does not need to have grid
        structure.

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        jitter: scalar
            A small correction added to the diagonal of positive semi-definite
            covariance matrices to ensure numerical stability.  For conditionals
            the default value is 0.0.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """
        mu, cov = self._build_conditional(Xnew, jitter)
        return pm.MvNormal(name, mu=mu, cov=cov, **kwargs)


@conditioned_vars(["Xs", "y", "sigma"])
class MarginalKron(Base):
    R"""
    Marginal Gaussian process whose covariance is a tensor product kernel.

    The `gp.MarginalKron` class is an implementation of the sum of a
    Kronecker GP prior and additive white noise. It has
    `marginal_likelihood`, `conditional` and `predict` methods. This GP
    implementation can be used to efficiently implement regression on
    data that are normally distributed with a tensor product kernel and
    are measured on a full grid of inputs: `cartesian(*Xs)`.
    `MarginalKron` is based on the `KroneckerNormal` distribution, see
    its docstring for more information. For more information on the
    `prior` and `conditional` methods, see their docstrings.

    Parameters
    ----------
    cov_funcs: list of Covariance objects
        The covariance functions that compose the tensor (Kronecker) product.
        Defaults to [zero].
    mean_func: None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

        # One dimensional column vectors of inputs
        X1 = np.linspace(0, 1, 10)[:, None]
        X2 = np.linspace(0, 2, 5)[:, None]
        Xs = [X1, X2]
        y = np.random.randn(len(X1)*len(X2))  # toy data
        with pm.Model() as model:
            # Specify the covariance functions for each Xi
            cov_func1 = pm.gp.cov.ExpQuad(1, ls=0.1)  # Must accept X1 without error
            cov_func2 = pm.gp.cov.ExpQuad(1, ls=0.3)  # Must accept X2 without error

            # Specify the GP.  The default mean function is `Zero`.
            gp = pm.gp.MarginalKron(cov_funcs=[cov_func1, cov_func2])

            # Place a GP prior over the function f.
            sigma = pm.HalfCauchy("sigma", beta=3)
            y_ = gp.marginal_likelihood("y", Xs=Xs, y=y, sigma=sigma)

            # ...

        # After fitting or sampling, specify the distribution
        # at new points with .conditional
        # Xnew need not be on a full grid
        Xnew1 = np.linspace(-1, 2, 10)[:, None]
        Xnew2 = np.linspace(0, 3, 10)[:, None]
        Xnew = np.concatenate((Xnew1, Xnew2), axis=1)  # Not full grid, works
        Xnew = pm.math.cartesian(Xnew1, Xnew2)  # Full grid, also works

        with model:
            fcond = gp.conditional("fcond", Xnew=Xnew)
    """

    def __init__(self, *, mean_func=Zero(), cov_funcs=(Constant(0.0))):
        try:
            self.cov_funcs = list(cov_funcs)
        except TypeError:
            self.cov_funcs = [cov_funcs]
        cov_func = pm.gp.cov.Kron(self.cov_funcs)
        super().__init__(mean_func=mean_func, cov_func=cov_func)

    def __add__(self, other):
        raise TypeError("Additive, Kronecker-structured processes not implemented")

    def _build_marginal_likelihood(self, Xs):
        self.X = cartesian(*Xs)
        mu = self.mean_func(self.X)
        covs = [f(X) for f, X in zip(self.cov_funcs, Xs)]
        return mu, covs

    def _check_inputs(self, Xs, y):
        N = int(np.prod([len(X) for X in Xs]))
        if len(Xs) != len(self.cov_funcs):
            raise ValueError("Must provide a covariance function for each X")
        if N != len(y):
            raise ValueError(
                f"Length of y ({len(y)}) must match length of cartesian product of Xs ({N})"
            )

    def marginal_likelihood(self, name, Xs, y, sigma, is_observed=True, **kwargs):
        """
        Returns the marginal likelihood distribution, given the input
        locations `cartesian(*Xs)` and the data `y`.

        Parameters
        ----------
        name: string
            Name of the random variable
        Xs: list of array-like
            Function input values for each covariance function. Each entry
            must be passable to its respective covariance without error. The
            total covariance function is measured on the full grid
            `cartesian(*Xs)`.
        y: array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        sigma: scalar, Variable
            Standard deviation of the white Gaussian noise.
        is_observed: bool
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        **kwargs
            Extra keyword arguments that are passed to `KroneckerNormal`
            distribution constructor.
        """
        self._check_inputs(Xs, y)
        mu, covs = self._build_marginal_likelihood(Xs)
        self.Xs = Xs
        self.y = y
        self.sigma = sigma
        if is_observed:
            return pm.KroneckerNormal(name, mu=mu, covs=covs, sigma=sigma, observed=y, **kwargs)
        else:
            warnings.warn(
                "The 'is_observed' argument has been deprecated.  If the GP is "
                "unobserved use gp.LatentKron instead.",
                FutureWarning,
            )
            size = int(np.prod([len(X) for X in Xs]))
            return pm.KroneckerNormal(name, mu=mu, covs=covs, sigma=sigma, size=size, **kwargs)

    def _build_conditional(self, Xnew, diag, pred_noise):
        Xs, y, sigma = self.Xs, self.y, self.sigma

        # Old points
        X = cartesian(*Xs)
        delta = y - self.mean_func(X)
        Kns = [f(x) for f, x in zip(self.cov_funcs, Xs)]
        eigs_sep, Qs = zip(*map(eigh, Kns))  # Unzip
        QTs = list(map(at.transpose, Qs))
        eigs = kron_diag(*eigs_sep)  # Combine separate eigs
        if sigma is not None:
            eigs += sigma**2

        Km = self.cov_func(Xnew, diag=diag)
        Knm = self.cov_func(X, Xnew)
        Kmn = Knm.T

        # Build conditional mu
        alpha = kron_dot(QTs, delta)
        alpha = alpha / eigs[:, None]
        alpha = kron_dot(Qs, alpha)
        mu = at.dot(Kmn, alpha).ravel() + self.mean_func(Xnew)

        # Build conditional cov
        A = kron_dot(QTs, Knm)
        A = A / at.sqrt(eigs[:, None])
        if diag:
            Asq = at.sum(at.square(A), 0)
            cov = Km - Asq
            if pred_noise:
                cov += sigma
        else:
            Asq = at.dot(A.T, A)
            cov = Km - Asq
            if pred_noise:
                cov += sigma * at.identity_like(cov)
        return mu, cov

    def conditional(self, name, Xnew, pred_noise=False, diag=False, **kwargs):
        """
        Returns the conditional distribution evaluated over new input
        locations `Xnew`, just as in `Marginal`.

        `Xnew` will be split by columns and fed to the relevant
        covariance functions based on their `input_dim`. For example, if
        `cov_func1`, `cov_func2`, and `cov_func3` have `input_dim` of 2,
        1, and 4, respectively, then `Xnew` must have 7 columns and a
        covariance between the prediction points

        .. code:: python

            cov_func(Xnew) = cov_func1(Xnew[:, :2]) * cov_func1(Xnew[:, 2:3]) * cov_func1(Xnew[:, 3:])

        The distribution returned by `conditional` does not have a
        Kronecker structure regardless of whether the input points lie
        on a full grid.  Therefore, `Xnew` does not need to have grid
        structure.

        Parameters
        ----------
        name: string
            Name of the random variable
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        **kwargs
            Extra keyword arguments that are passed to `MvNormal` distribution
            constructor.
        """
        mu, cov = self._build_conditional(Xnew, diag, pred_noise)
        return pm.MvNormal(name, mu=mu, cov=cov, **kwargs)

    def predict(self, Xnew, point=None, diag=False, pred_noise=False, model=None):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as numpy arrays, given a `point`, such as the MAP
        estimate or a sample from a `trace`.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        point: pymc.model.Point
            A specific point to condition on.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        """
        mu, cov = self._predict_at(Xnew, diag, pred_noise)
        return replace_with_values([mu, cov], replacements=point, model=model)

    def _predict_at(self, Xnew, diag=False, pred_noise=False):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as symbolic variables.

        Parameters
        ----------
        Xnew: array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        diag: bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise: bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        """
        mu, cov = self._build_conditional(Xnew, diag, pred_noise)
        return mu, cov
