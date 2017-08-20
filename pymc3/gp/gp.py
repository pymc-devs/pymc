import numpy as np

import theano
import theano.tensor as tt
import theano.tensor.slinalg

import pymc3 as pm
from pymc3.gp.cov import Covariance
from pymc3.gp.mean import Constant
from pymc3.gp.util import (conditioned_vars,
    infer_shape, stabilize, cholesky, solve, solve_lower, solve_upper)
from pymc3.distributions import draw_values

__all__ = ['Latent', 'Marginal', 'TP', 'MarginalSparse']


class Base(object):
    """
    Base class.  Can be used as a GP placeholder object in
    additive models for GPs that won't be used for prediction.
    """
    def __init__(self, mean_func=None, cov_func=None):
        # check if not None, args are correct subclasses.
        # easy for users to get this wrong
        if mean_func is None:
            mean_func = pm.gp.mean.Zero()

        if cov_func is None:
            cov_func = pm.gp.cov.Constant(0.0)

        self.mean_func = mean_func
        self.cov_func = cov_func

    def __add__(self, other):
        same_attrs = set(self.__dict__.keys()) == set(other.__dict__.keys())
        if not isinstance(self, type(other)) and not same_attrs:
            raise ValueError("cant add different GP types")
        mean_total = self.mean_func + other.mean_func
        cov_total = self.cov_func + other.cov_func
        return self.__class__(mean_total, cov_total)

    def prior(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def marginal_likelihood(self, name, X, *args, **kwargs):
        raise NotImplementedError

    def conditional(self, name, n_points, Xnew, *args, **kwargs):
        raise NotImplementedError

    def predict(self, Xnew, point=None, given=None, diag=False):
        raise NotImplementedError


@conditioned_vars(["X", "f"])
class Latent(Base):
    R"""
    The `gp.Latent` class is a direct implementation of a GP.  No addiive
    noise is assumed.  It is called "Latent" because the underlying function
    values are treated as latent variables.  It has a `prior` method and a
    `conditional` method.  Given a mean and covariance function the
    function $f(x)$ is modeled as,

    .. math::

    f(x) \sim \mathcal{GP}\left(\mu(x), k(x, x')\right)

    Use the `prior` and `conditional` methods to actually construct random
    variables representing the unknown, or latent, function whose
    distribution is the GP prior or GP conditional.  This GP implementation
    can be used to implement regression on data that is not normally
    distributed.

    Parameters
    ----------
    cov_func : None, 2D array, or instance of Covariance
        The covariance function.  Defaults to zero.
    mean_func : None, instance of Mean
        The mean function.  Defaults to zero.

    Examples
    --------
    .. code:: python

    # A one dimensional column vector of inputs.
    X = np.linspace(0, 1, 10)[:, None]

    with pm.Model() as model:
        # Specify the covariance function.
        cov_func = pm.gp.cov.ExpQuad(1, lengthscales=0.1)

        # Specify the GP.  The default mean function is `Zero`.
        gp = pm.gp.Latent(cov_func=cov_func)

        # Place a GP prior over the function f.
        f = gp.prior("f", n_points=10, X=X)

    ...

    # After fitting or sampling, specify the distribution
    # at new points with .conditional
    Xnew = np.linspace(-1, 2, 50)[:, None]

    with model:
        fcond = gp.conditional("fcond", Xnew=Xnew)

    Notes
    -----
    - After initializing the GP object with a mean and covariance
    function, it can be added to other `Latent` GP objects.

    - For more information on the `prior` and `conditional` methods,
    see their docstrings.
    """

    def __init__(self, mean_func=None, cov_func=None):
        super(Latent, self).__init__(mean_func, cov_func)

    def _build_prior(self, name, X, n_points, reparameterize=True):
        mu = self.mean_func(X)
        chol = cholesky(stabilize(self.cov_func(X)))
        if reparameterize:
            v = pm.Normal(name + "_rotated_", mu=0.0, sd=1.0, shape=n_points)
            f = pm.Deterministic(name, mu + tt.dot(chol, v))
        else:
            f = pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)
        return f

    def prior(self, name, X, n_points=None, reparameterize=True):
        R"""
	Returns the GP prior distribution evaluated over the input
        locations `X`.  This is the prior probability over the space
        of functions described by its mean and covariance function.

        .. math::

        f \mid X \sim \text{MvNormal}\left(\boldsymbol\mu, \mathbf{K}\right)

        Parameters
        ----------
        name : string
            Name of the random variable
        X : array-like
            Function input values.
        n_points : int, optional
            Required if `X` is a random variable or a Theano object.
            This is the number of points the GP is evaluated over, the
            number of rows in `X`.
        reparameterize : bool
            Reparameterize the distribution by rotating the random
            variable by the Cholesky factor of the covariance matrix.
        """
        n_points = infer_shape(X, n_points)
        f = self._build_prior(name, X, n_points, reparameterize)
        self.X = X
        self.f = f
        return f

    def _get_given_vals(self, **given):
        if 'gp' in given:
            cov_total = given['gp'].cov_func
            mean_total = given['gp'].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ['X', 'f']):
            X, f = given['X'], given['f']
        else:
            X, f = self.X, self.f
        return X, f, cov_total, mean_total

    def _build_conditional(self, Xnew, X, f, cov_total, mean_total):
        Kxx = cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        v = solve_lower(L, f - mean_total(X))
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        Kss = self.cov_func(Xnew)
        cov = Kss - tt.dot(tt.transpose(A), A)
        return mu, cov

    def conditional(self, name, Xnew, n_points=None, given=None):
        R"""
	Returns the conditional distribution evaluated over new input
        locations `Xnew`.  Given a set of function values `f` that
        the GP prior was over, the conditional distribution over a
        set of new points, `f_*` is

        .. math::

        f^* \mid f, X, X_{\text{new}} \sim \mathcal{GP}\left(\mu(x), k(x, x')\right)

        Parameters
        ----------
        name : string
            Name of the random variable
        Xnew : array-like
            Function input values.
        n_points : int, optional
            Required if `Xnew` is a random variable or a Theano object.
            This is the number of points the GP is evaluated over, the
            number of rows in `Xnew`.
        given : keyword arguments
            Can optionally take as keyword args, `X`, `f`, and `gp`.
            See the tutorial on additive GP models in PyMC3 for more
            information.
        """
        givens = self._get_given_vals(**given)
        mu, cov = self._build_conditional(Xnew, *givens)
        chol = cholesky(stabilize(cov))
        n_points = infer_shape(Xnew, n_points)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)


@conditioned_vars(["X", "f", "nu"])
class TP(Latent):
    """ StudentT process
    https://www.cs.cmu.edu/~andrewgw/tprocess.pdf
    """
    def __init__(self, mean_func=None, cov_func=None, nu=None):
        if nu is None:
            raise ValueError("T Process requires a degrees of freedom parameter, 'nu'")
        self.nu = nu
        super(TP, self).__init__(mean_func, cov_func)

    def __add__(self, other):
        raise ValueError("Student T processes aren't additive")

    def _build_prior(self, name, n_points, X, reparameterize=True):
        mu = self.mean_func(X)
        chol = cholesky(stabilize(self.cov_func(X)))
        if reparameterize:
            chi2 = pm.ChiSquared("chi2_", self.nu)
            v = pm.Normal(name + "_rotated_", mu=0.0, sd=1.0, shape=n_points)
            f = pm.Deterministic(name, (tt.sqrt(self.nu) / chi2) * (mu + tt.dot(chol, v)))
        else:
            f = pm.MvStudentT(name, nu=self.nu, mu=mu, chol=chol, shape=n_points)
        return f

    def prior(self, name, n_points, X, reparameterize=True):
        f = self._build_prior(name, n_points, X, reparameterize)
        self.X = X
        self.f = f
        return f

    def _build_conditional(self, Xnew, X, f):
        Kxx = self.cov_func(X)
        Kxs = self.cov_func(X, Xnew)
        Kss = self.cov_func(Xnew)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        cov = Kss - tt.dot(tt.transpose(A), A)
        v = solve_lower(L, f - self.mean_func(X))
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        beta = tt.dot(v, v)
        nu2 = self.nu + X.shape[0]
        covT = (self.nu + beta - 2)/(nu2 - 2) * cov
        return nu2, mu, covT

    def conditional(self, name, Xnew, n_points=None):
        R"""
	Returns the conditional distribution evaluated over new input
        locations `Xnew`.  Given a set of function values `f` that
        the TP prior was over, the conditional distribution over a
        set of new points, `f_*` is

        .. math::

        f^* \mid f, X, X_{\text{new}} \sim \mathcal{TP}\left(\mu(x), k(x, x'), \nu, \right)

        Parameters
        ----------
        name : string
            Name of the random variable
        Xnew : array-like
            Function input values.
        n_points : int, optional
            Required if `Xnew` is a random variable or a Theano object.
            This is the number of points the GP is evaluated over, the
            number of rows in `Xnew`.
        """
        X = self.X
        f = self.f
        nu2, mu, covT = self._build_conditional(Xnew, X, f)
        chol = cholesky(stabilize(covT))
        n_points = infer_shape(Xnew, n_points)
        return pm.MvStudentT(name, nu=nu2, mu=mu, chol=chol, shape=n_points)


@conditioned_vars(["X", "y", "noise"])
class Marginal(Base):

    def __init__(self, mean_func=None, cov_func=None):
        super(Marginal, self).__init__(mean_func, cov_func)

    def _build_marginal_likelihood(self, X, noise):
        mu = self.mean_func(X)
        Kxx = self.cov_func(X)
        Knx = noise(X)
        cov = Kxx + Knx
        return mu, cov

    def marginal_likelihood(self, name, X, y, noise, n_points=None, is_observed=True):
        R"""
	Returns the marginal likelihood distribution, given the input
        locations `X` and the data `y`.  This is integral over the product of the GP
        prior and a normal likelihood.

        .. math::

        y \mid X,\theta \sim \int p(y \mid f,\, X,\, \theta) \, p(f \mid X,\, \theta) \, df

        Parameters
        ----------
        name : string
            Name of the random variable
        X : array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        y : array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        noise : scalar, Variable, or Covariance
            Standard deviation of the Gaussian noise.  Can also be a Covariance for
            non-white noise.
        n_points : int, optional
            Required if `X` is a random variable or a Theano object.
            This is the number of points the GP is evaluated over, the
            number of rows in `X`.
        is_observed : bool
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        """
        if not isinstance(noise, Covariance):
            noise = pm.gp.cov.WhiteNoise(noise)
        mu, cov = self._build_marginal_likelihood(X, noise)
        chol = cholesky(stabilize(cov))
        self.X = X
        self.y = y
        self.noise = noise
        if is_observed:
            return pm.MvNormal(name, mu=mu, chol=chol, observed=y)
        else:
            n_points = infer_shape(X, n_points)
            return pm.MvNormal(name, mu=mu, chol=chol, size=n_points)

    def _get_given_vals(self, **given):
        if 'gp' in given:
            cov_total = given['gp'].cov_func
            mean_total = given['gp'].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ['X', 'y', 'noise']):
            X, y, noise = given['X'], given['y'], given['noise']
            if not isinstance(noise, Covariance):
                noise = pm.gp.cov.WhiteNoise(noise)
        else:
            X, y, noise = self.X, self.y, self.noise
        return X, y, noise, cov_total, mean_total

    def _build_conditional(self, Xnew, X, y, noise, cov_total, mean_total,
                           pred_noise, diag=False):
        Kxx = cov_total(X)
        Kxs = self.cov_func(X, Xnew)
        Knx = noise(X)
        rxx = y - mean_total(X)
        L = cholesky(stabilize(Kxx) + Knx)
        A = solve_lower(L, Kxs)
        v = solve_lower(L, rxx)
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(A), v)
        if diag:
            Kss = self.cov_func(Xnew, diag=True)
            var = Kss - tt.sum(tt.square(A), 0)
            if pred_noise:
                var += noise(Xnew, diag=True)
            return mu, var
        else:
            Kss = self.cov_func(Xnew)
            cov = Kss - tt.dot(tt.transpose(A), A)
            if pred_noise:
                cov += noise(Xnew)
            return mu, stabilize(cov)

    def conditional(self, name, Xnew, pred_noise=False, n_points=None, **given):
        R"""
	Returns the conditional distribution evaluated over new input
        locations `Xnew`.  Given a set of function values `f` that
        the GP prior was over, the conditional distribution over a
        set of new points, `f_*` is

        .. math::

        f^* \mid f, X, X_{\text{new}} \sim \mathcal{GP}\left(\mu(x), k(x, x')\right)

        Parameters
        ----------
        name : string
            Name of the random variable
        Xnew : array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise : bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        n_points : int, optional
            Required if `Xnew` is a random variable or a Theano object.
            This is the number of points the GP is evaluated over, the
            number of rows in `Xnew`.
        given : keyword arguments
            Can optionally take as keyword args, `X`, `y`, `noise`,
            and `gp`.  See the tutorial on additive GP models in PyMC3
            for more information.
        """
        givens = self._get_given_vals(**given)
        mu, cov = self._build_conditional(Xnew, *givens, pred_noise, diag=False)
        chol = cholesky(cov)
        n_points = infer_shape(Xnew, n_points)
        return pm.MvNormal(name, mu=mu, chol=chol, shape=n_points)

    def predict(self, Xnew, point, diag=False, pred_noise=False, **given):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as numpy arrays, given a `point`, such as the MAP
        estimate or a sample from a `trace`.

        Parameters
        ----------
        Xnew : array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        point : pymc3.model.Point
            A specific point to condition on.
        diag : bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise : bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given : keyword arguments
            Can optionally take the same keyword args as `conditional`.
        """
        mu, cov = self.predictt(Xnew, diag, pred_noise, **given)
        mu, cov = draw_values([mu, cov], point=point)
        return mu, cov

    def predictt(self, Xnew, diag=False, pred_noise=False, **given):
        R"""
        Return the mean vector and covariance matrix of the conditional
        distribution as symbolic variables.

        Parameters
        ----------
        Xnew : array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        diag : bool
            If `True`, return the diagonal instead of the full covariance
            matrix.  Default is `False`.
        pred_noise : bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        given : keyword arguments
            Can optionally take the same keyword args as `conditional`.
        """
        givens = self._get_given_vals(**given)
        mu, cov = self._build_conditional(Xnew, *givens, pred_noise, diag)
        return mu, cov


@conditioned_vars(["X", "Xu", "y", "sigma"])
class MarginalSparse(Marginal):
    _available_approx = ("FITC", "VFE", "DTC")
    """ FITC and VFE sparse approximations
    """
    def __init__(self, mean_func=None, cov_func=None, approx="FITC"):
        if approx not in self._available_approx:
            raise NotImplementedError(approx)
        self.approx = approx
        super(MarginalSparse, self).__init__(mean_func, cov_func)

    def __add__(self, other):
        # new_gp will default to FITC approx
        new_gp = super(MarginalSparse, self).__add__(other)
        # make sure new gp has correct approx
        if not self.approx == other.approx:
            raise ValueError("Cant add GPs with different approximations")
        new_gp.approx = self.approx
        return new_gp

    def _build_marginal_likelihood_logp(self, X, Xu, y, sigma):
        sigma2 = tt.square(sigma)
        Kuu = self.cov_func(Xu)
        Kuf = self.cov_func(Xu, X)
        Luu = cholesky(stabilize(Kuu))
        A = solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if self.approx == "FITC":
            Kffd = self.cov_func(X, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
            trace = 0.0
        elif self.approx == "VFE":
            Lamd = tt.ones_like(Qffd) * sigma2
            trace = ((1.0 / (2.0 * sigma2)) *
                     (tt.sum(self.cov_func(X, diag=True)) -
                      tt.sum(tt.sum(A * A, 0))))
        else: # DTC
            Lamd = tt.ones_like(Qffd) * sigma2
            trace = 0.0
        A_l = A / Lamd
        L_B = cholesky(tt.eye(Xu.shape[0]) + tt.dot(A_l, tt.transpose(A)))
        r = y - self.mean_func(X)
        r_l = r / Lamd
        c = solve_lower(L_B, tt.dot(A, r_l))
        constant = 0.5 * X.shape[0] * tt.log(2.0 * np.pi)
        logdet = 0.5 * tt.sum(tt.log(Lamd)) + tt.sum(tt.log(tt.diag(L_B)))
        quadratic = 0.5 * (tt.dot(r, r_l) - tt.dot(c, c))
        return -1.0 * (constant + logdet + quadratic + trace)

    def marginal_likelihood(self, name, X, Xu, y, sigma, n_points=None, is_observed=True):
        R"""
	Returns the approximate marginal likelihood distribution, given the input
        locations `X` and the data `y`.  This is integral over the product of the GP
        prior and a normal likelihood.

        .. math::

        y \mid X,\theta \sim \int p(y \mid f,\, X,\, \theta) \, p(f \mid X,\, \theta) \, df

        Parameters
        ----------
        name : string
            Name of the random variable
        X : array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        Xu: array-like
            The inducing points.  Must have the same number of columns as `X`.
        y : array-like
            Data that is the sum of the function with the GP prior and Gaussian
            noise.  Must have shape `(n, )`.
        sigma : scalar, Variable
            Standard deviation of the Gaussian noise.
        n_points : int, optional
            Required if `X` is a random variable or a Theano object.
            This is the number of points the GP is evaluated over, the
            number of rows in `X`.
        is_observed : bool
            Whether to set `y` as an `observed` variable in the `model`.
            Default is `True`.
        """
        self.X = X
        self.Xu = Xu
        self.y = y
        self.sigma = sigma
        logp = lambda y: self._build_marginal_likelihood_logp(X, Xu, y, sigma)
        if is_observed:  # same thing ith n_points here?? check
            return pm.DensityDist(name, logp, observed=y)
        else:
            n_points = infer_shape(X, n_points)
            return pm.DensityDist(name, logp, size=n_points)

    def _build_conditional(self, Xnew, Xu, X, y, sigma, cov_total, mean_total,
                           pred_noise, diag=False):
        sigma2 = tt.square(sigma)
        Kuu = cov_total(Xu)
        Kuf = cov_total(Xu, X)
        Luu = cholesky(stabilize(Kuu))
        A = solve_lower(Luu, Kuf)
        Qffd = tt.sum(A * A, 0)
        if self.approx == "FITC":
            Kffd = cov_total(X, diag=True)
            Lamd = tt.clip(Kffd - Qffd, 0.0, np.inf) + sigma2
        else: # VFE or DTC
            Lamd = tt.ones_like(Qffd) * sigma2
        A_l = A / Lamd
        L_B = cholesky(tt.eye(Xu.shape[0]) + tt.dot(A_l, tt.transpose(A)))
        r = y - mean_total(X)
        r_l = r / Lamd
        c = solve_lower(L_B, tt.dot(A, r_l))
        Kus = self.cov_func(Xu, Xnew)
        As = solve_lower(Luu, Kus)
        mu = self.mean_func(Xnew) + tt.dot(tt.transpose(As), solve_upper(tt.transpose(L_B), c))
        C = solve_lower(L_B, As)
        if diag:
            Kss = self.cov_func(Xnew, diag=True)
            var = Kss - tt.sum(tt.sqaure(As), 0) + tt.sum(tt.square(C), 0)
            if pred_noise:
                var += sigma2
            return mu, var
        else:
            cov = (self.cov_func(Xnew) - tt.dot(tt.transpose(As), As) +
                   tt.dot(tt.transpose(C), C))
            if pred_noise:
                cov += sigma2 * tt.identity_like(cov)
            return mu, stabilize(cov)

    def _get_given_vals(self, **given):
        if 'gp' in given:
            cov_total = given['gp'].cov_func
            mean_total = given['gp'].mean_func
        else:
            cov_total = self.cov_func
            mean_total = self.mean_func
        if all(val in given for val in ['X', 'Xu', 'y', 'sigma']):
            X, Xu, y, sigma = given['X'], given['Xu'], given['y'], given['sigma']
        else:
            X, Xu, y, sigma = self.X, self.y, self.sigma
        return X, y, sigma, cov_total, mean_total

    def conditional(self, name, Xnew, pred_noise=False, n_points=None, **given):
        R"""
	Returns the conditional distribution evaluated over new input
        locations `Xnew`.  Given a set of function values `f` that
        the GP prior was over, the conditional distribution over a
        set of new points, `f_*` is

        .. math::

        f^* \mid f, X, X_{\text{new}} \sim \mathcal{GP}\left(\mu(x), k(x, x')\right)

        Parameters
        ----------
        name : string
            Name of the random variable
        Xnew : array-like
            Function input values.  If one-dimensional, must be a column
            vector with shape `(n, 1)`.
        pred_noise : bool
            Whether or not observation noise is included in the conditional.
            Default is `False`.
        n_points : int, optional
            Required if `Xnew` is a random variable or a Theano object.
            This is the number of points the GP is evaluated over, the
            number of rows in `Xnew`.
        given : keyword arguments
            Can optionally take as keyword args, `X`, `Xu`, `y`, `sigma`,
            and `gp`.  See the tutorial on additive GP models in PyMC3
            for more information.
        """
        rv = super(MarginalSparse, self).conditional(name, Xnew, pred_noise, n_points, **given)
        return rv

