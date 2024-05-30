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

import numpy as np
import theano.tensor as tt

from scipy import stats
from theano import scan

from pymc3.distributions import distribution, multivariate
from pymc3.distributions.continuous import Flat, Normal, get_tau_sigma
from pymc3.distributions.shape_utils import to_tuple

__all__ = [
    "AR1",
    "AR",
    "GaussianRandomWalk",
    "GARCH11",
    "EulerMaruyama",
    "MvGaussianRandomWalk",
    "MvStudentTRandomWalk",
]


class AR1(distribution.Continuous):
    """
    Autoregressive process with 1 lag.

    Parameters
    ----------
    k: tensor
       effect of lagged value on current value
    tau_e: tensor
       precision for innovations
    """

    def __init__(self, k, tau_e, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k = tt.as_tensor_variable(k)
        self.tau_e = tau_e = tt.as_tensor_variable(tau_e)
        self.tau = tau_e * (1 - k ** 2)
        self.mode = tt.as_tensor_variable(0.0)

    def logp(self, x):
        """
        Calculate log-probability of AR1 distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        k = self.k
        tau_e = self.tau_e  # innovation precision
        tau = tau_e * (1 - k ** 2)  # ar1 precision

        x_im1 = x[:-1]
        x_i = x[1:]
        boundary = Normal.dist(0.0, tau=tau).logp

        innov_like = Normal.dist(k * x_im1, tau=tau_e).logp(x_i)
        return boundary(x[0]) + tt.sum(innov_like)


class AR(distribution.Continuous):
    r"""
    Autoregressive process with p lags.

    .. math::

       x_t = \rho_0 + \rho_1 x_{t-1} + \ldots + \rho_p x_{t-p} + \epsilon_t,
       \epsilon_t \sim N(0,\sigma^2)

    The innovation can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}

    Parameters
    ----------
    rho: tensor
        Tensor of autoregressive coefficients. The first dimension is the p lag.
    sigma: float
        Standard deviation of innovation (sigma > 0). (only required if tau is not specified)
    tau: float
        Precision of innovation (tau > 0). (only required if sigma is not specified)
    constant: bool (optional, default = False)
        Whether to include a constant.
    init: distribution
        distribution for initial values (Defaults to Flat())
    """

    def __init__(
        self, rho, sigma=None, tau=None, constant=False, init=Flat.dist(), sd=None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd

        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.sigma = self.sd = tt.as_tensor_variable(sigma)
        self.tau = tt.as_tensor_variable(tau)

        self.mean = tt.as_tensor_variable(0.0)

        if isinstance(rho, list):
            p = len(rho)
        else:
            try:
                shape_ = rho.shape.tag.test_value
            except AttributeError:
                shape_ = rho.shape

            if hasattr(shape_, "size") and shape_.size == 0:
                p = 1
            else:
                p = shape_[0]

        if constant:
            self.p = p - 1
        else:
            self.p = p

        self.constant = constant
        self.rho = rho = tt.as_tensor_variable(rho)
        self.init = init

    def logp(self, value):
        """
        Calculate log-probability of AR distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        if self.constant:
            x = tt.add(
                *[self.rho[i + 1] * value[self.p - (i + 1) : -(i + 1)] for i in range(self.p)]
            )
            eps = value[self.p :] - self.rho[0] - x
        else:
            if self.p == 1:
                x = self.rho * value[:-1]
            else:
                x = tt.add(
                    *[self.rho[i] * value[self.p - (i + 1) : -(i + 1)] for i in range(self.p)]
                )
            eps = value[self.p :] - x

        innov_like = Normal.dist(mu=0.0, tau=self.tau).logp(eps)
        init_like = self.init.logp(value[: self.p])

        return tt.sum(innov_like) + tt.sum(init_like)


class GaussianRandomWalk(distribution.Continuous):
    r"""Random Walk with Normal innovations

    Note that this is mainly a user-friendly wrapper to enable an easier specification
    of GRW. You are not restricted to use only Normal innovations but can use any
    distribution: just use `theano.tensor.cumsum()` to create the random walk behavior.

    Parameters
    ----------
    mu: tensor
        innovation drift, defaults to 0.0
        For vector valued mu, first dimension must match shape of the random walk, and
        the first element will be discarded (since there is no innovation in the first timestep)
    sigma: tensor
        sigma > 0, innovation standard deviation (only required if tau is not specified)
        For vector valued sigma, first dimension must match shape of the random walk, and
        the first element will be discarded (since there is no innovation in the first timestep)
    tau: tensor
        tau > 0, innovation precision (only required if sigma is not specified)
        For vector valued tau, first dimension must match shape of the random walk, and
        the first element will be discarded (since there is no innovation in the first timestep)
    init: distribution
        distribution for initial value (Defaults to Flat())
    """

    def __init__(self, tau=None, init=Flat.dist(), sigma=None, mu=0.0, sd=None, *args, **kwargs):
        kwargs.setdefault("shape", 1)
        super().__init__(*args, **kwargs)
        if sum(self.shape) == 0:
            raise TypeError("GaussianRandomWalk must be supplied a non-zero shape argument!")
        if sd is not None:
            sigma = sd
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.tau = tt.as_tensor_variable(tau)
        sigma = tt.as_tensor_variable(sigma)
        self.sigma = self.sd = sigma
        self.mu = tt.as_tensor_variable(mu)
        self.init = init
        self.mean = tt.as_tensor_variable(0.0)

    def _mu_and_sigma(self, mu, sigma):
        """Helper to get mu and sigma if they are high dimensional."""
        if sigma.ndim > 0:
            sigma = sigma[1:]
        if mu.ndim > 0:
            mu = mu[1:]
        return mu, sigma

    def logp(self, x):
        """
        Calculate log-probability of Gaussian Random Walk distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        if x.ndim > 0:
            x_im1 = x[:-1]
            x_i = x[1:]
            mu, sigma = self._mu_and_sigma(self.mu, self.sigma)
            innov_like = Normal.dist(mu=x_im1 + mu, sigma=sigma).logp(x_i)
            return self.init.logp(x[0]) + tt.sum(innov_like)
        return self.init.logp(x)

    def random(self, point=None, size=None):
        """Draw random values from GaussianRandomWalk.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        sigma, mu = distribution.draw_values([self.sigma, self.mu], point=point, size=size)
        return distribution.generate_samples(
            self._random,
            sigma=sigma,
            mu=mu,
            size=size,
            dist_shape=self.shape,
            not_broadcast_kwargs={"sample_shape": to_tuple(size)},
        )

    def _random(self, sigma, mu, size, sample_shape):
        """Implement a Gaussian random walk as a cumulative sum of normals.
        axis = len(size) - 1 denotes the axis along which cumulative sum would be calculated.
        This might need to be corrected in future when issue #4010 is fixed.
        """
        if size[len(sample_shape)] == sample_shape:
            axis = len(sample_shape)
        else:
            axis = len(size) - 1
        rv = stats.norm(mu, sigma)
        data = rv.rvs(size).cumsum(axis=axis)
        data = np.array(data)

        # the following lines center the random walk to start at the origin.
        if len(data.shape) > 1:
            for i in range(data.shape[0]):
                data[i] = data[i] - data[i][0]
        else:
            data = data - data[0]
        return data

    def _distr_parameters_for_repr(self):
        return ["mu", "sigma"]


class GARCH11(distribution.Continuous):
    r"""
    GARCH(1,1) with Normal innovations. The model is specified by

    .. math::
        y_t = \sigma_t * z_t

    .. math::
        \sigma_t^2 = \omega + \alpha_1 * y_{t-1}^2 + \beta_1 * \sigma_{t-1}^2

    with z_t iid and Normal with mean zero and unit standard deviation.

    Parameters
    ----------
    omega: tensor
        omega > 0, mean variance
    alpha_1: tensor
        alpha_1 >= 0, autoregressive term coefficient
    beta_1: tensor
        beta_1 >= 0, alpha_1 + beta_1 < 1, moving average term coefficient
    initial_vol: tensor
        initial_vol >= 0, initial volatility, sigma_0
    """

    def __init__(self, omega, alpha_1, beta_1, initial_vol, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.omega = omega = tt.as_tensor_variable(omega)
        self.alpha_1 = alpha_1 = tt.as_tensor_variable(alpha_1)
        self.beta_1 = beta_1 = tt.as_tensor_variable(beta_1)
        self.initial_vol = tt.as_tensor_variable(initial_vol)
        self.mean = tt.as_tensor_variable(0.0)

    def get_volatility(self, x):
        x = x[:-1]

        def volatility_update(x, vol, w, a, b):
            return tt.sqrt(w + a * tt.square(x) + b * tt.square(vol))

        vol, _ = scan(
            fn=volatility_update,
            sequences=[x],
            outputs_info=[self.initial_vol],
            non_sequences=[self.omega, self.alpha_1, self.beta_1],
        )
        return tt.concatenate([[self.initial_vol], vol])

    def logp(self, x):
        """
        Calculate log-probability of GARCH(1, 1) distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        vol = self.get_volatility(x)
        return tt.sum(Normal.dist(0.0, sigma=vol).logp(x))

    def _distr_parameters_for_repr(self):
        return ["omega", "alpha_1", "beta_1"]


class EulerMaruyama(distribution.Continuous):
    r"""
    Stochastic differential equation discretized with the Euler-Maruyama method.

    Parameters
    ----------
    dt: float
        time step of discretization
    sde_fn: callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars: tuple
        parameters of the SDE, passed as ``*args`` to ``sde_fn``
    """

    def __init__(self, dt, sde_fn, sde_pars, *args, **kwds):
        super().__init__(*args, **kwds)
        self.dt = dt = tt.as_tensor_variable(dt)
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        """
        Calculate log-probability of EulerMaruyama distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        xt = x[:-1]
        f, g = self.sde_fn(x[:-1], *self.sde_pars)
        mu = xt + self.dt * f
        sd = tt.sqrt(self.dt) * g
        return tt.sum(Normal.dist(mu=mu, sigma=sd).logp(x[1:]))

    def _distr_parameters_for_repr(self):
        return ["dt"]


class MvGaussianRandomWalk(distribution.Continuous):
    r"""
    Multivariate Random Walk with Normal innovations

    Parameters
    ----------
    mu: tensor
        innovation drift, defaults to 0.0
    cov: tensor
        pos def matrix, innovation covariance matrix
    tau: tensor
        pos def matrix, inverse covariance matrix
    chol: tensor
        Cholesky decomposition of covariance matrix
    init: distribution
        distribution for initial value (Defaults to Flat())

    Notes
    -----
    Only one of cov, tau or chol is required.

    """

    def __init__(
        self, mu=0.0, cov=None, tau=None, chol=None, lower=True, init=Flat.dist(), *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.init = init
        self.innovArgs = (mu, cov, tau, chol, lower)
        self.innov = multivariate.MvNormal.dist(*self.innovArgs, shape=self.shape)
        self.mean = tt.as_tensor_variable(0.0)

    def logp(self, x):
        """
        Calculate log-probability of Multivariate Gaussian
        Random Walk distribution at specified value.

        Parameters
        ----------
        x: numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """

        if x.ndim == 1:
            x = x[np.newaxis, :]

        x_im1 = x[:-1]
        x_i = x[1:]

        return self.init.logp_sum(x[0]) + self.innov.logp_sum(x_i - x_im1)

    def _distr_parameters_for_repr(self):
        return ["mu", "cov"]

    def random(self, point=None, size=None):
        """
        Draw random values from MvGaussianRandomWalk.

        Parameters
        ----------
        point: dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size: int or tuple of ints, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array


        Examples
        --------
        .. code-block:: python

            with pm.Model():
                mu = np.array([1.0, 0.0])
                cov = np.array([[1.0, 0.0],
                                [0.0, 2.0]])

                # draw one sample from a 2-dimensional Gaussian random walk with 10 timesteps
                sample = MvGaussianRandomWalk(mu, cov, shape=(10, 2)).random()

                # draw three samples from a 2-dimensional Gaussian random walk with 10 timesteps
                sample = MvGaussianRandomWalk(mu, cov, shape=(10, 2)).random(size=3)

                # draw four samples from a 2-dimensional Gaussian random walk with 10 timesteps,
                # indexed with a (2, 2) array
                sample = MvGaussianRandomWalk(mu, cov, shape=(10, 2)).random(size=(2, 2))
        """

        # for each draw specified by the size input, we need to draw time_steps many
        # samples from MvNormal.

        size = to_tuple(size)
        multivariate_samples = self.innov.random(point=point, size=size)
        # this has shape (size, self.shape)
        if len(self.shape) == 2:
            # have time dimension in first slot of shape. Therefore the time
            # component can be accessed with the index equal to the length of size.
            time_axis = len(size)
            multivariate_samples = multivariate_samples.cumsum(axis=time_axis)
            if time_axis != 0:
                # this for loop covers the case where size is a tuple
                for idx in np.ndindex(size):
                    multivariate_samples[idx] = (
                        multivariate_samples[idx] - multivariate_samples[idx][0]
                    )
            else:
                # size was passed as None
                multivariate_samples = multivariate_samples - multivariate_samples[0]

        # if the above statement fails, then only a spatial dimension was passed in for self.shape.
        # Therefore don't subtract off the initial value since otherwise you get all zeros
        # as your output.
        return multivariate_samples


class MvStudentTRandomWalk(MvGaussianRandomWalk):
    r"""
    Multivariate Random Walk with StudentT innovations

    Parameters
    ----------
    nu: degrees of freedom
    mu: tensor
        innovation drift, defaults to 0.0
    cov: tensor
        pos def matrix, innovation covariance matrix
    tau: tensor
        pos def matrix, inverse covariance matrix
    chol: tensor
        Cholesky decomposition of covariance matrix
    init: distribution
        distribution for initial value (Defaults to Flat())
    """

    def __init__(self, nu, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nu = tt.as_tensor_variable(nu)
        self.innov = multivariate.MvStudentT.dist(self.nu, None, *self.innovArgs)

    def _distr_parameters_for_repr(self):
        return ["nu", "mu", "cov"]
