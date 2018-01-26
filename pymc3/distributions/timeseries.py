import theano.tensor as tt
from theano import scan

from pymc3.util import get_variable_name
from .continuous import get_tau_sd, Normal, Flat
from .dist_math import Cholesky
from . import multivariate
from . import distribution


__all__ = [
    'AR1',
    'AR',
    'GaussianRandomWalk',
    'GARCH11',
    'EulerMaruyama',
    'MvGaussianRandomWalk',
    'MvStudentTRandomWalk'
]


class AR1(distribution.Continuous):
    """
    Autoregressive process with 1 lag.

    Parameters
    ----------
    k : tensor
       effect of lagged value on current value
    tau_e : tensor
       precision for innovations
    """

    def __init__(self, k, tau_e, *args, **kwargs):
        super(AR1, self).__init__(*args, **kwargs)
        self.k = k = tt.as_tensor_variable(k)
        self.tau_e = tau_e = tt.as_tensor_variable(tau_e)
        self.tau = tau_e * (1 - k ** 2)
        self.mode = tt.as_tensor_variable(0.)

    def logp(self, x):
        k = self.k
        tau_e = self.tau_e

        x_im1 = x[:-1]
        x_i = x[1:]
        boundary = Normal.dist(0., tau=tau_e).logp

        innov_like = Normal.dist(k * x_im1, tau=tau_e).logp(x_i)
        return boundary(x[0]) + tt.sum(innov_like)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        k = dist.k
        tau_e = dist.tau_e
        name = r'\text{%s}' % name
        return r'${} \sim \text{{AR1}}(\mathit{{k}}={},~\mathit{{tau_e}}={})$'.format(name,
                 get_variable_name(k), get_variable_name(tau_e))


class AR(distribution.Continuous):
    R"""
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
    rho : tensor
        Vector of autoregressive coefficients.
    sd : float
        Standard deviation of innovation (sd > 0). (only required if tau is not specified)
    tau : float
        Precision of innovation (tau > 0). (only required if sd is not specified)
    constant: bool (optional, default = False)
        Whether to include a constant.
    init : distribution
        distribution for initial values (Defaults to Flat())
    """

    def __init__(self, rho, sd=None, tau=None,
                 constant=False, init=Flat.dist(),
                 *args, **kwargs):

        super(AR, self).__init__(*args, **kwargs)
        tau, sd = get_tau_sd(tau=tau, sd=sd)
        self.sd = tt.as_tensor_variable(sd)
        self.tau = tt.as_tensor_variable(tau)

        self.mean = tt.as_tensor_variable(0.)

        rho = tt.as_tensor_variable(rho, ndim=1)
        if constant:
            self.p = rho.shape[0] - 1
        else:
            self.p = rho.shape[0]

        self.constant = constant
        self.rho = rho
        self.init = init

    def logp(self, value):

        y = value[self.p:]
        results, _ = scan(lambda l, obs, p: obs[p - l:-l],
                          outputs_info=None, sequences=[tt.arange(1, self.p + 1)],
                          non_sequences=[value, self.p])
        x = tt.stack(results)

        if self.constant:
            y = y - self.rho[0]
            eps = y - self.rho[1:].dot(x)
        else:
            eps = y - self.rho.dot(x)

        innov_like = Normal.dist(mu=0.0, tau=self.tau).logp(eps)
        init_like = self.init.logp(value[:self.p])

        return tt.sum(innov_like) + tt.sum(init_like)


class GaussianRandomWalk(distribution.Continuous):
    R"""
    Random Walk with Normal innovations

    Parameters
    ----------
    mu: tensor
        innovation drift, defaults to 0.0
    sd : tensor
        sd > 0, innovation standard deviation (only required if tau is not specified)
    tau : tensor
        tau > 0, innovation precision (only required if sd is not specified)
    init : distribution
        distribution for initial value (Defaults to Flat())
    """

    def __init__(self, tau=None, init=Flat.dist(), sd=None, mu=0.,
                 *args, **kwargs):
        super(GaussianRandomWalk, self).__init__(*args, **kwargs)
        tau, sd = get_tau_sd(tau=tau, sd=sd)
        self.tau = tau = tt.as_tensor_variable(tau)
        self.sd = sd = tt.as_tensor_variable(sd)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.init = init
        self.mean = tt.as_tensor_variable(0.)

    def logp(self, x):
        tau = self.tau
        sd = self.sd
        mu = self.mu
        init = self.init

        x_im1 = x[:-1]
        x_i = x[1:]

        innov_like = Normal.dist(mu=x_im1 + mu, sd=sd).logp(x_i)
        return init.logp(x[0]) + tt.sum(innov_like)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        sd = dist.sd
        name = r'\text{%s}' % name
        return r'${} \sim \text{{GaussianRandomWalk}}(\mathit{{mu}}={},~\mathit{{sd}}={})$'.format(name,
                                                get_variable_name(mu),
                                                get_variable_name(sd))


class GARCH11(distribution.Continuous):
    R"""
    GARCH(1,1) with Normal innovations. The model is specified by

    .. math::
        y_t = \sigma_t * z_t
        \sigma_t^2 = \omega + \alpha_1 * y_{t-1}^2 + \beta_1 * \sigma_{t-1}^2

    with z_t iid and Normal with mean zero and unit standard deviation.

    Parameters
    ----------
    omega : distribution
        omega > 0, distribution for mean variance
    alpha_1 : distribution
        alpha_1 >= 0, distribution for autoregressive term
    beta_1 : distribution
        beta_1 >= 0, alpha_1 + beta_1 < 1, distribution for moving
        average term
    initial_vol : distribution
        initial_vol >= 0, distribution for initial volatility, sigma_0
    """

    def __init__(self, omega, alpha_1, beta_1,
                 initial_vol, *args, **kwargs):
        super(GARCH11, self).__init__(*args, **kwargs)

        self.omega = omega = tt.as_tensor_variable(omega)
        self.alpha_1 = alpha_1 = tt.as_tensor_variable(alpha_1)
        self.beta_1 = beta_1 = tt.as_tensor_variable(beta_1)
        self.initial_vol = initial_vol
        self.mean = tt.as_tensor_variable(0.)

    def get_volatility(self, x):
        x = x[:-1]

        def volatility_update(x, vol, w, a, b):
            return tt.sqrt(w + a * tt.square(x) + b * tt.square(vol))

        vol, _ = scan(fn=volatility_update,
                      sequences=[x],
                      outputs_info=[self.initial_vol],
                      non_sequences=[self.omega, self.alpha_1,
                                     self.beta_1])
        return tt.concatenate(self.initial_vol, vol)

    def logp(self, x):
        vol = self.get_volatility(x)
        return tt.sum(Normal.dist(0., sd=vol).logp(x))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        omega = dist.omega
        alpha_1 = dist.alpha_1
        beta_1 = dist.beta_1
        name = r'\text{%s}' % name
        return r'${} \sim \text{GARCH}(1,~1,~\mathit{{omega}}={},~\mathit{{alpha_1}}={},~\mathit{{beta_1}}={})$'.format(
            name,
            get_variable_name(omega),
            get_variable_name(alpha_1),
            get_variable_name(beta_1))


class EulerMaruyama(distribution.Continuous):
    R"""
    Stochastic differential equation discretized with the Euler-Maruyama method.

    Parameters
    ----------
    dt : float
        time step of discretization
    sde_fn : callable
        function returning the drift and diffusion coefficients of SDE
    sde_pars : tuple
        parameters of the SDE, passed as *args to sde_fn
    """
    def __init__(self, dt, sde_fn, sde_pars, *args, **kwds):
        super(EulerMaruyama, self).__init__(*args, **kwds)
        self.dt = dt = tt.as_tensor_variable(dt)
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        xt = x[:-1]
        f, g = self.sde_fn(x[:-1], *self.sde_pars)
        mu = xt + self.dt * f
        sd = tt.sqrt(self.dt) * g
        return tt.sum(Normal.dist(mu=mu, sd=sd).logp(x[1:]))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        dt = dist.dt
        name = r'\text{%s}' % name
        return r'${} \sim \text{EulerMaruyama}(\mathit{{dt}}={})$'.format(name,
                                                get_variable_name(dt))


class _CovSet():
    R"""
    Convenience class to set Covariance, Inverse Covariance and Cholesky
    descomposition of Covariance marrices.
    """
    def __initCov__(self, cov=None, tau=None, chol=None, lower=True):
        if all([val is None for val in [cov, tau, chol]]):
            raise ValueError('One of cov, tau or chol arguments must be provided.')

        self.cov = self.tau = self.chol_cov = None

        cholesky = Cholesky(nofail=True, lower=True)
        if cov is not None:
            self.k = cov.shape[0]
            self._cov_type = 'cov'
            cov = tt.as_tensor_variable(cov)
            if cov.ndim != 2:
                raise ValueError('cov must be two dimensional.')
            self.chol_cov = cholesky(cov)
            self.cov = cov
            self._n = self.cov.shape[-1]
        elif tau is not None:
            self.k = tau.shape[0]
            self._cov_type = 'tau'
            tau = tt.as_tensor_variable(tau)
            if tau.ndim != 2:
                raise ValueError('tau must be two dimensional.')
            self.chol_tau = cholesky(tau)
            self.tau = tau
            self._n = self.tau.shape[-1]
        else:
            if chol is not None and not lower:
                chol = chol.T
            self.k = chol.shape[0]
            self._cov_type = 'chol'
            if chol.ndim != 2:
                raise ValueError('chol must be two dimensional.')
            self.chol_cov = tt.as_tensor_variable(chol)
            self._n = self.chol_cov.shape[-1]


class MvGaussianRandomWalk(distribution.Continuous, _CovSet):
    R"""
    Multivariate Random Walk with Normal innovations

    Parameters
    ----------
    mu : tensor
        innovation drift, defaults to 0.0
    cov : tensor
        pos def matrix, innovation covariance matrix
    tau : tensor
        pos def matrix, inverse covariance matrix
    chol : tensor
        Cholesky decomposition of covariance matrix
    init : distribution
        distribution for initial value (Defaults to Flat())

    Notes
    -----
    Only one of cov, tau or chol is required.

    """
    def __init__(self, mu=0., cov=None, tau=None, chol=None, lower=True, init=Flat.dist(),
                 *args, **kwargs):
        super(MvGaussianRandomWalk, self).__init__(*args, **kwargs)
        super(MvGaussianRandomWalk, self).__initCov__(cov, tau, chol, lower)

        self.mu = mu = tt.as_tensor_variable(mu)
        self.init = init
        self.mean = tt.as_tensor_variable(0.)

    def logp(self, x):
        x_im1 = x[:-1]
        x_i = x[1:]

        innov_like = multivariate.MvNormal.dist(mu=x_im1 + self.mu, cov=self.cov,
                                                tau=self.tau, chol=self.chol_cov).logp(x_i)
        return self.init.logp(x[0]) + tt.sum(innov_like)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        cov = dist.cov
        name = r'\text{%s}' % name
        return r'${} \sim \text{MvGaussianRandomWalk}(\mathit{{mu}}={},~\mathit{{cov}}={})$'.format(name,
                                                get_variable_name(mu),
                                                get_variable_name(cov))


class MvStudentTRandomWalk(distribution.Continuous, _CovSet):
    R"""
    Multivariate Random Walk with StudentT innovations

    Parameters
    ----------
    nu : degrees of freedom
    mu : tensor
        innovation drift, defaults to 0.0
    cov : tensor
        pos def matrix, innovation covariance matrix
    tau : tensor
        pos def matrix, inverse covariance matrix
    chol : tensor
        Cholesky decomposition of covariance matrix
    init : distribution
        distribution for initial value (Defaults to Flat())
    """
    def __init__(self, nu, mu=0., cov=None, tau=None, chol=None, lower=True, init=Flat.dist(),
                 *args, **kwargs):
        super(MvStudentTRandomWalk, self).__init__(*args, **kwargs)
        super(MvStudentTRandomWalk, self).__initCov__(cov, tau, chol, lower)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.nu = nu = tt.as_tensor_variable(nu)
        self.init = init
        self.mean = tt.as_tensor_variable(0.)

    def logp(self, x):
        x_im1 = x[:-1]
        x_i = x[1:]
        innov_like = multivariate.MvStudentT.dist(self.nu, mu=x_im1 + self.mu,
                                                  cov=self.cov, tau=self.tau,
                                                  chol=self.chol_cov).logp(x_i)
        return self.init.logp(x[0]) + tt.sum(innov_like)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        nu = dist.nu
        mu = dist.mu
        cov = dist.cov
        name = r'\text{%s}' % name
        return r'${} \sim \text{MvStudentTRandomWalk}(\mathit{{nu}}={},~\mathit{{mu}}={},~\mathit{{cov}}={})$'.format(name,
                                                get_variable_name(nu),
                                                get_variable_name(mu),
                                                get_variable_name(cov))
