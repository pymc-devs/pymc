import theano.tensor as tt
from theano import scan

from .continuous import Normal, Flat
from .distribution import Continuous

__all__ = ['AR1', 'GaussianRandomWalk', 'GARCH11', 'EulerMaruyama']


class AR1(Continuous):
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
        self.k = k
        self.tau_e = tau_e
        self.tau = tau_e * (1 - k ** 2)
        self.mode = 0.

    def logp(self, x):
        k = self.k
        tau_e = self.tau_e

        x_im1 = x[:-1]
        x_i = x[1:]
        boundary = Normal.dist(0, tau_e).logp

        innov_like = Normal.dist(k * x_im1, tau_e).logp(x_i)
        return boundary(x[0]) + tt.sum(innov_like) + boundary(x[-1])


class GaussianRandomWalk(Continuous):
    """
    Random Walk with Normal innovations

    Parameters
    ----------
    tau : tensor
        tau > 0, innovation precision
    sd : tensor
        sd > 0, innovation standard deviation (alternative to specifying tau)
    mu: tensor
        innovation drift, defaults to 0.0
    init : distribution
        distribution for initial value (Defaults to Flat())
    """

    def __init__(self, tau=None, init=Flat.dist(), sd=None, mu=0.,
                 *args, **kwargs):
        super(GaussianRandomWalk, self).__init__(*args, **kwargs)
        self.tau = tau
        self.sd = sd
        self.mu = mu
        self.init = init
        self.mean = 0.

    def logp(self, x):
        tau = self.tau
        sd = self.sd
        mu = self.mu
        init = self.init

        x_im1 = x[:-1]
        x_i = x[1:]

        innov_like = Normal.dist(mu=x_im1 + mu, tau=tau, sd=sd).logp(x_i)
        return init.logp(x[0]) + tt.sum(innov_like)


class GARCH11(Continuous):
    """
    GARCH(1,1) with Normal innovations. The model is specified by

    y_t = sigma_t * z_t
    sigma_t^2 = omega + alpha_1 * y_{t-1}^2 + beta_1 * sigma_{t-1}^2

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

    def __init__(self, omega=None, alpha_1=None, beta_1=None,
                 initial_vol=None, *args, **kwargs):
        super(GARCH11, self).__init__(*args, **kwargs)

        self.omega = omega
        self.alpha_1 = alpha_1
        self.beta_1 = beta_1
        self.initial_vol = initial_vol
        self.mean = 0

    def _get_volatility(self, x):

        def volatility_update(x, vol, w, a, b):
            return tt.sqrt(w + a * tt.square(x) + b * tt.square(vol))

        vol, _ = scan(fn=volatility_update,
                      sequences=[x],
                      outputs_info=[self.initial_vol],
                      non_sequences=[self.omega, self.alpha_1,
                                     self.beta_1])
        return vol

    def logp(self, x):
        vol = self._get_volatility(x[:-1])
        return (Normal.dist(0., sd=self.initial_vol).logp(x[0]) +
                tt.sum(Normal.dist(0, sd=vol).logp(x[1:])))


class EulerMaruyama(Continuous):
    """
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
        self.dt = dt
        self.sde_fn = sde_fn
        self.sde_pars = sde_pars

    def logp(self, x):
        xt = x[:-1]
        f, g = self.sde_fn(x[:-1], *self.sde_pars)
        mu = xt + self.dt * f
        sd = tt.sqrt(self.dt) * g
        return tt.sum(Normal.dist(mu=mu, sd=sd).logp(x[1:]))
