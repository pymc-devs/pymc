"""
pymc3.distributions

A collection of common probability distributions for stochastic
nodes in PyMC.

"""
from __future__ import division

import numpy as np
import theano.tensor as T
from scipy import stats

from . import transforms
from .dist_math import bound, logpow, gammaln, betaln, std_cdf
from .distribution import Continuous, draw_values, generate_samples


__all__ = ['Uniform', 'Flat', 'Normal', 'Beta', 'Exponential', 'Laplace',
           'StudentT', 'Cauchy', 'HalfCauchy', 'Gamma', 'Weibull',
           'Bound', 'StudentTpos', 'Lognormal', 'ChiSquared', 'HalfNormal',
           'Wald', 'Pareto', 'InverseGamma', 'ExGaussian']


class PositiveContinuous(Continuous):
    """Base class for positive continuous distributions"""
    def __init__(self, transform=transforms.log, *args, **kwargs):
        super(PositiveContinuous, self).__init__(
            transform=transform, *args, **kwargs)


class UnitContinuous(Continuous):
    """Base class for continuous distributions on [0,1]"""
    def __init__(self, transform=transforms.logodds, *args, **kwargs):
        super(UnitContinuous, self).__init__(
            transform=transform, *args, **kwargs)


def get_tau_sd(tau=None, sd=None):
    """
    Find precision and standard deviation

    .. math::
        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    tau : array-like, optional
    sd : array-like, optional

    Results
    -------
    Returns tuple (tau, sd)

    Notes
    -----
    If neither tau nor sd is provided, returns (1., 1.)
    """
    if tau is None:
        if sd is None:
            sd = 1.
            tau = 1.
        else:
            tau = sd**-2.

    else:
        if sd is not None:
            raise ValueError("Can't pass both tau and sd")
        else:
            sd = tau**-.5

    # cast tau and sd to float in a way that works for both np.arrays
    # and pure python
    tau = 1. * tau
    sd = 1. * sd

    return (tau, sd)


class Uniform(Continuous):
    """
    Continuous uniform log-likelihood.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    Parameters
    ----------
    lower : float
        Lower limit (defaults to 0)
    upper : float
        Upper limit (defaults to 1)
    """
    def __init__(self, lower=0, upper=1, transform='interval',
                 *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)

        self.lower = lower
        self.upper = upper
        self.mean = (upper + lower) / 2.
        self.median = self.mean

        if transform is 'interval':
            self.transform = transforms.interval(lower, upper)

    def random(self, point=None, size=None, repeat=None):
        lower, upper = draw_values([self.lower, self.upper],
                                   point=point)
        return generate_samples(stats.uniform.rvs, loc=lower,
                                scale=upper - lower,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        lower = self.lower
        upper = self.upper

        return bound(-T.log(upper - lower),
                     lower <= value, value <= upper)


class Flat(Continuous):
    """
    Uninformative log-likelihood that returns 0 regardless of
    the passed value.
    """
    def __init__(self, *args, **kwargs):
        super(Flat, self).__init__(*args, **kwargs)
        self.median = 0

    def random(self, point=None, size=None, repeat=None):
        raise ValueError('Cannot sample from Flat distribution')

    def logp(self, value):
        return T.zeros_like(value)


class Normal(Continuous):
    """
    Normal log-likelihood.

    .. math::

       f(x \mid \mu, \tau) =
           \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    Parameters
    ----------
    mu : float
        Mean of the distribution.
    tau : float
        Precision of the distribution, which corresponds to
        :math:`1/\sigma^2` (tau > 0).
    sd : float
        Standard deviation of the distribution. Alternative parameterization.

    .. note::
    - :math:`E(X) = \mu`
    - :math:`Var(X) = 1/\tau`
    """
    def __init__(self, mu=0.0, tau=None, sd=None, *args, **kwargs):
        super(Normal, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.tau, self.sd = get_tau_sd(tau=tau, sd=sd)
        self.variance = 1. / self.tau

    def random(self, point=None, size=None, repeat=None):
        mu, tau, sd = draw_values([self.mu, self.tau, self.sd],
                                  point=point)
        return generate_samples(stats.norm.rvs, loc=mu, scale=tau**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        tau = self.tau
        sd = self.sd
        mu = self.mu
        return bound((-tau * (value - mu)**2 + T.log(tau / np.pi / 2.)) / 2.,
                     tau > 0, sd > 0)


class HalfNormal(PositiveContinuous):
    """
    Half-normal log-likelihood, a normal distribution with mean 0 limited
    to the domain :math:`x \in [0, \infty)`.

    .. math::

       f(x \mid \tau) =
           \sqrt{\frac{2\tau}{\pi}}
           \exp\left\{ {\frac{-x^2 \tau}{2}}\right\}

    :Parameters:
      - `x` : :math:`x \ge 0`
      - `tau` : tau > 0
      - `sd` : sd > 0 (alternative parameterization)

    """
    def __init__(self, tau=None, sd=None, *args, **kwargs):
        super(HalfNormal, self).__init__(*args, **kwargs)
        self.tau, self.sd = get_tau_sd(tau=tau, sd=sd)
        self.mean = T.sqrt(2 / (np.pi * self.tau))
        self.variance = (1. - 2/np.pi) / self.tau

    def random(self, point=None, size=None, repeat=None):
        tau = draw_values([self.tau], point=point)
        return generate_samples(stats.halfnorm.rvs, loc=0., scale=tau**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        tau = self.tau
        sd = self.sd
        return bound(
            -0.5 * tau * value**2 + 0.5 * T.log(tau * 2. / np.pi),
            tau > 0,
            sd > 0,
            value >= 0
        )


class Wald(PositiveContinuous):
    """
    Wald random variable with support :math:`x \in (0, \infty)`.

    .. math::

       f(x \mid \mu, \lambda) =
           \left(\frac{\lambda}{2\np.pi)}\right)^{1/2} x^{-3/2}
           \exp\left\{
               -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2
           \right\}

    Parameters
    ----------
    mu : float, optional
        Mean of the distribution (mu > 0).
    lam : float, optional
        Relative precision (lam > 0).
    phi : float, optional
        Shape. Alternative parametrisation where phi = lam / mu (phi > 0).
    alpha : float, optional
        Shift/location (alpha >= 0).

    The Wald can be instantiated by specifying mu only (so lam=1),
    mu and lam, mu and phi, or lam and phi.

    .. note::
        - :math:`E(X) = \mu`
        - :math:`Var(X) = \frac{\mu^3}{\lambda}`

    References
    ----------
    .. [Tweedie1957]
       Tweedie, M. C. K. (1957).
       Statistical Properties of Inverse Gaussian Distributions I.
       The Annals of Mathematical Statistics, Vol. 28, No. 2, pp. 362-377

    .. [Michael1976]
        Michael, J. R., Schucany, W. R. and Hass, R. W. (1976).
        Generating Random Variates Using Transformations with Multiple Roots.
        The American Statistician, Vol. 30, No. 2, pp. 88-90
    """
    def __init__(self, mu=None, lam=None, phi=None, alpha=0., *args, **kwargs):
        super(Wald, self).__init__(*args, **kwargs)
        self.mu, self.lam, self.phi = self.get_mu_lam_phi(mu, lam, phi)
        self.alpha = alpha
        self.mean = self.mu + alpha
        self.mode = self.mu * (T.sqrt(1. + (1.5 * self.mu / self.lam)**2)
                               - 1.5 * self.mu / self.lam) + alpha
        self.variance = (self.mu**3) / self.lam

    def get_mu_lam_phi(self, mu, lam, phi):
        if mu is None:
            if lam is not None and phi is not None:
                return lam / phi, lam, phi
        else:
            if lam is None:
                if phi is None:
                    return mu, 1., 1. / mu
                else:
                    return mu, mu * phi, phi
            else:
                if phi is None:
                    return mu, lam, lam / mu

        raise ValueError('Wald distribution must specify either mu only, '
                         'mu and lam, mu and phi, or lam and phi.')

    def _random(self, mu, lam, alpha, size=None):
        v = np.random.normal(size=size)**2
        value = (mu + (mu**2) * v / (2. * lam) - mu / (2. * lam)
                 * np.sqrt(4. * mu * lam * v + (mu * v)**2))
        z = np.random.uniform(size=size)
        i = np.floor(z - mu / (mu + value)) * 2 + 1
        value = (value**-i) * (mu**(i + 1))
        return value + alpha

    def random(self, point=None, size=None, repeat=None):
        mu, lam, alpha = draw_values([self.mu, self.lam, self.alpha],
                                     point=point)
        return generate_samples(self._random,
                                mu, lam, alpha,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        lam = self.lam
        alpha = self.alpha
        # value *must* be iid. Otherwise this is wrong.
        return bound(logpow(lam / (2. * np.pi), 0.5)
                     - logpow(value - alpha, 1.5)
                     - (0.5 * lam / (value - alpha)
                        * ((value - alpha - mu) / mu)**2),
                     # XXX these two are redundant. Please, check.
                     value > 0, value - alpha > 0,
                     mu > 0, lam > 0, alpha >= 0)


class Beta(UnitContinuous):
    """
    Beta log-likelihood. The conjugate prior for the parameter
    :math:`p` of the binomial distribution.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)}
           x^{\alpha - 1} (1 - x)^{\beta - 1}

    Parameters
    ----------
    alpha : float
        alpha > 0
    beta : float
        beta > 0

    Alternative parameterization:
    mu : float
        1 > mu > 0
    sd : float
        sd > 0
    .. math::
        alpha = mu * sd
        beta = (1 - mu) * sd

    .. note::
    - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    """
    def __init__(self, alpha=None, beta=None, mu=None, sd=None,
                 *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)

        alpha, beta = self.get_alpha_beta(alpha, beta, mu, sd)
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha / (alpha + beta)
        self.variance = alpha * beta / (
            (alpha + beta)**2 * (alpha + beta + 1))

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sd=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sd is not None):
            alpha = mu * sd
            beta = (1 - mu) * sd
        else:
            raise ValueError('Incompatible parameterization. Either use alpha '
                             'and beta, or mu and sd to specify distribution.')

        return alpha, beta

    def random(self, point=None, size=None, repeat=None):
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point)
        return generate_samples(stats.beta.rvs, alpha, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta

        return bound(logpow(value, alpha - 1) + logpow(1 - value, beta - 1)
                     - betaln(alpha, beta),
                     value >= 0, value <= 1,
                     alpha > 0, beta > 0)


class Exponential(PositiveContinuous):
    """
    Exponential distribution

    Parameters
    ----------
    lam : float
        lam > 0
        rate or inverse scale
    """
    def __init__(self, lam, *args, **kwargs):
        super(Exponential, self).__init__(*args, **kwargs)
        self.lam = lam
        self.mean = 1. / lam
        self.median = self.mean * T.log(2)
        self.mode = 0

        self.variance = lam**-2

    def random(self, point=None, size=None, repeat=None):
        lam = draw_values([self.lam], point=point)
        return generate_samples(np.random.exponential, scale=1./lam,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        lam = self.lam
        return bound(T.log(lam) - lam * value, value > 0, lam > 0)


class Laplace(Continuous):
    """
    Laplace distribution

    Parameters
    ----------
    mu : float
        mean
    b : float
        scale
    """

    def __init__(self, mu, b, *args, **kwargs):
        super(Laplace, self).__init__(*args, **kwargs)
        self.b = b
        self.mean = self.median = self.mode = self.mu = mu

        self.variance = 2 * b**2

    def random(self, point=None, size=None, repeat=None):
        mu, b = draw_values([self.mu, self.b], point=point)
        return generate_samples(np.random.laplace, mu, b,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        b = self.b

        return -T.log(2 * b) - abs(value - mu) / b


class Lognormal(PositiveContinuous):
    """
    Log-normal log-likelihood.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.,


    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\np.pi}}\frac{
        \exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}}{x}

    :Parameters:
      - `x` : x > 0
      - `mu` : Location parameter.
      - `tau` : Scale parameter (tau > 0).

    .. note::

       :math:`E(X)=e^{\mu+\frac{1}{2\tau}}`
       :math:`Var(X)=(e^{1/\tau}-1)e^{2\mu+\frac{1}{\tau}}`

    """
    def __init__(self, mu=0, tau=1, *args, **kwargs):
        super(Lognormal, self).__init__(*args, **kwargs)

        self.mu = mu
        self.tau = tau
        self.mean = T.exp(mu + 1. / (2 * tau))
        self.median = T.exp(mu)
        self.mode = T.exp(mu - 1. / tau)
        self.variance = (T.exp(1./tau) - 1) * T.exp(2 * mu + 1. / tau)

    def _random(self, mu, tau, size=None):
        samples = np.random.normal(size=size)
        return np.exp(mu + (tau**-0.5) * samples)

    def random(self, point=None, size=None, repeat=None):
        mu, tau = draw_values([self.mu, self.tau], point=point)
        return generate_samples(self._random, mu, tau,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        tau = self.tau
        return bound(-0.5 * tau * (T.log(value) - mu)**2
                     + 0.5 * T.log(tau/(2. * np.pi))
                     - T.log(value),
                     tau > 0)


class StudentT(Continuous):
    """
    Non-central Student's T log-likelihood.

    Describes a normal variable whose precision is gamma distributed. If
    only nu parameter is passed, this specifies a standard (central)
    Student's T.

    .. math::
        f(x|\mu,\lambda,\nu) = \frac{\Gamma(\frac{\nu +
        1}{2})}{\Gamma(\frac{\nu}{2})}
        \left(\frac{\lambda}{\np.pi\nu}\right)^{\frac{1}{2}}
        \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    Parameters
    ----------
    nu : int
        Degrees of freedom
    mu : float
        Location parameter (defaults to 0)
    lam : float
        Scale parameter (defaults to 1)
    """
    def __init__(self, nu, mu=0, lam=None, sd=None, *args, **kwargs):
        super(StudentT, self).__init__(*args, **kwargs)
        self.nu = nu = T.as_tensor_variable(nu)
        self.lam, self.sd = get_tau_sd(tau=lam, sd=sd)
        self.mean = self.median = self.mode = self.mu = mu

        self.variance = T.switch((nu > 2) * 1,
                                 (1 / self.lam) * (nu / (nu - 2)),
                                 np.inf)

    def random(self, point=None, size=None, repeat=None):
        nu, mu, lam = draw_values([self.nu, self.mu, self.lam],
                                  point=point)
        return generate_samples(stats.t.rvs, nu, loc=mu, scale=lam**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        nu = self.nu
        mu = self.mu
        lam = self.lam
        sd = self.sd

        return bound(gammaln((nu + 1.0) / 2.0)
                     + .5 * T.log(lam / (nu * np.pi))
                     - gammaln(nu / 2.0)
                     - (nu + 1.0) / 2.0 * T.log1p(lam * (value - mu)**2 / nu),
                     lam > 0, nu > 0, sd > 0)


class Pareto(PositiveContinuous):
    """
    Pareto log-likelihood. The Pareto is a continuous, positive
    probability distribution with two parameters. It is often used
    to characterize wealth distribution, or other examples of the
    80/20 rule.

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha>0)
    m : float
        Scale parameter (m>0)

    .. note::
       - :math:`E(x)=\frac{\alpha m}{\alpha-1} if \alpha > 1`
       - :math:`Var(x)=\frac{m^2 \alpha}{(\alpha-1)^2(\alpha-2)} if \alpha > 2`
    """
    def __init__(self, alpha, m, *args, **kwargs):
        super(Pareto, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.m = m
        self.mean = T.switch(T.gt(alpha, 1), alpha * m / (alpha - 1.), np.inf)
        self.median = m * 2.**(1./alpha)
        self.variance = T.switch(
            T.gt(alpha, 2),
            (alpha * m**2) / ((alpha - 2.) * (alpha - 1.)**2),
            np.inf)

    def _random(self, alpha, m, size=None):
        u = np.random.uniform(size=size)
        return m * (1. - u)**(-1. / alpha)

    def random(self, point=None, size=None, repeat=None):
        alpha, m = draw_values([self.alpha, self.m],
                               point=point)
        return generate_samples(self._random, alpha, m,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        m = self.m
        return bound(T.log(alpha) + logpow(m, alpha)
                     - logpow(value, alpha + 1),
                     value >= m, alpha > 0, m > 0)


class Cauchy(Continuous):
    """
    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{1}{\np.pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    Parameters
    ----------
    alpha : float
        Location parameter
    beta : float
        Scale parameter > 0

    .. note::
    Mode and median are at alpha.

    """

    def __init__(self, alpha, beta, *args, **kwargs):
        super(Cauchy, self).__init__(*args, **kwargs)
        self.median = self.mode = self.alpha = alpha
        self.beta = beta

    def _random(self, alpha, beta, size=None):
        u = np.random.uniform(size=size)
        return alpha + beta * np.tan(np.pi*(u - 0.5))

    def random(self, point=None, size=None, repeat=None):
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point)
        return generate_samples(self._random, alpha, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(- T.log(np.pi) - T.log(beta)
                     - T.log(1 + ((value - alpha) / beta)**2),
                     beta > 0)


class HalfCauchy(PositiveContinuous):
    """
    Half-Cauchy log-likelihood. Simply the absolute value of Cauchy.

    .. math::

       f(x \mid \beta) = \frac{2}{\np.pi \beta [1 + (\frac{x}{\beta})^2]}

    :Parameters:
      - `beta` : Scale parameter (beta > 0).

    .. note::
      - x must be non-negative.
    """

    def __init__(self, beta, *args, **kwargs):
        super(HalfCauchy, self).__init__(*args, **kwargs)
        self.mode = 0
        self.median = beta
        self.beta = beta

    def _random(self, beta, size=None):
        u = np.random.uniform(size=size)
        return beta * np.abs(np.tan(np.pi * (u - 0.5)))

    def random(self, point=None, size=None, repeat=None):
        beta = draw_values([self.beta], point=point)
        return generate_samples(self._random, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        beta = self.beta
        return bound(T.log(2) - T.log(np.pi) - T.log(beta)
                     - T.log(1 + (value / beta)**2),
                     value >= 0, beta > 0)


class Gamma(PositiveContinuous):
    """
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables,
    each of which has mean beta.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    Parameters
    ----------
    x : float
        math:`x \ge 0`
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).

    Alternative parameterization:
    mu : float
        mu > 0
    sd : float
        sd > 0
    .. math::
        alpha =  \frac{mu^2}{sd^2}
        beta = \frac{mu}{sd^2}

    .. note::
    - :math:`E(X) = \frac{\alpha}{\beta}`
    - :math:`Var(X) = \frac{\alpha}{\beta^2}`

    """
    def __init__(self, alpha=None, beta=None, mu=None, sd=None,
                 *args, **kwargs):
        super(Gamma, self).__init__(*args, **kwargs)
        alpha, beta = self.get_alpha_beta(alpha, beta, mu, sd)
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha / beta
        self.mode = T.maximum((alpha - 1) / beta, 0)
        self.variance = alpha / beta**2

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sd=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sd is not None):
            alpha = mu**2 / sd**2
            beta = mu / sd**2
        else:
            raise ValueError('Incompatible parameterization. Either use '
                             'alpha and beta, or mu and sd to specify '
                             'distribution.')

        return alpha, beta

    def random(self, point=None, size=None, repeat=None):
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point)
        return generate_samples(stats.gamma.rvs, alpha, scale=1. / beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(
            -gammaln(alpha) + logpow(
                beta, alpha) - beta * value + logpow(value, alpha - 1),

            value >= 0,
            alpha > 0,
            beta > 0)


class InverseGamma(PositiveContinuous):
    """
    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    Parameters
    ----------
      alpha : float
          Shape parameter (alpha > 0).
      beta : float
          Scale parameter (beta > 0).

    .. note::

    :math:`E(X)=\frac{\beta}{\alpha-1}`  for :math:`\alpha > 1`
    :math:`Var(X)=\frac{\beta^2}{(\alpha-1)^2(\alpha)}`  for :math:`\alpha > 2`
    """
    def __init__(self, alpha, beta=1, *args, **kwargs):
        super(InverseGamma, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.mean = (alpha > 1) * beta / (alpha - 1.) or np.inf
        self.mode = beta / (alpha + 1.)
        self.variance = T.switch(T.gt(alpha, 2),
                                 (beta**2) / (alpha * (alpha - 1.)**2),
                                 np.inf)

    def random(self, point=None, size=None, repeat=None):
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point)
        return generate_samples(stats.invgamma.rvs, a=alpha, scale=beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(logpow(beta, alpha) - gammaln(alpha) - beta / value
                     + logpow(value, -alpha - 1),
                     value > 0, alpha > 0, beta > 0)


class ChiSquared(Gamma):
    """
    Chi-squared :math:`\chi^2` log-likelihood.

    .. math::

       f(x \mid \nu) = \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    :Parameters:
      - `x` : > 0
      - `nu` : [int] Degrees of freedom ( nu > 0 )

    .. note::
      - :math:`E(X)=\nu`
      - :math:`Var(X)=2\nu`
    """
    def __init__(self, nu, *args, **kwargs):
        self.nu = nu
        super(ChiSquared, self).__init__(
            alpha=nu/2., beta=0.5, *args, **kwargs)


class Weibull(PositiveContinuous):
    """
    Weibull log-likelihood

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\alpha x^{\alpha - 1}
           \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    :Parameters:
      - `x` : :math:`x \ge 0`
      - `alpha` : alpha > 0
      - `beta` : beta > 0

    .. note::
      - :math:`E(x)=\beta \Gamma(1+\frac{1}{\alpha})`
      - :math:`median(x)=\Gamma(\log(2))^{1/\alpha}`
      - :math:`Var(x)=\beta^2 \Gamma(1+\frac{2}{\alpha} - \mu^2)`
    """
    def __init__(self, alpha, beta, *args, **kwargs):
        super(Weibull, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.mean = beta * T.exp(gammaln(1 + 1./alpha))
        self.median = beta * T.exp(gammaln(T.log(2)))**(1./alpha)
        self.variance = (beta**2) * T.exp(gammaln(1 + 2./alpha - self.mean**2))

    def random(self, point=None, size=None, repeat=None):
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point)

        def _random(a, b, size=None):
            return b * (-np.log(np.random.uniform(size=size)))**a

        return generate_samples(_random, alpha, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(T.log(alpha) - T.log(beta)
                     + (alpha - 1) * T.log(value/beta)
                     - (value/beta)**alpha,
                     value >= 0, alpha > 0, beta > 0)


class Bounded(Continuous):
    """A bounded distribution."""
    def __init__(self, distribution, lower, upper, *args, **kwargs):
        self.dist = distribution.dist(*args, **kwargs)

        self.__dict__.update(self.dist.__dict__)
        self.__dict__.update(locals())

        if hasattr(self.dist, 'mode'):
            self.mode = self.dist.mode

    def _random(self, lower, upper, point=None, size=None):
        samples = np.zeros(size).flatten()
        i, n = 0, len(samples)
        while i < len(samples):
            sample = self.dist.random(point=point, size=n)
            select = sample[np.logical_and(sample > lower, sample <= upper)]
            samples[i:(i+len(select))] = select[:]
            i += len(select)
            n -= len(select)
        if size is not None:
            return np.reshape(samples, size)
        else:
            return samples

    def random(self, point=None, size=None, repeat=None):
        lower, upper = draw_values([self.lower, self.upper], point=point)
        return generate_samples(self._random, lower, upper, point,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        return bound(self.dist.logp(value),
                     value >= self.lower, value <= self.upper)


class Bound(object):
    """Creates a new bounded distribution"""
    def __init__(self, distribution, lower=-np.inf, upper=np.inf):
        self.distribution = distribution
        self.lower = lower
        self.upper = upper

    def __call__(self, *args, **kwargs):
        first, args = args[0], args[1:]

        return Bounded(first, self.distribution, self.lower, self.upper,
                       *args, **kwargs)

    def dist(self, *args, **kwargs):
        return Bounded.dist(self.distribution, self.lower, self.upper,
                            *args, **kwargs)


StudentTpos = Bound(StudentT, 0)


class ExGaussian(Continuous):
    """
    Exponentially modified Gaussian random variable with
    support :math:`x \in [-\infty, \infty]`.This results from
    the convolution of a normal distribution with an exponential
    distribution.

    .. math::

       f(x \mid \mu, \sigma, \tau) =
           \frac{1}{\nu}\;
           \exp\left\{\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right\}
           \Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)

    where :math:`\Phi` is the cumulative distribution function of the
    standard normal distribution.

    Parameters
    ----------
    mu : float
        Mean of the normal distribution (-inf < mu < inf).
    sigma : float
        Standard deviation of the normal distribution (sigma > 0).
    nu : float
        Mean of the exponential distribution (nu > 0).

    .. note::
        - :math:`E(X) = \mu + \nu`
        - :math:`Var(X) = \sigma^2 + \nu^2`


    References
    ----------
    .. [Rigby2005]
        Rigby R.A. and Stasinopoulos D.M. (2005).
        "Generalized additive models for location, scale and shape"
        Applied Statististics., 54, part 3, pp 507-554.
    .. [Lacouture2008]
        Lacouture, Y. and Couseanou, D. (2008).
        "How to use MATLAB to fit the ex-Gaussian and other probability functions to a distribution of response times".
        Tutorials in Quantitative Methods for Psychology, Vol. 4, No. 1, pp 35-45.
    """
    def __init__(self, mu, sigma, nu, *args, **kwargs):
        super(ExGaussian, self).__init__(*args, **kwargs)
        self.mu = mu
        self.sigma = sigma
        self.nu = nu
        self.mean = mu + nu
        self.variance = (sigma**2) + (nu**2)

    def random(self, point=None, size=None, repeat=None):
        mu, sigma, nu = draw_values([self.mu, self.sigma, self.nu],
                                    point=point)

        def _random(mu, sigma, nu, size=None):
            return (np.random.normal(mu, sigma, size=size)
                    + np.random.exponential(scale=nu, size=size))

        return generate_samples(_random, mu, sigma, nu,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        sigma = self.sigma
        nu = self.nu

        # This condition suggested by exGAUS.R from gamlss
        lp = T.switch(T.gt(nu,  0.05 * sigma),
                      - T.log(nu) + (mu - value) / nu + 0.5 * (sigma / nu)**2
                      + logpow(std_cdf((value - mu) / sigma - sigma / nu), 1.),
                      - T.log(sigma * T.sqrt(2 * np.pi))
                      - 0.5 * ((value - mu) / sigma)**2)
        return bound(lp, sigma > 0., nu > 0.)
