"""
pymc.distributions

A collection of common probability distributions for stochastic
nodes in PyMC.

"""
from __future__ import division

from .dist_math import *
from numpy.random import uniform as runiform, normal as rnormal

__all__ = ['Uniform', 'Flat', 'Normal', 'Beta', 'Exponential', 'Laplace',
           'T', 'Cauchy', 'Gamma', 'Bound', 'Tpos', 'Lognormal']

def get_tau(tau=None, sd=None):
    if tau is None:
        if sd is None:
            return 1.
        else:
            return sd ** -2

    else:
        if sd is not None:
            raise ValueError("Can't pass both tau and sd")
        else:
            return tau

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
    def __init__(self, lower=0, upper=1, *args, **kwargs):
        Continuous.__init__(self, *args, **kwargs)

        mean = (upper + lower) / 2.
        median = mean

        self.__dict__.update(locals())

    def logp(self, value):
        lower = self.lower 
        upper = self.upper

        return bound(
            -log(upper - lower),
            lower <= value, value <= upper)

    def random(self, size=None):
        return runiform(self.upper, self.lower, size)


class Flat(Continuous):
    """
    Uninformative log-likelihood that returns 0 regardless of
    the passed value.
    """
    def __init__(self, *args, **kwargs):
        Continuous.__init__(self, *args, **kwargs)
        median = 0
        self.__dict__.update(locals())

    def logp(self, value):
        return zeros_like(value)



class Normal(Continuous):
    """
    Normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

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
        Continuous.__init__(self, *args, **kwargs)

        tau = get_tau(tau=tau, sd=sd)
        mean = mu
        variance = 1. / tau

        median = mean
        mode = mean

        self.__dict__.update(locals())

    def logp(self, value):
        tau = self.tau 
        mu = self.mu

        return bound(
            (-tau * (value - mu) ** 2 + log(tau / pi / 2.)) / 2.,
            tau > 0)


class Beta(Continuous):
    """
    Beta log-likelihood. The conjugate prior for the parameter
    :math:`p` of the binomial distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}

    Parameters
    ----------
    alpha : float
        alpha > 0
    beta : float
        beta > 0

    .. note::
    - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    def __init__(self, alpha, beta, *args, **kwargs):
        Continuous.__init__(self, *args, **kwargs)

        mean = alpha / (alpha + beta)
        variance = alpha * beta / (
            (alpha + beta) ** 2 * (alpha + beta + 1))

        self.__dict__.update(locals())



    def logp(self, value):
        alpha = self.alpha
        beta = self.beta

        return bound(
            gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) +
            logpow(
                value, alpha - 1) + logpow(1 - value, beta - 1),
            0 <= value, value <= 1,
            alpha > 0,
            beta > 0)


class Exponential(Continuous):
    """
    Exponential distribution

    Parameters
    ----------
    lam : float
        lam > 0
        rate or inverse scale
    """
    def __init__(self, lam, *args, **kwargs):
        Continuous.__init__(self, *args, **kwargs)

        mean = 1. / lam
        median = mean * log(2)
        mode = 0

        variance = lam ** -2
        self.__dict__.update(locals())

    def logp(self, value):
        lam = self.lam
        return bound(log(lam) - lam * value,
                     value > 0,
                     lam > 0)



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
        Continuous.__init__(self, *args, **kwargs)
        mean = median = mode = mu

        variance = 2 * b ** 2

        self.__dict__.update(locals())

    def logp(self, value):
        mu = self.mu 
        b = self.b

        return -log(2 * b) - abs(value - mu) / b



class Lognormal(Continuous):
    """
    Log-normal log-likelihood.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}}\frac{
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
        Continuous.__init__(self, *args, **kwargs)
        mean = exp(mu + 1./(2*tau))
        median = exp(mu)
        mode = exp(mu - 1./tau)

        variance = (exp(1./tau) - 1) * exp(2*mu + 1./tau)

        self.__dict__.update(locals())

    def logp(self, value):
        mu = self.mu
        tau = self.tau

        return bound(
            -0.5*tau*(log(value) - mu)**2 + 0.5*log(tau/(2.*pi)) - log(value),
            tau > 0)


class T(Continuous):
    """
    Non-central Student's T log-likelihood.

    Describes a normal variable whose precision is gamma distributed. If
    only nu parameter is passed, this specifies a standard (central)
    Student's T.

    .. math::
        f(x|\mu,\lambda,\nu) = \frac{\Gamma(\frac{\nu +
        1}{2})}{\Gamma(\frac{\nu}{2})}
        \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
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
    def __init__(self, nu, mu=0, lam=1, *args, **kwargs):
        Continuous.__init__(self, *args, **kwargs)
        mean = mu
        median = mu
        mode = mu

        variance = switch((nu > 2) * 1, nu / (nu - 2) / lam, inf)
        self.__dict__.update(locals())

    def logp(self, value):
        nu = self.nu
        mu = self.mu
        lam = self.lam 

        return bound(
            gammaln((nu + 1.0) / 2.0) + .5 * log(lam / (nu * pi)) - gammaln(nu / 2.0) - (nu + 1.0) / 2.0 * log(1.0 + lam * (value - mu) ** 2 / nu),
            lam > 0,
            nu > 0)


class Cauchy(Continuous): 
    """
    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

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
        Continuous.__init__(self, *args, **kwargs)
        median = mode = alpha
        self.__dict__.update(locals())

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(
            -log(pi) - log(beta) - log(1 + ((
                                            value - alpha) / beta) ** 2),
            beta > 0)

class Gamma(Continuous): 
    """
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables, each
    of which has mean beta.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    Parameters
    ----------
    x : float
        math:`x \ge 0`
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).

    .. note::
    - :math:`E(X) = \frac{\alpha}{\beta}`
    - :math:`Var(X) = \frac{\alpha}{\beta^2}`

    """
    def __init__(self, alpha, beta, *args, **kwargs):
        Continuous.__init__(self, *args, **kwargs)

        mean = alpha / beta
        median = maximum((alpha - 1) / beta, 0)
        variance = alpha / beta ** 2

        self.__dict__.update(locals())

    def logp(self, value):
        alpha = self.alpha 
        beta = self.beta
        return bound(
            -gammaln(alpha) + logpow(
                beta, alpha) - beta * value + logpow(value, alpha - 1),

            value >= 0,
            alpha > 0,
            beta > 0)

class Bounded(Continuous):
    def __init__(self, distribution, lower, upper, *args, **kwargs):
        self.dist = distribution.dist(*args, **kwargs)

        self.__dict__.update(self.dist.__dict__)
        self.__dict__.update(locals())

        if hasattr(self.dist, 'mode'):
            self.mode = self.dist.mode

    def logp(self, value):
        return bound(
            self.dist.logp(value),

            self.lower <= value, value <= self.upper)



class Bound(object): 
    def __init__(self, distribution, lower=-inf, upper=inf):
        """A bounded distribution."""
        self.distribution = distribution
        self.lower = lower
        self.upper = upper

    def __call__(self, *args, **kwargs):
        first, args = args[0], args[1:]
        
        return Bounded(first, self.distribution, self.lower, self.upper, *args, **kwargs)

    def dist(*args, **kwargs):
        return Bounded.dist(self.distribution, self.lower, self.upper, *args, **kwargs)


Tpos = Bound(T, 0)
