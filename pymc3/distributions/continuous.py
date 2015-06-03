"""
pymc3.distributions

A collection of common probability distributions for stochastic
nodes in PyMC.

"""
from __future__ import division

from .dist_math import *
from numpy.random import uniform as runiform, normal as rnormal

__all__ = ['Uniform', 'Flat', 'Normal', 'Beta', 'Exponential', 'Laplace',
           'T', 'StudentT', 'Cauchy', 'HalfCauchy', 'Gamma', 'Weibull','Bound',
           'Tpos', 'Lognormal', 'ChiSquared', 'HalfNormal', 'Wald',
           'Pareto', 'InverseGamma']

def get_tau_sd(tau=None, sd=None):
    if tau is None:
        if sd is None:
            sd = 1.
            tau = 1.
        else:
            tau = sd ** -2

    else:
        if sd is not None:
            raise ValueError("Can't pass both tau and sd")
        else:
            sd = tau ** -.5

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
    def __init__(self, lower=0, upper=1, *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)
        self.lower = lower
        self.upper = upper
        self.mean = (upper + lower) / 2.
        self.median = self.mean

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
        super(Flat, self).__init__(*args, **kwargs)
        self.median = 0

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
        super(Normal, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu
        self.tau, self.sd = get_tau_sd(tau=tau, sd=sd)
        self.variance = 1. / self.tau

    def logp(self, value):
        tau = self.tau
        sd = self.sd
        mu = self.mu

        return bound(
            (-tau * (value - mu) ** 2 + log(tau / pi / 2.)) / 2.,
            tau > 0,
            sd > 0
        )


class HalfNormal(Continuous):
    """
    Half-normal log-likelihood, a normal distribution with mean 0 limited
    to the domain :math:`x \in [0, \infty)`.

    .. math::
        f(x \mid \tau) = \sqrt{\frac{2\tau}{\pi}}\exp\left\{ {\frac{-x^2 \tau}{2}}\right\}

    :Parameters:
      - `x` : :math:`x \ge 0`
      - `tau` : tau > 0
      - `sd` : sd > 0 (alternative parameterization)

    """
    def __init__(self, tau=None, sd=None, *args, **kwargs):
        super(HalfNormal, self).__init__(*args, **kwargs)
        self.tau, self.sd = get_tau_sd(tau=tau, sd=sd)
        self.mean = sqrt(2 / (pi * self.tau))
        self.variance = (1. - 2/pi) / self.tau

    def logp(self, value):
        tau = self.tau
        sd = self.sd

        return bound(
            -0.5 * tau * value**2 + 0.5 * log(tau * 2. / pi),
            tau > 0,
            sd > 0,
            value >= 0
        )


class Wald(Continuous):
    """
    Wald (inverse Gaussian) log likelihood.

    .. math::
        f(x \mid \mu) = \sqrt{\frac{1}{2\pi x^3}} \exp\left\{ \frac{-(x-\mu)^2}{2 \mu^2 x} \right\}

    Parameters
    ----------
    mu : float
        Mean of the distribution.

    .. note::
    - :math:`E(X) = \mu`
    - :math:`Var(X) = \mu^3`
    """
    def __init__(self, mu, *args, **kwargs):
        super(Wald, self).__init__(*args, **kwargs)
        self.mu = mu
        self.mean = mu
        self.variance = mu**3

    def logp(self, value):
        mu = self.mu

        return bound(
            ((- (value - mu) ** 2) /
                (2. * value * (mu ** 2))) + logpow(2. * pi * value **3, -0.5),
            mu > 0)



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
    def __init__(self, alpha=None, beta=None, mu=None, sd=None, *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)
        alpha, beta = self.get_alpha_beta(alpha, beta, mu, sd)
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha / (alpha + beta)
        self.variance = alpha * beta / (
            (alpha + beta) ** 2 * (alpha + beta + 1))

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sd=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sd is not None):
            alpha = mu * sd
            beta = (1 - mu) * sd
        else:
            raise ValueError('Incompatible parameterization. Either use alpha and beta, or mu and sd to specify distribution. ')

        return alpha, beta

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
        super(Exponential, self).__init__(*args, **kwargs)
        self.lam = lam
        self.mean = 1. / lam
        self.median = self.mean * log(2)
        self.mode = 0

        self.variance = lam ** -2

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
        super(Laplace, self).__init__(*args, **kwargs)
        self.b = b
        self.mean = self.median = self.mode = self.mu = mu

        self.variance = 2 * b ** 2

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
        super(Lognormal, self).__init__(*args, **kwargs)
        self.mu = mu
        self.tau = tau
        self.mean = exp(mu + 1./(2*tau))
        self.median = exp(mu)
        self.mode = exp(mu - 1./tau)

        self.variance = (exp(1./tau) - 1) * exp(2*mu + 1./tau)

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
    def __init__(self, nu, mu=0, lam=None, sd=None, *args, **kwargs):
        super(T, self).__init__(*args, **kwargs)
        self.nu = nu = as_tensor_variable(nu)
        self.lam, self.sd = get_tau_sd(tau=lam, sd=sd)
        self.mean = self.median = self.mode = self.mu = mu

        self.variance = switch((nu > 2) * 1, (1 / self.lam) * (nu / (nu - 2)) , inf)

    def logp(self, value):
        nu = self.nu
        mu = self.mu
        lam = self.lam
        sd = self.sd

        return bound(
            gammaln((nu + 1.0) / 2.0) + .5 * log(lam / (nu * pi)) - gammaln(nu / 2.0) - (nu + 1.0) / 2.0 * log(1.0 + lam * (value - mu) ** 2 / nu),
            lam > 0,
            nu > 0,
            sd > 0)

StudentT = T


class Pareto(Continuous):
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
        self.mean = switch(gt(alpha,1), alpha * m / (alpha - 1.), inf)
        self.variance = switch(gt(alpha,2), (alpha * m**2) / ((alpha - 2.) * (alpha - 1.)**2), inf)

    def logp(self, value):
        alpha = self.alpha
        m = self.m
        return bound(
            log(alpha) + logpow(m, alpha) - logpow(value, alpha+1),

            alpha > 0,
            m > 0,
            value >= m)


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
        super(Cauchy, self).__init__(*args, **kwargs)
        self.median = self.mode = self.alpha = alpha
        self.beta = beta

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(
            -log(pi) - log(beta) - log(1 + ((
                                            value - alpha) / beta) ** 2),
            beta > 0)

class HalfCauchy(Continuous):
    """
    Half-Cauchy log-likelihood. Simply the absolute value of Cauchy.

    .. math::
        f(x \mid \beta) = \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    :Parameters:
      - `beta` : Scale parameter (beta > 0).

    .. note::
      - x must be non-negative.
    """

    def __init__(self, beta, *args, **kwargs):
        super(HalfCauchy, self).__init__(*args, **kwargs)
        self.mode = 0
        self.beta = beta

    def logp(self, value):
        beta = self.beta
        return bound(
            log(2) - log(pi) - log(beta) - log(1 + (value / beta) ** 2),
            beta > 0,
            value >= 0)


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
    def __init__(self, alpha=None, beta=None, mu=None, sd=None, *args, **kwargs):
        super(Gamma, self).__init__(*args, **kwargs)
        alpha, beta = self.get_alpha_beta(alpha, beta, mu, sd)
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha / beta
        self.mode = maximum((alpha - 1) / beta, 0)
        self.variance = alpha / beta ** 2

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sd=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sd is not None):
            alpha = mu ** 2 / sd ** 2
            beta = mu / sd ** 2
        else:
            raise ValueError('Incompatible parameterization. Either use alpha and beta, or mu and sd to specify distribution. ')

        return alpha, beta

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(
            -gammaln(alpha) + logpow(
                beta, alpha) - beta * value + logpow(value, alpha - 1),

            value >= 0,
            alpha > 0,
            beta > 0)


class InverseGamma(Continuous):
    """
    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1} \exp\left(\frac{-\beta}{x}\right)

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
        self.mean = (alpha > 1) * beta / (alpha - 1.) or inf
        self.mode = beta / (alpha + 1.)
        self.variance = switch(gt(alpha, 2), (beta ** 2) / (alpha * (alpha - 1.)**2), inf)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(
            logpow(beta, alpha) - gammaln(alpha) - beta / value + logpow(value, -alpha-1),

            value > 0,
            alpha > 0,
            beta > 0)


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
        super(ChiSquared, self).__init__(alpha=nu/2., beta=0.5, *args, **kwargs)


class Weibull(Continuous):
    """
    Weibull log-likelihood

    .. math::
        f(x \mid \alpha, \beta) = \frac{\alpha x^{\alpha - 1}
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
        self.mean = beta * exp(gammaln(1 + 1./alpha))
        self.median = beta * exp(gammaln(log(2)))**(1./alpha)
        self.variance = (beta**2) * exp(gammaln(1 + 2./alpha - self.mean**2))

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(
            (log(alpha) - log(beta) + (alpha - 1)*log(value/beta)
            - (value/beta)**alpha),
            value >= 0,
            alpha > 0,
            beta > 0)


class Bounded(Continuous):
    """A bounded distribution."""
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
    """Creates a new bounded distribution"""
    def __init__(self, distribution, lower=-inf, upper=inf):
        self.distribution = distribution
        self.lower = lower
        self.upper = upper

    def __call__(self, *args, **kwargs):
        first, args = args[0], args[1:]

        return Bounded(first, self.distribution, self.lower, self.upper, *args, **kwargs)

    def dist(*args, **kwargs):
        return Bounded.dist(self.distribution, self.lower, self.upper, *args, **kwargs)


Tpos = Bound(T, 0)
