"""
pymc3.distributions

A collection of common probability distributions for stochastic
nodes in PyMC.

"""
from __future__ import division

import numpy as np
import theano.tensor as tt
from scipy import stats
import warnings

from pymc3.theanof import floatX
from . import transforms

from .dist_math import bound, logpow, gammaln, betaln, std_cdf, i0, i1, alltrue_elemwise
from .distribution import Continuous, draw_values, generate_samples, Bound

__all__ = ['Uniform', 'Flat', 'Normal', 'Beta', 'Exponential', 'Laplace',
           'StudentT', 'Cauchy', 'HalfCauchy', 'Gamma', 'Weibull',
           'HalfStudentT', 'StudentTpos', 'Lognormal', 'ChiSquared',
           'HalfNormal', 'Wald', 'Pareto', 'InverseGamma', 'ExGaussian',
           'VonMises', 'SkewNormal']


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

def assert_negative_support(var, label, distname, value=-1e-6):
    # Checks for evidence of positive support for a variable
    if var is None:
        return
    try:
        # Transformed distribution
        support = np.isfinite(var.transformed.distribution.dist
                                .logp(value).tag.test_value)
    except AttributeError:
        try:
            # Untransformed distribution
            support = np.isfinite(var.distribution.logp(value).tag.test_value)
        except AttributeError:
            # Otherwise no direct evidence of non-positive support
            support = False

    if np.any(support):
        msg = "The variable specified for {0} has negative support for {1}, ".format(label, distname)
        msg += "likely making it unsuitable for this parameter."
        warnings.warn(msg)


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

    return (floatX(tau), floatX(sd))


class Uniform(Continuous):
    R"""
    Continuous uniform log-likelihood.

    .. math::

       f(x \mid lower, upper) = \frac{1}{upper-lower}

    ========  =====================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper}{2}`
    Variance  :math:`\dfrac{(upper - lower)^2}{12}`
    ========  =====================================

    Parameters
    ----------
    lower : float
        Lower limit.
    upper : float
        Upper limit.
    """

    def __init__(self, lower=0, upper=1, transform='interval',
                 *args, **kwargs):
        super(Uniform, self).__init__(*args, **kwargs)

        self.lower = lower = tt.as_tensor_variable(lower)
        self.upper = upper = tt.as_tensor_variable(upper)
        self.mean = (upper + lower) / 2.
        self.median = self.mean

        if transform == 'interval':
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
        return bound(-tt.log(upper - lower),
                     value >= lower, value <= upper)


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
        return tt.zeros_like(value)


class Normal(Continuous):
    R"""
    Univariate normal log-likelihood.

    .. math::

       f(x \mid \mu, \tau) =
           \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{1}{\tau}` or :math:`\sigma^2`
    ========  ==========================================

    Normal distribution can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}

    Parameters
    ----------
    mu : float
        Mean.
    sd : float
        Standard deviation (sd > 0).
    tau : float
        Precision (tau > 0).
    """

    def __init__(self, mu=0, sd=None, tau=None, **kwargs):
        tau, sd = get_tau_sd(tau=tau, sd=sd)
        self.sd = tt.as_tensor_variable(sd)
        self.tau = tt.as_tensor_variable(tau)

        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(mu)
        self.variance = 1. / self.tau

        assert_negative_support(sd, 'sd', 'Normal')
        assert_negative_support(tau, 'tau', 'Normal')

        super(Normal, self).__init__(**kwargs)

    def random(self, point=None, size=None, repeat=None):
        mu, tau, _ = draw_values([self.mu, self.tau, self.sd],
                                 point=point)
        return generate_samples(stats.norm.rvs, loc=mu, scale=tau**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        sd = self.sd
        tau = self.tau
        mu = self.mu

        return bound((-tau * (value - mu)**2 + tt.log(tau / np.pi / 2.)) / 2.,
                     sd > 0)


class HalfNormal(PositiveContinuous):
    R"""
    Half-normal log-likelihood.

    .. math::

       f(x \mid \tau) =
           \sqrt{\frac{2\tau}{\pi}}
           \exp\left\{ {\frac{-x^2 \tau}{2}}\right\}

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`0`
    Variance  :math:`\dfrac{1}{\tau}` or :math:`\sigma^2`
    ========  ==========================================

    Parameters
    ----------
    sd : float
        Standard deviation (sd > 0).
    tau : float
        Precision (tau > 0).
    """

    def __init__(self, sd=None, tau=None, *args, **kwargs):
        super(HalfNormal, self).__init__(*args, **kwargs)
        tau, sd = get_tau_sd(tau=tau, sd=sd)

        self.sd = sd = tt.as_tensor_variable(sd)
        self.tau = tau = tt.as_tensor_variable(tau)

        self.mean = tt.sqrt(2 / (np.pi * self.tau))
        self.variance = (1. - 2 / np.pi) / self.tau

        assert_negative_support(tau, 'tau', 'HalfNormal')
        assert_negative_support(sd, 'sd', 'HalfNormal')

    def random(self, point=None, size=None, repeat=None):
        sd = draw_values([self.sd], point=point)
        return generate_samples(stats.halfnorm.rvs, loc=0., scale=sd,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        tau = self.tau
        sd = self.sd
        return bound(-0.5 * tau * value**2 + 0.5 * tt.log(tau * 2. / np.pi),
                     value >= 0,
                     tau > 0, sd > 0)


class Wald(PositiveContinuous):
    R"""
    Wald log-likelihood.

    .. math::

       f(x \mid \mu, \lambda) =
           \left(\frac{\lambda}{2\pi)}\right)^{1/2} x^{-3/2}
           \exp\left\{
               -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2
           \right\}

    ========  =============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{\mu^3}{\lambda}`
    ========  =============================

    Wald distribution can be parameterized either in terms of lam or phi.
    The link between the two parametrizations is given by

    .. math::

       \phi = \dfrac{\lambda}{\mu}

    Parameters
    ----------
    mu : float, optional
        Mean of the distribution (mu > 0).
    lam : float, optional
        Relative precision (lam > 0).
    phi : float, optional
        Alternative shape parameter (phi > 0).
    alpha : float, optional
        Shift/location parameter (alpha >= 0).

    Notes
    -----
    To instantiate the distribution specify any of the following

    - only mu (in this case lam will be 1)
    - mu and lam
    - mu and phi
    - lam and phi

    References
    ----------
    .. [Tweedie1957] Tweedie, M. C. K. (1957).
       Statistical Properties of Inverse Gaussian Distributions I.
       The Annals of Mathematical Statistics, Vol. 28, No. 2, pp. 362-377

    .. [Michael1976] Michael, J. R., Schucany, W. R. and Hass, R. W. (1976).
        Generating Random Variates Using Transformations with Multiple Roots.
        The American Statistician, Vol. 30, No. 2, pp. 88-90
    """

    def __init__(self, mu=None, lam=None, phi=None, alpha=0., *args, **kwargs):
        super(Wald, self).__init__(*args, **kwargs)
        mu, lam, phi = self.get_mu_lam_phi(mu, lam, phi)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.lam = lam = tt.as_tensor_variable(lam)
        self.phi = phi =tt.as_tensor_variable(phi)

        self.mean = self.mu + self.alpha
        self.mode = self.mu * (tt.sqrt(1. + (1.5 * self.mu / self.lam)**2)
                               - 1.5 * self.mu / self.lam) + self.alpha
        self.variance = (self.mu**3) / self.lam

        assert_negative_support(phi, 'phi', 'Wald')
        assert_negative_support(mu, 'mu', 'Wald')
        assert_negative_support(lam, 'lam', 'Wald')

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
    R"""
    Beta log-likelihood.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`\dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
    ========  ==============================================================

    Beta distribution can be parameterized either in terms of alpha and
    beta or mean and standard deviation. The link between the two
    parametrizations is given by

    .. math::

       \alpha &= \mu \kappa \\
       \beta  &= (1 - \mu) \kappa

       \text{where } \kappa = \frac{\mu(1-\mu)}{\sigma^2} - 1

    Parameters
    ----------
    alpha : float
        alpha > 0.
    beta : float
        beta > 0.
    mu : float
        Alternative mean (0 < mu < 1).
    sd : float
        Alternative standard deviation (sd > 0).

    Notes
    -----
    Beta distribution is a conjugate prior for the parameter :math:`p` of
    the binomial distribution.
    """

    def __init__(self, alpha=None, beta=None, mu=None, sd=None,
                 *args, **kwargs):
        super(Beta, self).__init__(*args, **kwargs)

        alpha, beta = self.get_alpha_beta(alpha, beta, mu, sd)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.beta = beta = tt.as_tensor_variable(beta)

        self.mean = self.alpha / (self.alpha + self.beta)
        self.variance = self.alpha * self.beta / (
            (self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))

        assert_negative_support(alpha, 'alpha', 'Beta')
        assert_negative_support(beta, 'beta', 'Beta')

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sd=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sd is not None):
            kappa = mu * (1 - mu) / sd**2 - 1
            alpha = mu * kappa
            beta = (1 - mu) * kappa
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
    R"""
    Exponential log-likelihood.

    .. math::

       f(x \mid \lambda) = \lambda \exp\left\{ -\lambda x \right\}

    ========  ============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{1}{\lambda}`
    Variance  :math:`\dfrac{1}{\lambda^2}`
    ========  ============================

    Parameters
    ----------
    lam : float
        Rate or inverse scale (lam > 0)
    """

    def __init__(self, lam, *args, **kwargs):
        super(Exponential, self).__init__(*args, **kwargs)
        self.lam = lam = tt.as_tensor_variable(lam)
        self.mean = 1. / self.lam
        self.median = self.mean * tt.log(2)
        self.mode = 0

        self.variance = self.lam**-2

        assert_negative_support(lam, 'lam', 'Exponential')

    def random(self, point=None, size=None, repeat=None):
        lam = draw_values([self.lam], point=point)
        return generate_samples(np.random.exponential, scale=1. / lam,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        lam = self.lam
        return bound(tt.log(lam) - lam * value, value > 0, lam > 0)


class Laplace(Continuous):
    R"""
    Laplace log-likelihood.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{1}{2b} \exp \left\{ - \frac{|x - \mu|}{b} \right\}

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`2 b^2`
    ========  ========================

    Parameters
    ----------
    mu : float
        Location parameter.
    b : float
        Scale parameter (b > 0).
    """

    def __init__(self, mu, b, *args, **kwargs):
        super(Laplace, self).__init__(*args, **kwargs)
        self.b = b = tt.as_tensor_variable(b)
        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(mu)

        self.variance = 2 * self.b**2

        assert_negative_support(b, 'b', 'Laplace')

    def random(self, point=None, size=None, repeat=None):
        mu, b = draw_values([self.mu, self.b], point=point)
        return generate_samples(np.random.laplace, mu, b,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        b = self.b

        return -tt.log(2 * b) - abs(value - mu) / b


class Lognormal(PositiveContinuous):
    R"""
    Log-normal log-likelihood.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    .. math::

       f(x \mid \mu, \tau) =
           \sqrt{\frac{\tau}{2\pi}}
           \frac{\exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}}{x}

    ========  ================================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`\exp\{\mu + \frac{1}{2\tau}\}`
    Variance  :math:`\exp\{\frac{1}{\tau} - 1\} \exp\{2\mu + \frac{1}{\tau}\}`
    ========  ================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    tau : float
        Scale parameter (tau > 0).
    """

    def __init__(self, mu=0, sd=None, tau=None, *args, **kwargs):
        super(Lognormal, self).__init__(*args, **kwargs)
        tau, sd = get_tau_sd(tau=tau, sd=sd)

        self.mu = mu = tt.as_tensor_variable(mu)
        self.tau = tau = tt.as_tensor_variable(tau)
        self.sd = sd = tt.as_tensor_variable(sd)

        self.mean = tt.exp(self.mu + 1. / (2 * self.tau))
        self.median = tt.exp(self.mu)
        self.mode = tt.exp(self.mu - 1. / self.tau)
        self.variance = (tt.exp(1. / self.tau) - 1) * tt.exp(2 * self.mu + 1. / self.tau)

        assert_negative_support(tau, 'tau', 'Lognormal')
        assert_negative_support(sd, 'sd', 'Lognormal')

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
        return bound(-0.5 * tau * (tt.log(value) - mu)**2
                     + 0.5 * tt.log(tau / (2. * np.pi))
                     - tt.log(value),
                     tau > 0)


class StudentT(Continuous):
    R"""
    Non-central Student's T log-likelihood.

    Describes a normal variable whose precision is gamma distributed.
    If only nu parameter is passed, this specifies a standard (central)
    Student's T.

    .. math::

       f(x|\mu,\lambda,\nu) =
           \frac{\Gamma(\frac{\nu + 1}{2})}{\Gamma(\frac{\nu}{2})}
           \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
           \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    ========  ========================

    Parameters
    ----------
    nu : int
        Degrees of freedom (nu > 0).
    mu : float
        Location parameter.
    lam : float
        Scale parameter (lam > 0).
    """

    def __init__(self, nu, mu=0, lam=None, sd=None, *args, **kwargs):
        super(StudentT, self).__init__(*args, **kwargs)
        self.nu = nu = tt.as_tensor_variable(nu)
        lam, sd = get_tau_sd(tau=lam, sd=sd)
        self.lam = lam = tt.as_tensor_variable(lam)
        self.sd = sd = tt.as_tensor_variable(sd)
        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(mu)

        self.variance = tt.switch((nu > 2) * 1,
                                  (1 / self.lam) * (nu / (nu - 2)),
                                  np.inf)

        assert_negative_support(lam, 'lam (sd)', 'StudentT')
        assert_negative_support(nu, 'nu', 'StudentT')

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
                     + .5 * tt.log(lam / (nu * np.pi))
                     - gammaln(nu / 2.0)
                     - (nu + 1.0) / 2.0 * tt.log1p(lam * (value - mu)**2 / nu),
                     lam > 0, nu > 0, sd > 0)


class Pareto(PositiveContinuous):
    R"""
    Pareto log-likelihood.

    Often used to characterize wealth distribution, or other examples of the
    80/20 rule.

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    ========  =============================================================
    Support   :math:`x \in [m, \infty)`
    Mean      :math:`\dfrac{\alpha m}{\alpha - 1}` for :math:`\alpha \ge 1`
    Variance  :math:`\dfrac{m \alpha}{(\alpha - 1)^2 (\alpha - 2)}`
              for :math:`\alpha > 2`
    ========  =============================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    m : float
        Scale parameter (m > 0).
    """

    def __init__(self, alpha, m, *args, **kwargs):
        super(Pareto, self).__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.m = m = tt.as_tensor_variable(m)

        self.mean = tt.switch(tt.gt(alpha, 1), alpha *
                              m / (alpha - 1.), np.inf)
        self.median = m * 2.**(1. / alpha)
        self.variance = tt.switch(
            tt.gt(alpha, 2),
            (alpha * m**2) / ((alpha - 2.) * (alpha - 1.)**2),
            np.inf)

        assert_negative_support(alpha, 'alpha', 'Pareto')
        assert_negative_support(m, 'm', 'Pareto')


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
        return bound(tt.log(alpha) + logpow(m, alpha)
                     - logpow(value, alpha + 1),
                     value >= m, alpha > 0, m > 0)


class Cauchy(Continuous):
    R"""
    Cauchy log-likelihood.

    Also known as the Lorentz or the Breit-Wigner distribution.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mode      :math:`\alpha`
    Mean      undefined
    Variance  undefined
    ========  ========================

    Parameters
    ----------
    alpha : float
        Location parameter
    beta : float
        Scale parameter > 0
    """

    def __init__(self, alpha, beta, *args, **kwargs):
        super(Cauchy, self).__init__(*args, **kwargs)
        self.median = self.mode = self.alpha = tt.as_tensor_variable(alpha)
        self.beta = tt.as_tensor_variable(beta)

        assert_negative_support(beta, 'beta', 'Cauchy')

    def _random(self, alpha, beta, size=None):
        u = np.random.uniform(size=size)
        return alpha + beta * np.tan(np.pi * (u - 0.5))

    def random(self, point=None, size=None, repeat=None):
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point)
        return generate_samples(self._random, alpha, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(- tt.log(np.pi) - tt.log(beta)
                     - tt.log1p(((value - alpha) / beta)**2),
                     beta > 0)


class HalfCauchy(PositiveContinuous):
    R"""
    Half-Cauchy log-likelihood.

    .. math::

       f(x \mid \beta) = \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mode      0
    Mean      undefined
    Variance  undefined
    ========  ========================

    Parameters
    ----------
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, beta, *args, **kwargs):
        super(HalfCauchy, self).__init__(*args, **kwargs)
        self.mode = tt.as_tensor_variable(0)
        self.median = tt.as_tensor_variable(beta)
        self.beta = tt.as_tensor_variable(beta)

        assert_negative_support(beta, 'beta', 'HalfCauchy')

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
        return bound(tt.log(2) - tt.log(np.pi) - tt.log(beta)
                     - tt.log1p((value / beta)**2),
                     value >= 0, beta > 0)


class Gamma(PositiveContinuous):
    R"""
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables,
    each of which has mean beta.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    ========  ===============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\alpha}{\beta}`
    Variance  :math:`\dfrac{\alpha}{\beta^2}`
    ========  ===============================

    Gamma distribution can be parameterized either in terms of alpha and
    beta or mean and standard deviation. The link between the two
    parametrizations is given by

    .. math::

       \alpha &= \frac{\mu^2}{\sigma^2} \\
       \beta &= \frac{\mu}{\sigma^2}

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).
    mu : float
        Alternative shape parameter (mu > 0).
    sd : float
        Alternative scale parameter (sd > 0).
    """

    def __init__(self, alpha=None, beta=None, mu=None, sd=None,
                 *args, **kwargs):
        super(Gamma, self).__init__(*args, **kwargs)
        alpha, beta = self.get_alpha_beta(alpha, beta, mu, sd)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.beta = beta = tt.as_tensor_variable(beta)
        self.mean = alpha / beta
        self.mode = tt.maximum((alpha - 1) / beta, 0)
        self.variance = alpha / beta**2

        assert_negative_support(alpha, 'alpha', 'Gamma')
        assert_negative_support(beta, 'beta', 'Gamma')

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
    R"""
    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    ========  ======================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
    Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha)}`
              for :math:`\alpha > 2`
    ========  ======================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, alpha, beta=1, *args, **kwargs):
        super(InverseGamma, self).__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.beta = beta = tt.as_tensor_variable(beta)

        self.mean = self._calculate_mean()
        self.mode = beta / (alpha + 1.)
        self.variance = tt.switch(tt.gt(alpha, 2),
                                  (beta**2) / (alpha * (alpha - 1.)**2),
                                  np.inf)
        assert_negative_support(alpha, 'alpha', 'InverseGamma')
        assert_negative_support(beta, 'beta', 'InverseGamma')

    def _calculate_mean(self):
        m = self.beta / (self.alpha - 1.)
        try:
            return (self.alpha > 1) * m or np.inf
        except ValueError:  # alpha is an array
            m[self.alpha <= 1] = np.inf
            return m

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
    R"""
    :math:`\chi^2` log-likelihood.

    .. math::

       f(x \mid \nu) = \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    ========  ===============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\nu`
    Variance  :math:`2 \nu`
    ========  ===============================

    Parameters
    ----------
    nu : int
        Degrees of freedom (nu > 0).
    """

    def __init__(self, nu, *args, **kwargs):
        self.nu = nu = tt.as_tensor_variable(nu)
        super(ChiSquared, self).__init__(alpha=nu / 2., beta=0.5,
                                         *args, **kwargs)


class Weibull(PositiveContinuous):
    R"""
    Weibull log-likelihood.

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\alpha x^{\alpha - 1}
           \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    ========  ====================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\beta \Gamma(1 + \frac{1}{\alpha})`
    Variance  :math:`\beta^2 \Gamma(1 + \frac{2}{\alpha} - \mu^2)`
    ========  ====================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, alpha, beta, *args, **kwargs):
        super(Weibull, self).__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.beta = beta = tt.as_tensor_variable(beta)
        self.mean = beta * tt.exp(gammaln(1 + 1. / alpha))
        self.median = beta * tt.exp(gammaln(tt.log(2)))**(1. / alpha)
        self.variance = (beta**2) * \
            tt.exp(gammaln(1 + 2. / alpha - self.mean**2))

        assert_negative_support(alpha, 'alpha', 'Weibull')
        assert_negative_support(beta, 'beta', 'Weibull')

    def random(self, point=None, size=None, repeat=None):
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point)

        def _random(a, b, size=None):
            return b * (-np.log(np.random.uniform(size=size)))**(1 / a)

        return generate_samples(_random, alpha, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(tt.log(alpha) - tt.log(beta)
                     + (alpha - 1) * tt.log(value / beta)
                     - (value / beta)**alpha,
                     value >= 0, alpha > 0, beta > 0)


def StudentTpos(*args, **kwargs):
    warnings.warn("StudentTpos has been deprecated. In future, use HalfStudentT instead.",
                DeprecationWarning)
    return HalfStudentT(*args, **kwargs)

HalfStudentT = Bound(StudentT, lower=0)


class ExGaussian(Continuous):
    R"""
    Exponentially modified Gaussian log-likelihood.

    Results from the convolution of a normal distribution with an exponential
    distribution.

    .. math::

       f(x \mid \mu, \sigma, \tau) =
           \frac{1}{\nu}\;
           \exp\left\{\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right\}
           \Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)

    where :math:`\Phi` is the cumulative distribution function of the
    standard normal distribution.

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \nu`
    Variance  :math:`\sigma^2 + \nu^2`
    ========  ========================

    Parameters
    ----------
    mu : float
        Mean of the normal distribution.
    sigma : float
        Standard deviation of the normal distribution (sigma > 0).
    nu : float
        Mean of the exponential distribution (nu > 0).

    References
    ----------
    .. [Rigby2005] Rigby R.A. and Stasinopoulos D.M. (2005).
        "Generalized additive models for location, scale and shape"
        Applied Statististics., 54, part 3, pp 507-554.

    .. [Lacouture2008] Lacouture, Y. and Couseanou, D. (2008).
        "How to use MATLAB to fit the ex-Gaussian and other probability
        functions to a distribution of response times".
        Tutorials in Quantitative Methods for Psychology,
        Vol. 4, No. 1, pp 35-45.
    """

    def __init__(self, mu, sigma, nu, *args, **kwargs):
        super(ExGaussian, self).__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.sigma = sigma = tt.as_tensor_variable(sigma)
        self.nu = nu = tt.as_tensor_variable(nu)
        self.mean = mu + nu
        self.variance = (sigma**2) + (nu**2)

        assert_negative_support(sigma, 'sigma', 'ExGaussian')
        assert_negative_support(nu, 'nu', 'ExGaussian')

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
        lp = tt.switch(tt.gt(nu,  0.05 * sigma),
                       - tt.log(nu) + (mu - value) / nu + 0.5 * (sigma / nu)**2
                       + logpow(std_cdf((value - mu) / sigma - sigma / nu), 1.),
                       - tt.log(sigma * tt.sqrt(2 * np.pi))
                       - 0.5 * ((value - mu) / sigma)**2)
        return bound(lp, sigma > 0., nu > 0.)


class VonMises(Continuous):
    R"""
    Univariate VonMises log-likelihood.

    .. math::
        f(x \mid \mu, \kappa) =
            \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    where :I_0 is the modified Bessel function of order 0.

    ========  ==========================================
    Support   :math:`x \in [-\pi, \pi]`
    Mean      :math:`\mu`
    Variance  :math:`1-\frac{I_1(\kappa)}{I_0(\kappa)}`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Mean.
    kappa : float
        Concentration (\frac{1}{kappa} is analogous to \sigma^2).
    """

    def __init__(self, mu=0.0, kappa=None, transform='circular',
                 *args, **kwargs):
        super(VonMises, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(mu)
        self.kappa = kappa = tt.as_tensor_variable(kappa)
        self.variance = 1 - i1(kappa) / i0(kappa)

        if transform == 'circular':
            self.transform = transforms.Circular()

        assert_negative_support(kappa, 'kappa', 'VonMises')

    def random(self, point=None, size=None, repeat=None):
        mu, kappa = draw_values([self.mu, self.kappa],
                                point=point)
        return generate_samples(stats.vonmises.rvs, loc=mu, kappa=kappa,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        kappa = self.kappa
        return bound(kappa * tt.cos(mu - value) - tt.log(2 * np.pi * i0(kappa)), value >= -np.pi, value <= np.pi, kappa >= 0)


class SkewNormal(Continuous):
    R"""
    Univariate skew-normal log-likelihood.

    .. math::
       f(x \mid \mu, \tau, \alpha) =
       2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \sigma \sqrt{\frac{2}{\pi}} \frac {\alpha }{{\sqrt {1+\alpha ^{2}}}}`
    Variance  :math:`\sigma^2 \left(  1-\frac{2\alpha^2}{(\alpha^2+1) \pi} \right)`
    ========  ==========================================

    Skew-normal distribution can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::
       \tau = \dfrac{1}{\sigma^2}

    Parameters
    ----------
    mu : float
        Location parameter.
    sd : float
        Scale parameter (sd > 0).
    tau : float
        Alternative scale parameter (tau > 0).
    alpha : float
        Skewness parameter.

    Notes
    -----
    When alpha=0 we recover the Normal distribution and mu becomes the mean,
    tau the precision and sd the standard deviation. In the limit of alpha
    approaching plus/minus infinite we get a half-normal distribution.

    """
    def __init__(self, mu=0.0, sd=None, tau=None, alpha=1,  *args, **kwargs):
        super(SkewNormal, self).__init__(*args, **kwargs)
        tau, sd = get_tau_sd(tau=tau, sd=sd)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.tau = tt.as_tensor_variable(tau)
        self.sd = tt.as_tensor_variable(sd)

        self.alpha = alpha = tt.as_tensor_variable(alpha)

        self.mean = mu + self.sd * (2 / np.pi)**0.5 * alpha / (1 + alpha**2)**0.5
        self.variance = self.sd**2 * (1 - (2 * alpha**2) / ((1 + alpha**2) * np.pi))

        assert_negative_support(tau, 'tau', 'SkewNormal')
        assert_negative_support(sd, 'sd', 'SkewNormal')

    def random(self, point=None, size=None, repeat=None):
        mu, tau, _, alpha = draw_values(
            [self.mu, self.tau, self.sd, self.alpha], point=point)
        return generate_samples(stats.skewnorm.rvs,
                                a=alpha, loc=mu, scale=tau**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        tau = self.tau
        sd = self.sd
        mu = self.mu
        alpha = self.alpha
        return bound(
            tt.log(1 +
            tt.erf(((value - mu) * tt.sqrt(tau) * alpha) / tt.sqrt(2)))
            + (-tau * (value - mu)**2
            + tt.log(tau / np.pi / 2.)) / 2.,
            tau > 0, sd > 0)


class Triangular(Continuous):
    """
    Continuous Triangular log-likelihood
    Implemented by J. A. Fonseca 22/12/16

    Parameters
    ----------
    lower : float
        Lower limit.
    c: float
        mode
    upper : float
        Upper limit.
    """

    def __init__(self, lower=0, upper=1, c=0.5,
                 *args, **kwargs):
        super(Triangular, self).__init__(*args, **kwargs)

        self.c = c
        self.lower = lower
        self.upper = upper
        self.mean = c
        self.median = self.mean

    def random(self, point=None, size=None):
        c, lower, upper = draw_values([self.c, self.lower, self.upper],
                                      point=point)
        return generate_samples(stats.triang.rvs, c=c-lower, loc=lower, scale=upper-lower,
                                size=size, dist_shape=self.shape, random_state=None)

    def logp(self, value):
        c = self.c
        lower = self.lower
        upper = self.upper
        return tt.switch(alltrue_elemwise([lower <= value, value < c]),
                         tt.log(2 * (value - lower) / ((upper - lower) * (c - lower))),
                         tt.switch(tt.eq(value, c), tt.log(2 / (upper - lower)),
                         tt.switch(alltrue_elemwise([c < value, value <= upper]),
                         tt.log(2 * (upper - value) / ((upper - lower) * (upper - c))),np.inf)))
