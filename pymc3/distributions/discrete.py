from functools import partial 
import numpy as np
import theano
import theano.tensor as tt
from scipy import stats

from .dist_math import bound, factln, binomln, betaln, logpow
from .distribution import Discrete, draw_values, generate_samples, reshape_sampled

__all__ = ['Binomial',  'BetaBinomial',  'Bernoulli',  'DiscreteWeibull',
           'Poisson', 'NegativeBinomial', 'ConstantDist', 'Constant',
           'ZeroInflatedPoisson', 'ZeroInflatedNegativeBinomial',
           'DiscreteUniform', 'Geometric', 'Categorical']


class Binomial(Discrete):
    R"""
    Binomial log-likelihood.

    The discrete probability distribution of the number of successes
    in a sequence of n independent yes/no experiments, each of which
    yields success with probability p.

    .. math:: f(x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}

    ========  ==========================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n p`
    Variance  :math:`n p (1 - p)`
    ========  ==========================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    p : float
        Probability of success in each trial (0 < p < 1).
    """

    def __init__(self, n, p, *args, **kwargs):
        super(Binomial, self).__init__(*args, **kwargs)
        self.n = n = tt.as_tensor_variable(n)
        self.p = p = tt.as_tensor_variable(p)
        self.mode = tt.cast(tt.round(n * p), self.dtype)

    def random(self, point=None, size=None, repeat=None):
        n, p = draw_values([self.n, self.p], point=point)
        return generate_samples(stats.binom.rvs, n=n, p=p,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        n = self.n
        p = self.p

        return bound(
            binomln(n, value) + logpow(p, value) + logpow(1 - p, n - value),
            0 <= value, value <= n,
            0 <= p, p <= 1)


class BetaBinomial(Discrete):
    R"""
    Beta-binomial log-likelihood.

    Equivalent to binomial random variable with success probability
    drawn from a beta distribution.

    .. math::

       f(x \mid \alpha, \beta, n) =
           \binom{n}{x}
           \frac{B(x + \alpha, n - x + \beta)}{B(\alpha, \beta)}

    ========  =================================================================
    Support   :math:`x \in \{0, 1, \ldots, n\}`
    Mean      :math:`n \dfrac{\alpha}{\alpha + \beta}`
    Variance  :math:`n \dfrac{\alpha \beta}{(\alpha+\beta)^2 (\alpha+\beta+1)}`
    ========  =================================================================

    Parameters
    ----------
    n : int
        Number of Bernoulli trials (n >= 0).
    alpha : float
        alpha > 0.
    beta : float
        beta > 0.
    """

    def __init__(self, alpha, beta, n, *args, **kwargs):
        super(BetaBinomial, self).__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.beta = beta = tt.as_tensor_variable(beta)
        self.n = n = tt.as_tensor_variable(n)
        self.mode = tt.cast(tt.round(alpha / (alpha + beta)), 'int8')

    def _random(self, alpha, beta, n, size=None):
        size = size or 1
        p = np.atleast_1d(stats.beta.rvs(a=alpha, b=beta, size=np.prod(size)))
        # Sometimes scipy.beta returns nan. Ugh.
        while np.any(np.isnan(p)):
            i = np.isnan(p)
            p[i] = stats.beta.rvs(a=alpha, b=beta, size=np.sum(i))
        # Sigh...
        _n, _p, _size = np.atleast_1d(n).flatten(), p.flatten(), np.prod(size)
        samples = np.reshape(stats.binom.rvs(n=_n, p=_p, size=_size), size)
        return samples

    def random(self, point=None, size=None, repeat=None):
        alpha, beta, n = \
            draw_values([self.alpha, self.beta, self.n], point=point)
        return generate_samples(self._random, alpha=alpha, beta=beta, n=n,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        return bound(binomln(self.n, value)
                     + betaln(value + alpha, self.n - value + beta)
                     - betaln(alpha, beta),
                     value >= 0, value <= self.n,
                     alpha > 0, beta > 0)


class Bernoulli(Discrete):
    R"""Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes
    (x=1) and failures (x=0).

    .. math:: f(x \mid p) = p^{x} (1-p)^{1-x}

    ========  ======================
    Support   :math:`x \in \{0, 1\}`
    Mean      :math:`p`
    Variance  :math:`p (1 - p)`
    ========  ======================

    Parameters
    ----------
    p : float
        Probability of success (0 < p < 1).
    """

    def __init__(self, p, *args, **kwargs):
        super(Bernoulli, self).__init__(*args, **kwargs)
        self.p = p = tt.as_tensor_variable(p)
        self.mode = tt.cast(tt.round(p), 'int8')

    def random(self, point=None, size=None, repeat=None):
        p = draw_values([self.p], point=point)
        return generate_samples(stats.bernoulli.rvs, p,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        p = self.p
        return bound(
            tt.switch(value, tt.log(p), tt.log(1 - p)),
            value >= 0, value <= 1,
            p >= 0, p <= 1)


class DiscreteWeibull(Discrete):
    R"""Discrete Weibull log-likelihood

    The discrete Weibull distribution is a flexible model of count data that
    can handle both over- and under-dispersion.

    .. math:: f(x \mid q, \beta) = q^{x^{\beta}} - q^{(x + 1)^{\beta}}

    ========  ======================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu = \sum_{x = 1}^{\infty} q^{x^{\beta}}`
    Variance  :math:`2 \sum_{x = 1}^{\infty} x q^{x^{\beta}} - \mu - \mu^2`
    ========  ======================
    """
    def __init__(self, q, beta, *args, **kwargs):
        super(DiscreteWeibull, self).__init__(*args, defaults=['median'], **kwargs)
        
        self.q = q
        self.beta = beta

        self.median = self._ppf(0.5)
    
    def logp(self, value):
        q = self.q
        beta = self.beta
        
        return bound(tt.log(tt.power(q, tt.power(value, beta)) - tt.power(q, tt.power(value + 1, beta))),
                     0 <= value,
                     0 < q, q < 1,
                     0 < beta)

    def _ppf(self, p):
        """
        The percentile point function (the inverse of the cumulative
        distribution function) of the discrete Weibull distribution.
        """
        q = self.q
        beta = self.beta

        return (tt.ceil(tt.power(tt.log(1 - p) / tt.log(q), 1. / beta)) - 1).astype('int64')

    def _random(self, q, beta, size=None):
        p = np.random.uniform(size=size)

        return np.ceil(np.power(np.log(1 - p) / np.log(q), 1. / beta)) - 1

    def random(self, point=None, size=None, repeat=None):
        q, beta = draw_values([self.q, self.beta], point=point)

        return generate_samples(self._random, q, beta,
                                dist_shape=self.shape,
                                size=size)


class Poisson(Discrete):
    R"""
    Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    Variance  :math:`\mu`
    ========  ==========================

    Parameters
    ----------
    mu : float
        Expected number of occurrences during the given interval
        (mu >= 0).

    Notes
    -----
    The Poisson distribution can be derived as a limiting case of the
    binomial distribution.
    """

    def __init__(self, mu, *args, **kwargs):
        super(Poisson, self).__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.mode = tt.floor(mu).astype('int32')

    def random(self, point=None, size=None, repeat=None):
        mu = draw_values([self.mu], point=point)
        return generate_samples(stats.poisson.rvs, mu,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        log_prob = bound(
            logpow(mu, value) - factln(value) - mu,
            mu >= 0, value >= 0)
        # Return zero when mu and value are both zero
        return tt.switch(1 * tt.eq(mu, 0) * tt.eq(value, 0),
                         0, log_prob)


class NegativeBinomial(Discrete):
    R"""
    Negative binomial log-likelihood.

    The negative binomial distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.

    .. math::

       f(x \mid \mu, \alpha) =
           \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)}
           (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu`
    ========  ==========================

    Parameters
    ----------
    mu : float
        Poission distribution parameter (mu > 0).
    alpha : float
        Gamma distribution parameter (alpha > 0).
    """

    def __init__(self, mu, alpha, *args, **kwargs):
        super(NegativeBinomial, self).__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.mode = tt.floor(mu).astype('int32')

    def random(self, point=None, size=None, repeat=None):
        mu, alpha = draw_values([self.mu, self.alpha], point=point)
        g = generate_samples(stats.gamma.rvs, alpha, scale=mu / alpha,
                             dist_shape=self.shape,
                             size=size)
        g[g == 0] = np.finfo(float).eps  # Just in case
        return reshape_sampled(stats.poisson.rvs(g), size, self.shape)

    def logp(self, value):
        mu = self.mu
        alpha = self.alpha
        negbinom = bound(binomln(value + alpha - 1, value)
                         + logpow(mu / (mu + alpha), value)
                         + logpow(alpha / (mu + alpha), alpha),
                         value >= 0, mu > 0, alpha > 0)

        # Return Poisson when alpha gets very large.
        return tt.switch(1 * (alpha > 1e10),
                         Poisson.dist(self.mu).logp(value),
                         negbinom)


class Geometric(Discrete):
    R"""
    Geometric log-likelihood.

    The probability that the first success in a sequence of Bernoulli
    trials occurs on the x'th trial.

    .. math:: f(x \mid p) = p(1-p)^{x-1}

    ========  =============================
    Support   :math:`x \in \mathbb{N}_{>0}`
    Mean      :math:`\dfrac{1}{p}`
    Variance  :math:`\dfrac{1 - p}{p^2}`
    ========  =============================

    Parameters
    ----------
    p : float
        Probability of success on an individual trial (0 < p <= 1).
    """

    def __init__(self, p, *args, **kwargs):
        super(Geometric, self).__init__(*args, **kwargs)
        self.p = p = tt.as_tensor_variable(p)
        self.mode = 1

    def random(self, point=None, size=None, repeat=None):
        p = draw_values([self.p], point=point)
        return generate_samples(np.random.geometric, p,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        p = self.p
        return bound(tt.log(p) + logpow(1 - p, value - 1),
                     0 <= p, p <= 1, value >= 1)


class DiscreteUniform(Discrete):
    R"""
    Discrete uniform distribution.

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower}

    ========  ===============================================
    Support   :math:`x \in {lower, lower + 1, \ldots, upper}`
    Mean      :math:`\dfrac{lower + upper}{2}`
    Variance  :math:`\dfrac{(upper - lower)^2}{12}`
    ========  ===============================================

    Parameters
    ----------
    lower : int
        Lower limit.
    upper : int
        Upper limit (upper > lower).
    """

    def __init__(self, lower, upper, *args, **kwargs):
        super(DiscreteUniform, self).__init__(*args, **kwargs)
        self.lower = tt.floor(lower).astype('int32')
        self.upper = tt.floor(upper).astype('int32')
        self.mode = tt.maximum(
            tt.floor((upper - lower) / 2.).astype('int32'), self.lower)

    def _random(self, lower, upper, size=None):
        # This way seems to be the only to deal with lower and upper
        # as array-like.
        samples = stats.uniform.rvs(lower, upper - lower - np.finfo(float).eps,
                                    size=size)
        return np.floor(samples).astype('int32')

    def random(self, point=None, size=None, repeat=None):
        lower, upper = draw_values([self.lower, self.upper], point=point)
        return generate_samples(self._random,
                                lower, upper,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        upper = self.upper
        lower = self.lower
        return bound(-tt.log(upper - lower + 1),
                     lower <= value, value <= upper)


class Categorical(Discrete):
    R"""
    Categorical log-likelihood.

    The most general discrete distribution.

    .. math:: f(x \mid p) = p_x

    ========  ===================================
    Support   :math:`x \in \{1, 2, \ldots, |p|\}`
    ========  ===================================

    Parameters
    ----------
    p : array of floats
        p > 0 and the elements of p must sum to 1. They will be automatically
        rescaled otherwise.
    """

    def __init__(self, p, *args, **kwargs):
        super(Categorical, self).__init__(*args, **kwargs)
        try:
            self.k = tt.shape(p)[-1].tag.test_value
        except AttributeError:
            self.k = tt.shape(p)[-1]
        self.p = p = tt.as_tensor_variable(p)
        self.p = (p.T / tt.sum(p, -1)).T
        self.mode = tt.argmax(p)

    def random(self, point=None, size=None, repeat=None):
        def random_choice(k, *args, **kwargs):
            if len(kwargs['p'].shape) > 1:
                return np.asarray(
                    [np.random.choice(k, p=p)
                     for p in kwargs['p']]
                )
            else:
                return np.random.choice(k, *args, **kwargs)

        p, k = draw_values([self.p, self.k], point=point)
        return generate_samples(partial(random_choice, np.arange(k)),
                                p=p,
                                broadcast_shape=p.shape[:-1] or (1,),
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        p = self.p
        k = self.k

        # Clip values before using them for indexing
        value_clip = tt.clip(value, 0, k - 1)

        sumto1 = theano.gradient.zero_grad(
            tt.le(abs(tt.sum(p, axis=-1) - 1), 1e-5))

        if p.ndim > 1:
            a = tt.log(p[tt.arange(p.shape[0]), value_clip])
        else:
            a = tt.log(p[value_clip])

        return bound(a, value >= 0, value <= (k - 1), sumto1)


class Constant(Discrete):
    """
    Constant log-likelihood.

    Parameters
    ----------
    value : float or int
        Constant parameter.
    """

    def __init__(self, c, *args, **kwargs):
        super(Constant, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.c = c = tt.as_tensor_variable(c)

    def random(self, point=None, size=None, repeat=None):
        c = draw_values([self.c], point=point)
        dtype = np.array(c).dtype

        def _random(c, dtype=dtype, size=None):
            return np.full(size, fill_value=c, dtype=dtype)

        return generate_samples(_random, c=c, dist_shape=self.shape,
                                size=size).astype(dtype)

    def logp(self, value):
        c = self.c
        return bound(0, tt.eq(value, c))


def ConstantDist(*args, **kwargs):
    import warnings
    warnings.warn("ConstantDist has been deprecated. In future, use Constant instead.",
                  DeprecationWarning)
    return Constant(*args, **kwargs)


class ZeroInflatedPoisson(Discrete):
    R"""
    Zero-inflated Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.

    .. math::

        f(x \mid \theta, \psi) = \left\{ \begin{array}{l}
            (1-\psi) + \psi e^{-\theta}, \text{if } x = 0 \\
            \psi \frac{e^{-\theta}\theta^x}{x!}, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\theta`
    Variance  :math:`\theta + \frac{1-\psi}{\psi}\theta^2`
    ========  ==========================

    Parameters
    ----------
    theta : float
        Expected number of occurrences during the given interval
        (theta >= 0).
    psi : float
        Expected proportion of Poisson variates (0 < psi < 1)

    """

    def __init__(self, theta, psi, *args, **kwargs):
        super(ZeroInflatedPoisson, self).__init__(*args, **kwargs)
        self.theta = theta = tt.as_tensor_variable(theta)
        self.psi = psi = tt.as_tensor_variable(psi)
        self.pois = Poisson.dist(theta)
        self.mode = self.pois.mode

    def random(self, point=None, size=None, repeat=None):
        theta, psi = draw_values([self.theta, self.psi], point=point)
        g = generate_samples(stats.poisson.rvs, theta,
                             dist_shape=self.shape,
                             size=size)
        sampled = g * (np.random.random(np.squeeze(g.shape)) < psi)
        return reshape_sampled(sampled, size, self.shape)

    def logp(self, value):
        return tt.switch(value > 0,
                         tt.log(self.psi) + self.pois.logp(value),
                         tt.log((1. - self.psi) + self.psi * tt.exp(-self.theta)))


class ZeroInflatedNegativeBinomial(Discrete):
    R"""
    Zero-Inflated Negative binomial log-likelihood.

    The Zero-inflated version of the Negative Binomial (NB).
    The NB distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.

    .. math::

       f(x \mid \mu, \alpha, \psi) = \left\{ \begin{array}{l}
            (1-\psi) + \psi \left (\frac{\alpha}{\alpha+\mu} \right) ^\alpha, \text{if } x = 0 \\
            \psi \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} \left (\frac{\alpha}{\mu+\alpha} \right)^\alpha \left( \frac{\mu}{\mu+\alpha} \right)^x, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\mu`
    Var       :math:`\psi\mu +  \left (1 + \frac{\mu}{\alpha} + \frac{1-\psi}{\mu} \right)`
    ========  ==========================

    Parameters
    ----------
    mu : float
        Poission distribution parameter (mu > 0).
    alpha : float
        Gamma distribution parameter (alpha > 0).
    psi : float
        Expected proportion of NegativeBinomial variates (0 < psi < 1)
    """

    def __init__(self, mu, alpha, psi, *args, **kwargs):
        super(ZeroInflatedNegativeBinomial, self).__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(mu)
        self.alpha = alpha = tt.as_tensor_variable(alpha)
        self.psi = psi = tt.as_tensor_variable(psi)
        self.nb = NegativeBinomial.dist(mu, alpha)
        self.mode = self.nb.mode

    def random(self, point=None, size=None, repeat=None):
        mu, alpha, psi = draw_values(
            [self.mu, self.alpha, self.psi], point=point)
        g = generate_samples(stats.gamma.rvs, alpha, scale=mu / alpha,
                             dist_shape=self.shape,
                             size=size)
        g[g == 0] = np.finfo(float).eps  # Just in case
        sampled = stats.poisson.rvs(g) * (np.random.random(np.squeeze(g.shape)) < psi)
        return reshape_sampled(sampled, size, self.shape)

    def logp(self, value):
        return tt.switch(value > 0,
                         tt.log(self.psi) + self.nb.logp(value),
                         tt.log((1. - self.psi) + self.psi * (self.alpha / (self.alpha + self.mu))**self.alpha))
