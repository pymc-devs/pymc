import numpy as np
import theano.tensor as tt
from scipy import stats
import warnings

from pymc3.util import get_variable_name
from .dist_math import bound, factln, binomln, betaln, logpow, random_choice
from .distribution import (Discrete, draw_values, generate_samples,
                           broadcast_distribution_samples)
from pymc3.math import tround, sigmoid, logaddexp, logit, log1pexp
from ..theanof import floatX, intX


__all__ = ['Binomial',  'BetaBinomial',  'Bernoulli',  'DiscreteWeibull',
           'Poisson', 'NegativeBinomial', 'ConstantDist', 'Constant',
           'ZeroInflatedPoisson', 'ZeroInflatedBinomial', 'ZeroInflatedNegativeBinomial',
           'DiscreteUniform', 'Geometric', 'Categorical', 'OrderedLogistic']


class Binomial(Discrete):
    R"""
    Binomial log-likelihood.

    The discrete probability distribution of the number of successes
    in a sequence of n independent yes/no experiments, each of which
    yields success with probability p.
    The pmf of this distribution is

    .. math:: f(x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(0, 22)
        ns = [10, 17]
        ps = [0.5, 0.7]
        for n, p in zip(ns, ps):
            pmf = st.binom.pmf(x, n, p)
            plt.plot(x, pmf, '-o', label='n = {}, p = {}'.format(n, p))
        plt.xlabel('x', fontsize=14)
        plt.ylabel('f(x)', fontsize=14)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.n = n = tt.as_tensor_variable(intX(n))
        self.p = p = tt.as_tensor_variable(floatX(p))
        self.mode = tt.cast(tround(n * p), self.dtype)

    def random(self, point=None, size=None):
        n, p = draw_values([self.n, self.p], point=point, size=size)
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

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        n = dist.n
        p = dist.p
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Binomial}}(\mathit{{n}}={},~\mathit{{p}}={})$'.format(name,
                                                get_variable_name(n),
                                                get_variable_name(p))

class BetaBinomial(Discrete):
    R"""
    Beta-binomial log-likelihood.

    Equivalent to binomial random variable with success probability
    drawn from a beta distribution.
    The pmf of this distribution is

    .. math::

       f(x \mid \alpha, \beta, n) =
           \binom{n}{x}
           \frac{B(x + \alpha, n - x + \beta)}{B(\alpha, \beta)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy import special
        plt.style.use('seaborn-darkgrid')

        def BetaBinom(a, b, n, x):
            pmf = special.binom(n, x) * (special.beta(x+a, n-x+b) / special.beta(a, b))
            return pmf

        x = np.arange(0, 11)
        alphas = [0.5, 1, 2.3]
        betas = [0.5, 1, 2]
        n = 10
        for a, b in zip(alphas, betas):
            pmf = BetaBinom(a, b, n, x)
            plt.plot(x, pmf, '-o', label=r'$\alpha$ = {}, $\beta$ = {}, n = {}'.format(a, b, n))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=9)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.beta = beta = tt.as_tensor_variable(floatX(beta))
        self.n = n = tt.as_tensor_variable(intX(n))
        self.mode = tt.cast(tround(alpha / (alpha + beta)), 'int8')

    def _random(self, alpha, beta, n, size=None):
        size = size or 1
        p = stats.beta.rvs(a=alpha, b=beta, size=size).flatten()
        # Sometimes scipy.beta returns nan. Ugh.
        while np.any(np.isnan(p)):
            i = np.isnan(p)
            p[i] = stats.beta.rvs(a=alpha, b=beta, size=np.sum(i))
        # Sigh...
        _n, _p, _size = np.atleast_1d(n).flatten(), p.flatten(), p.shape[0]

        quotient, remainder = divmod(_p.shape[0], _n.shape[0])
        if remainder != 0:
            raise TypeError('n has a bad size! Was cast to {}, must evenly divide {}'.format(
                _n.shape[0], _p.shape[0]))
        if quotient != 1:
            _n = np.tile(_n, quotient)
        samples = np.reshape(stats.binom.rvs(n=_n, p=_p, size=_size), size)
        return samples

    def random(self, point=None, size=None):
        alpha, beta, n = \
            draw_values([self.alpha, self.beta, self.n], point=point, size=size)
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

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        alpha = dist.alpha
        beta = dist.beta
        name = r'\text{%s}' % name
        return r'${} \sim \text{{BetaBinomial}}(\mathit{{alpha}}={},~\mathit{{beta}}={})$'.format(name,
                                                get_variable_name(alpha),
                                                get_variable_name(beta))


class Bernoulli(Discrete):
    R"""Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes
    (x=1) and failures (x=0).
    The pmf of this distribution is

    .. math:: f(x \mid p) = p^{x} (1-p)^{1-x}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = [0, 1]
        for p in [0, 0.5, 0.8]:
            pmf = st.bernoulli.pmf(x, p)
            plt.plot(x, pmf, '-o', label='p = {}'.format(p))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=9)
        plt.show()

    ========  ======================
    Support   :math:`x \in \{0, 1\}`
    Mean      :math:`p`
    Variance  :math:`p (1 - p)`
    ========  ======================

    Parameters
    ----------
    p : float
        Probability of success (0 < p < 1).
    logit_p : float
        Logit of success probability. Only one of `p` and `logit_p`
        can be specified.
    """

    def __init__(self, p=None, logit_p=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sum(int(var is None) for var in [p, logit_p]) != 1:
            raise ValueError('Specify one of p and logit_p')
        if p is not None:
            self._is_logit = False
            self.p = p = tt.as_tensor_variable(floatX(p))
            self._logit_p = logit(p)
        else:
            self._is_logit = True
            self.p = tt.nnet.sigmoid(floatX(logit_p))
            self._logit_p = tt.as_tensor_variable(logit_p)

        self.mode = tt.cast(tround(self.p), 'int8')

    def random(self, point=None, size=None):
        p = draw_values([self.p], point=point, size=size)[0]
        return generate_samples(stats.bernoulli.rvs, p,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        if self._is_logit:
            lp = tt.switch(value, self._logit_p, -self._logit_p)
            return -log1pexp(-lp)
        else:
            p = self.p
            return bound(
                tt.switch(value, tt.log(p), tt.log(1 - p)),
                value >= 0, value <= 1,
                p >= 0, p <= 1)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        p = dist.p
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Bernoulli}}(\mathit{{p}}={})$'.format(name,
                                                get_variable_name(p))


class DiscreteWeibull(Discrete):
    R"""Discrete Weibull log-likelihood

    The discrete Weibull distribution is a flexible model of count data that
    can handle both over- and under-dispersion.
    The pmf of this distribution is

    .. math:: f(x \mid q, \beta) = q^{x^{\beta}} - q^{(x + 1)^{\beta}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy import special
        plt.style.use('seaborn-darkgrid')

        def DiscreteWeibull(q, b, x):
            return q**(x**b) - q**((x + 1)**b)

        x = np.arange(0, 10)
        qs = [0.1, 0.9, 0.9]
        betas = [0.3, 1.3, 3]
        for q, b in zip(qs, betas):
            pmf = DiscreteWeibull(q, b, x)
            plt.plot(x, pmf, '-o', label=r'q = {}, $\beta$ = {}'.format(q, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=1)
        plt.show()

    ========  ======================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\mu = \sum_{x = 1}^{\infty} q^{x^{\beta}}`
    Variance  :math:`2 \sum_{x = 1}^{\infty} x q^{x^{\beta}} - \mu - \mu^2`
    ========  ======================
    """
    def __init__(self, q, beta, *args, **kwargs):
        super().__init__(*args, defaults=('median',), **kwargs)

        self.q = q = tt.as_tensor_variable(floatX(q))
        self.beta = beta = tt.as_tensor_variable(floatX(beta))

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

    def random(self, point=None, size=None):
        q, beta = draw_values([self.q, self.beta], point=point, size=size)
        q, beta = broadcast_distribution_samples([q, beta], size=size)

        return generate_samples(self._random, q, beta,
                                dist_shape=self.shape,
                                size=size)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        q = dist.q
        beta = dist.beta
        name = r'\text{%s}' % name
        return r'${} \sim \text{{DiscreteWeibull}}(\mathit{{q}}={},~\mathit{{beta}}={})$'.format(name,
                                                get_variable_name(q),
                                                get_variable_name(beta))


class Poisson(Discrete):
    R"""
    Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(0, 15)
        for m in [0.5, 3, 8]:
            pmf = st.poisson.pmf(x, m)
            plt.plot(x, pmf, '-o', label='$\mu$ = {}'.format(m))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.mode = intX(tt.floor(mu))

    def random(self, point=None, size=None):
        mu = draw_values([self.mu], point=point, size=size)[0]
        return generate_samples(stats.poisson.rvs, mu,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        mu = self.mu
        log_prob = bound(
            logpow(mu, value) - factln(value) - mu,
            mu >= 0, value >= 0)
        # Return zero when mu and value are both zero
        return tt.switch(tt.eq(mu, 0) * tt.eq(value, 0),
                         0, log_prob)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Poisson}}(\mathit{{mu}}={})$'.format(name,
                                                get_variable_name(mu))


class NegativeBinomial(Discrete):
    R"""
    Negative binomial log-likelihood.

    The negative binomial distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.
    The pmf of this distribution is

    .. math::

       f(x \mid \mu, \alpha) =
           \binom{x + \alpha - 1}{x}
           (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy import special
        plt.style.use('seaborn-darkgrid')

        def NegBinom(a, m, x):
            pmf = special.binom(x + a - 1, x) * (a / (m + a))**a * (m / (m + a))**x
            return pmf

        x = np.arange(0, 22)
        alphas = [0.9, 2, 4]
        mus = [1, 2, 8]
        for a, m in zip(alphas, mus):
            pmf = NegBinom(a, m, x)
            plt.plot(x, pmf, '-o', label=r'$\alpha$ = {}, $\mu$ = {}'.format(a, m))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.mode = intX(tt.floor(mu))

    def random(self, point=None, size=None):
        mu, alpha = draw_values([self.mu, self.alpha], point=point, size=size)
        g = generate_samples(stats.gamma.rvs, alpha, scale=mu / alpha,
                             dist_shape=self.shape,
                             size=size)
        g[g == 0] = np.finfo(float).eps  # Just in case
        return np.asarray(stats.poisson.rvs(g)).reshape(g.shape)

    def logp(self, value):
        mu = self.mu
        alpha = self.alpha
        negbinom = bound(binomln(value + alpha - 1, value)
                         + logpow(mu / (mu + alpha), value)
                         + logpow(alpha / (mu + alpha), alpha),
                         value >= 0, mu > 0, alpha > 0)

        # Return Poisson when alpha gets very large.
        return tt.switch(tt.gt(alpha, 1e10),
                         Poisson.dist(self.mu).logp(value),
                         negbinom)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        alpha = dist.alpha
        name = r'\text{%s}' % name
        return r'${} \sim \text{{NegativeBinomial}}(\mathit{{mu}}={},~\mathit{{alpha}}={})$'.format(name,
                                                get_variable_name(mu),
                                                get_variable_name(alpha))


class Geometric(Discrete):
    R"""
    Geometric log-likelihood.

    The probability that the first success in a sequence of Bernoulli
    trials occurs on the x'th trial.
    The pmf of this distribution is

    .. math:: f(x \mid p) = p(1-p)^{x-1}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(1, 11)
        for p in [0.1, 0.25, 0.75]:
            pmf = st.geom.pmf(x, p)
            plt.plot(x, pmf, '-o', label='p = {}'.format(p))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.p = p = tt.as_tensor_variable(floatX(p))
        self.mode = 1

    def random(self, point=None, size=None):
        p = draw_values([self.p], point=point, size=size)[0]
        return generate_samples(np.random.geometric, p,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        p = self.p
        return bound(tt.log(p) + logpow(1 - p, value - 1),
                     0 <= p, p <= 1, value >= 1)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        p = dist.p
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Geometric}}(\mathit{{p}}={})$'.format(name,
                                                get_variable_name(p))


class DiscreteUniform(Discrete):
    R"""
    Discrete uniform distribution.
    The pmf of this distribution is

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower+1}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        ls = [1, -2]
        us = [6, 2]
        for l, u in zip(ls, us):
            x = np.arange(l, u+1)
            pmf = [1.0 / (u - l + 1)] * len(x)
            plt.plot(x, pmf, '-o', label='lower = {}, upper = {}'.format(l, u))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 0.4)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.lower = intX(tt.floor(lower))
        self.upper = intX(tt.floor(upper))
        self.mode = tt.maximum(
            intX(tt.floor((upper + lower) / 2.)), self.lower)

    def _random(self, lower, upper, size=None):
        # This way seems to be the only to deal with lower and upper
        # as array-like.
        samples = stats.randint.rvs(lower, upper + 1, size=size)
        return samples

    def random(self, point=None, size=None):
        lower, upper = draw_values([self.lower, self.upper], point=point, size=size)
        return generate_samples(self._random,
                                lower, upper,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        upper = self.upper
        lower = self.lower
        return bound(-tt.log(upper - lower + 1),
                     lower <= value, value <= upper)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        lower = dist.lower
        upper = dist.upper
        name = r'\text{%s}' % name
        return r'${} \sim \text{{DiscreteUniform}}(\mathit{{lower}}={},~\mathit{{upper}}={})$'.format(name,
                                                get_variable_name(lower),
                                                get_variable_name(upper))


class Categorical(Discrete):
    R"""
    Categorical log-likelihood.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        ps = [[0.1, 0.6, 0.3], [0.3, 0.1, 0.1, 0.5]]
        for p in ps:
            x = range(len(p))
            plt.plot(x, p, '-o', label='p = {}'.format(p))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0)
        plt.legend(loc=1)
        plt.show()

    ========  ===================================
    Support   :math:`x \in \{0, 1, \ldots, |p|-1\}`
    ========  ===================================

    Parameters
    ----------
    p : array of floats
        p > 0 and the elements of p must sum to 1. They will be automatically
        rescaled otherwise.
    """

    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.k = tt.shape(p)[-1].tag.test_value
        except AttributeError:
            self.k = tt.shape(p)[-1]
        p = tt.as_tensor_variable(floatX(p))

        # From #2082, it may be dangerous to automatically rescale p at this
        # point without checking for positiveness
        self.p = p
        self.mode = tt.argmax(p, axis=-1)
        if self.mode.ndim == 1:
            self.mode = tt.squeeze(self.mode)

    def random(self, point=None, size=None):
        p, k = draw_values([self.p, self.k], point=point, size=size)
        p = p / np.sum(p, axis=-1, keepdims=True)

        return generate_samples(random_choice,
                                p=p,
                                broadcast_shape=p.shape[:-1] or (1,),
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        p_ = self.p
        k = self.k

        # Clip values before using them for indexing
        value_clip = tt.clip(value, 0, k - 1)

        p = p_ / tt.sum(p_, axis=-1, keepdims=True)

        if p.ndim > 1:
            pattern = (p.ndim - 1,) + tuple(range(p.ndim - 1))
            a = tt.log(p.dimshuffle(pattern)[value_clip])
        else:
            a = tt.log(p[value_clip])

        return bound(a, value >= 0, value <= (k - 1),
                     tt.all(p_ >= 0, axis=-1), tt.all(p <= 1, axis=-1))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        p = dist.p
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Categorical}}(\mathit{{p}}={})$'.format(name,
                                                get_variable_name(p))


class Constant(Discrete):
    """
    Constant log-likelihood.

    Parameters
    ----------
    value : float or int
        Constant parameter.
    """

    def __init__(self, c, *args, **kwargs):
        warnings.warn("Constant has been deprecated. We recommend using a Deterministic object instead.",
                    DeprecationWarning)
        super().__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.c = c = tt.as_tensor_variable(c)

    def random(self, point=None, size=None):
        c = draw_values([self.c], point=point, size=size)[0]
        dtype = np.array(c).dtype

        def _random(c, dtype=dtype, size=None):
            return np.full(size, fill_value=c, dtype=dtype)

        return generate_samples(_random, c=c, dist_shape=self.shape,
                                size=size).astype(dtype)

    def logp(self, value):
        c = self.c
        return bound(0, tt.eq(value, c))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Constant}}()$'.format(name)


ConstantDist = Constant


class ZeroInflatedPoisson(Discrete):
    R"""
    Zero-inflated Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \theta) = \left\{ \begin{array}{l}
            (1-\psi) + \psi e^{-\theta}, \text{if } x = 0 \\
            \psi \frac{e^{-\theta}\theta^x}{x!}, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(0, 22)
        psis = [0.7, 0.4]
        thetas = [8, 4]
        for psi, theta in zip(psis, thetas):
            pmf = st.poisson.pmf(x, theta)
            pmf[0] = (1 - psi) + pmf[0]
            pmf[1:] =  psi * pmf[1:]
            pmf /= pmf.sum()
            plt.plot(x, pmf, '-o', label='$\\psi$ = {}, $\\theta$ = {}'.format(psi, theta))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\theta`
    Variance  :math:`\theta + \frac{1-\psi}{\psi}\theta^2`
    ========  ==========================

    Parameters
    ----------
    psi : float
        Expected proportion of Poisson variates (0 < psi < 1)
    theta : float
        Expected number of occurrences during the given interval
        (theta >= 0).
    """

    def __init__(self, psi, theta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = theta = tt.as_tensor_variable(floatX(theta))
        self.psi = psi = tt.as_tensor_variable(floatX(psi))
        self.pois = Poisson.dist(theta)
        self.mode = self.pois.mode

    def random(self, point=None, size=None):
        theta, psi = draw_values([self.theta, self.psi], point=point, size=size)
        g = generate_samples(stats.poisson.rvs, theta,
                             dist_shape=self.shape,
                             size=size)
        g, psi = broadcast_distribution_samples([g, psi], size=size)
        return g * (np.random.random(g.shape) < psi)

    def logp(self, value):
        psi = self.psi
        theta = self.theta

        logp_val = tt.switch(
            tt.gt(value, 0),
            tt.log(psi) + self.pois.logp(value),
            logaddexp(tt.log1p(-psi), tt.log(psi) - theta))

        return bound(
            logp_val,
            0 <= value,
            0 <= psi, psi <= 1,
            0 <= theta)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        theta = dist.theta
        psi = dist.psi
        name = r'\text{%s}' % name
        return r'${} \sim \text{{ZeroInflatedPoisson}}(\mathit{{theta}}={},~\mathit{{psi}}={})$'.format(name,
                                                get_variable_name(theta),
                                                get_variable_name(psi))


class ZeroInflatedBinomial(Discrete):
    R"""
    Zero-inflated Binomial log-likelihood.

    The pmf of this distribution is

    .. math::

        f(x \mid \psi, n, p) = \left\{ \begin{array}{l}
            (1-\psi) + \psi (1-p)^{n}, \text{if } x = 0 \\
            \psi {n \choose x} p^x (1-p)^{n-x}, \text{if } x=1,2,3,\ldots,n
            \end{array} \right.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.arange(0, 25)
        ns = [10, 20]
        ps = [0.5, 0.7]
        psis = [0.7, 0.4]
        for n, p, psi in zip(ns, ps, psis):
            pmf = st.binom.pmf(x, n, p)
            pmf[0] = (1 - psi) + pmf[0]
            pmf[1:] =  psi * pmf[1:]
            pmf /= pmf.sum()
            plt.plot(x, pmf, '-o', label='n = {}, p = {}, $\\psi$ = {}'.format(n, p, psi))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`(1 - \psi) n p`
    Variance  :math:`(1-\psi) n p [1 - p(1 - \psi n)].`
    ========  ==========================

    Parameters
    ----------
    psi : float
        Expected proportion of Binomial variates (0 < psi < 1)
    n : int
        Number of Bernoulli trials (n >= 0).
    p : float
        Probability of success in each trial (0 < p < 1).

    """

    def __init__(self, psi, n, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n = n = tt.as_tensor_variable(intX(n))
        self.p = p = tt.as_tensor_variable(floatX(p))
        self.psi = psi = tt.as_tensor_variable(floatX(psi))
        self.bin = Binomial.dist(n, p)
        self.mode = self.bin.mode

    def random(self, point=None, size=None):
        n, p, psi = draw_values([self.n, self.p, self.psi], point=point, size=size)
        g = generate_samples(stats.binom.rvs, n, p,
                             dist_shape=self.shape,
                             size=size)
        g, psi = broadcast_distribution_samples([g, psi], size=size)
        return g * (np.random.random(g.shape) < psi)

    def logp(self, value):
        psi = self.psi
        p = self.p
        n = self.n

        logp_val = tt.switch(
            tt.gt(value, 0),
            tt.log(psi) + self.bin.logp(value),
            logaddexp(tt.log1p(-psi), tt.log(psi) + n * tt.log1p(-p)))

        return bound(
            logp_val,
            0 <= value, value <= n,
            0 <= psi, psi <= 1,
            0 <= p, p <= 1)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        n = dist.n
        p = dist.p
        psi = dist.psi

        name_n = get_variable_name(n)
        name_p = get_variable_name(p)
        name_psi = get_variable_name(psi)
        name = r'\text{%s}' % name
        return (r'${} \sim \text{{ZeroInflatedBinomial}}'
                r'(\mathit{{n}}={},~\mathit{{p}}={},~'
                r'\mathit{{psi}}={})$'
                .format(name, name_n, name_p, name_psi))


class ZeroInflatedNegativeBinomial(Discrete):
    R"""
    Zero-Inflated Negative binomial log-likelihood.

    The Zero-inflated version of the Negative Binomial (NB).
    The NB distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.
    The pmf of this distribution is

    .. math::

       f(x \mid \psi, \mu, \alpha) = \left\{
         \begin{array}{l}
           (1-\psi) + \psi \left (
             \frac{\alpha}{\alpha+\mu}
           \right) ^\alpha, \text{if } x = 0 \\
           \psi \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} \left (
             \frac{\alpha}{\mu+\alpha}
           \right)^\alpha \left(
             \frac{\mu}{\mu+\alpha}
           \right)^x, \text{if } x=1,2,3,\ldots
         \end{array}
       \right.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy import special
        plt.style.use('seaborn-darkgrid')

        def ZeroInfNegBinom(a, m, psi, x):
            pmf = special.binom(x + a - 1, x) * (a / (m + a))**a * (m / (m + a))**x
            pmf[0] = (1 - psi) + pmf[0]
            pmf[1:] =  psi * pmf[1:]
            pmf /= pmf.sum()
            return pmf

        x = np.arange(0, 25)
        alphas = [2, 4]
        mus = [2, 8]
        psis = [0.7, 0.7]
        for a, m, psi in zip(alphas, mus, psis):
            pmf = ZeroInfNegBinom(a, m, psi, x)
            plt.plot(x, pmf, '-o', label=r'$\alpha$ = {}, $\mu$ = {}, $\psi$ = {}'.format(a, m, psi))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\mu`
    Var       :math:`\psi\mu +  \left (1 + \frac{\mu}{\alpha} + \frac{1-\psi}{\mu} \right)`
    ========  ==========================

    Parameters
    ----------
    psi : float
        Expected proportion of NegativeBinomial variates (0 < psi < 1)
    mu : float
        Poission distribution parameter (mu > 0).
    alpha : float
        Gamma distribution parameter (alpha > 0).

    """

    def __init__(self, psi, mu, alpha, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.psi = psi = tt.as_tensor_variable(floatX(psi))
        self.nb = NegativeBinomial.dist(mu, alpha)
        self.mode = self.nb.mode

    def random(self, point=None, size=None):
        mu, alpha, psi = draw_values(
            [self.mu, self.alpha, self.psi], point=point, size=size)
        g = generate_samples(stats.gamma.rvs, alpha, scale=mu / alpha,
                             dist_shape=self.shape,
                             size=size)
        g[g == 0] = np.finfo(float).eps  # Just in case
        g, psi = broadcast_distribution_samples([g, psi], size=size)
        return stats.poisson.rvs(g) * (np.random.random(g.shape) < psi)

    def logp(self, value):
        alpha = self.alpha
        mu = self.mu
        psi = self.psi

        logp_other = tt.log(psi) + self.nb.logp(value)
        logp_0 = logaddexp(
            tt.log1p(-psi),
            tt.log(psi) + alpha * (tt.log(alpha) - tt.log(alpha + mu)))

        logp_val = tt.switch(
            tt.gt(value, 0),
            logp_other,
            logp_0)

        return bound(
            logp_val,
            0 <= value,
            0 <= psi, psi <= 1,
            mu > 0, alpha > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        alpha = dist.alpha
        psi = dist.psi

        name_mu = get_variable_name(mu)
        name_alpha = get_variable_name(alpha)
        name_psi = get_variable_name(psi)
        name = r'\text{%s}' % name
        return (r'${} \sim \text{{ZeroInflatedNegativeBinomial}}'
                r'(\mathit{{mu}}={},~\mathit{{alpha}}={},~'
                r'\mathit{{psi}}={})$'
                .format(name, name_mu, name_alpha, name_psi))


class OrderedLogistic(Categorical):
    R"""
    Ordered Logistic log-likelihood.

    Useful for regression on ordinal data values whose values range
    from 1 to K as a function of some predictor, :math:`\eta`. The
    cutpoints, :math:`c`, separate which ranges of :math:`\eta` are
    mapped to which of the K observed dependent variables.  The number
    of cutpoints is K - 1.  It is recommended that the cutpoints are
    constrained to be ordered.

    .. math::

       f(k \mid \eta, c) = \left\{
         \begin{array}{l}
           1 - \text{logit}^{-1}(\eta - c_1)
             \,, \text{if } k = 0 \\
           \text{logit}^{-1}(\eta - c_{k - 1}) -
           \text{logit}^{-1}(\eta - c_{k})
             \,, \text{if } 0 < k < K \\
           \text{logit}^{-1}(\eta - c_{K - 1})
             \,, \text{if } k = K \\
         \end{array}
       \right.

    Parameters
    ----------
    eta : float
        The predictor.
    c : array
        The length K - 1 array of cutpoints which break :math:`\eta` into
        ranges.  Do not explicitly set the first and last elements of
        :math:`c` to negative and positive infinity.

    Example
    --------
    .. code:: python

        # Generate data for a simple 1 dimensional example problem
        n1_c = 300; n2_c = 300; n3_c = 300
        cluster1 = np.random.randn(n1_c) + -1
        cluster2 = np.random.randn(n2_c) + 0
        cluster3 = np.random.randn(n3_c) + 2

        x = np.concatenate((cluster1, cluster2, cluster3))
        y = np.concatenate((1*np.ones(n1_c),
                            2*np.ones(n2_c),
                            3*np.ones(n3_c))) - 1

        # Ordered logistic regression
        with pm.Model() as model:
            cutpoints = pm.Normal("cutpoints", mu=[-1,1], sigma=10, shape=2,
                                  transform=pm.distributions.transforms.ordered)
            y_ = pm.OrderedLogistic("y", cutpoints=cutpoints, eta=x, observed=y)
            tr = pm.sample(1000)

        # Plot the results
        plt.hist(cluster1, 30, alpha=0.5);
        plt.hist(cluster2, 30, alpha=0.5);
        plt.hist(cluster3, 30, alpha=0.5);
        plt.hist(tr["cutpoints"][:,0], 80, alpha=0.2, color='k');
        plt.hist(tr["cutpoints"][:,1], 80, alpha=0.2, color='k');

    """

    def __init__(self, eta, cutpoints, *args, **kwargs):
        self.eta = tt.as_tensor_variable(floatX(eta))
        self.cutpoints = tt.as_tensor_variable(cutpoints)

        pa = sigmoid(tt.shape_padleft(self.cutpoints) - tt.shape_padright(self.eta))
        p_cum = tt.concatenate([
            tt.zeros_like(tt.shape_padright(pa[:, 0])),
            pa,
            tt.ones_like(tt.shape_padright(pa[:, 0]))
        ], axis=-1)
        p = p_cum[:, 1:] - p_cum[:, :-1]

        super().__init__(p=p, *args, **kwargs)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name_eta = get_variable_name(dist.eta)
        name_cutpoints = get_variable_name(dist.cutpoints)
        return (r'${} \sim \text{{OrderedLogistic}}'
                r'(\mathit{{eta}}={}, \mathit{{cutpoints}}={}$'
                .format(name, name_eta, name_cutpoints))
