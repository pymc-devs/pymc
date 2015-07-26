from .dist_math import *
from theano.tensor import as_tensor_variable
from .distribution import draw_values
import theano
import scipy.stats as st
import numpy as np
import numpy.random as nr

__all__ = ['Binomial',  'BetaBin',  'Bernoulli',  'Poisson', 'NegativeBinomial',
'ConstantDist', 'ZeroInflatedPoisson', 'DiscreteUniform', 'Geometric',
'Categorical']


class Binomial(Discrete):
    """
    Binomial log-likelihood.  The discrete probability distribution
    of the number of successes in a sequence of n independent yes/no
    experiments, each of which yields success with probability p.

    .. math::
        f(x \mid n, p) = \frac{n!}{x!(n-x)!} p^x (1-p)^{n-x}

    Parameters
    ----------
    n : int
        Number of Bernoulli trials, n > x
    p : float
        Probability of success in each trial, :math:`p \in [0,1]`

    .. note::
    - :math:`E(X)=np`
    - :math:`Var(X)=np(1-p)`

    """
    def __init__(self, n, p, *args, **kwargs):
        super(Binomial, self).__init__(*args, **kwargs)
        self.n = n
        self.p = p
        self.mode = cast(round(n * p), self.dtype)

    def random(self, point=None, size=None):
        n, p = draw_values([self.n, self.p], point=point)
        return st.binom.rvs(n=n, p=p, size=size)

    def logp(self, value):
        n = self.n
        p = self.p

        return bound(

            logpow(p, value) + logpow(
                1 - p, n - value) + factln(
                    n) - factln(value) - factln(n - value),

            0 <= value, value <= n,
            0 <= p, p <= 1)


class BetaBin(Discrete):
    """
    Beta-binomial log-likelihood. Equivalent to binomial random
    variables with probabilities drawn from a
    :math:`\texttt{Beta}(\alpha,\beta)` distribution.

    .. math::
        f(x \mid \alpha, \beta, n) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)} \frac{\Gamma(n+1)}{\Gamma(x+1)\Gamma(n-x+1)} \frac{\Gamma(\alpha + x)\Gamma(n+\beta-x)}{\Gamma(\alpha+\beta+n)}

    Parameters
    ----------
    alpha : float
        alpha > 0
    beta : float
        beta > 0
    n : int
        n=x,x+1,...

    .. note::
    - :math:`E(X)=n\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=n\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    def __init__(self, alpha, beta, n, *args, **kwargs):
        super(BetaBin, self).__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.mode = cast(round(alpha / (alpha + beta)), 'int8')

    def random(self, point=None, size=None):
        alpha, beta, n = \
            draw_values([self.alpha, self.beta, self.n], point=point)
        return st.binom.rvs(n, st.beta.rvs(a=alpha, b=beta, size=size))

    def logp(self, value):
        alpha = self.alpha
        beta = self.beta
        n = self.n

        return bound(gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) +
                     gammaln(n + 1) - gammaln(value + 1) - gammaln(n - value + 1) +
                     gammaln(alpha + value) + gammaln(n + beta - value) - gammaln(beta + alpha + n),

                     0 <= value, value <= n,
                     alpha > 0,
                     beta > 0)


class Bernoulli(Discrete):
    """Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes (x=1) and
    failures (x=0).

    .. math::  f(x \mid p) = p^{x} (1-p)^{1-x}

    Parameters
    ----------
    p : float
        Probability of success. :math:`0 < p < 1`.

    .. note::
    - :math:`E(x)= p`
    - :math:`Var(x)= p(1-p)`

    """
    def __init__(self, p, *args, **kwargs):
        super(Bernoulli, self).__init__(*args, **kwargs)
        self.p = p
        self.mode = cast(round(p), 'int8')

    def random(self, point=None, size=None):
        p = draw_values([self.p], point=point)
        return st.bernoulli.rvs(p, size=size)

    def logp(self, value):
        p = self.p
        return bound(
            switch(value, log(p), log(1 - p)),
            value >= 0, value <= 1,
            0 <= p, p <= 1)


class Poisson(Discrete):
    """
    Poisson log-likelihood.

    The Poisson is a discrete probability
    distribution.  It is often used to model the number of events
    occurring in a fixed period of time when the times at which events
    occur are independent. The Poisson distribution can be derived as
    a limiting case of the binomial distribution.

    .. math::
        f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    Parameters
    ----------
    mu : float
        Expected number of occurrences during the given interval, :math:`\mu \geq 0`.

    .. note::
       - :math:`E(x)=\mu`
       - :math:`Var(x)=\mu`

    """
    def __init__(self, mu, *args, **kwargs):
        super(Poisson, self).__init__(*args, **kwargs)
        self.mu = mu
        self.mode = floor(mu).astype('int32')

    def random(self, point=None, size=None):
        mu = draw_values([self.mu], point=point)
        return st.poisson.rvs(mu, size=size)

    def logp(self, value):
        mu = self.mu

        return bound(
            logpow(mu, value) - factln(value) - mu,

            mu >= 0, value >= 0)


class NegativeBinomial(Discrete):
    """
    Negative binomial log-likelihood.

     The negative binomial distribution  describes a Poisson random variable
     whose rate parameter is gamma distributed. PyMC's chosen parameterization
     is based on this mixture interpretation.

    .. math::
        f(x \mid \mu, \alpha) = \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x

    Parameters
    ----------
      mu : float
          mu > 0
      alpha : float
          alpha > 0

    .. note::
      - :math:`E[x]=\mu`

    """
    def __init__(self, mu, alpha, *args, **kwargs):
        super(NegativeBinomial, self).__init__(*args, **kwargs)
        self.mu = mu
        self.alpha = alpha
        self.mode = floor(mu).astype('int32')

    def random(self, point=None, size=None):
        mu, alpha = draw_values([self.mu, self.alpha], point=point)
        if alpha > 1e10:
            return st.poisson.rvs(mu, size=size)
        else:
            g = st.gamma.rvs(alpha, scale=alpha / mu, size=size)
            g[g==0] = np.finfo(float).eps
            return st.poisson.rvs(g)

    def logp(self, value):
        mu = self.mu
        alpha = self.alpha

        # Return Poisson when alpha gets very large
        pois = bound(logpow(mu, value) - factln(value) - mu,
                     mu > 0,
                     value >= 0)
        negbinom = bound(gammaln(value + alpha) - factln(value) - gammaln(alpha) +
                     logpow(mu / (mu + alpha), value) + logpow(alpha / (mu + alpha), alpha),
                      mu > 0, alpha > 0, value >= 0)

        return switch(1*(alpha > 1e10),
                      pois,
                      negbinom)


class Geometric(Discrete):
    """
    Geometric log-likelihood. The probability that the first success in a
    sequence of Bernoulli trials occurs on the x'th trial.

    .. math::
        f(x \mid p) = p(1-p)^{x-1}

    Parameters
    ----------
    p : float
        Probability of success on an individual trial, :math:`p \in [0,1]`

    .. note::
      - :math:`E(X)=1/p`
      - :math:`Var(X)=\frac{1-p}{p^2}`

    """
    def __init__(self, p, *args, **kwargs):
        super(Geometric, self).__init__(*args, **kwargs)
        self.p = p
        self.mode = 1

    def random(self, point=None, size=None):
        p = draw_values([self.p], point=point)
        return nr.geometric(p, size=size)

    def logp(self, value):
        p = self.p
        return bound(log(p) + logpow(1 - p, value - 1),
                     0 <= p, p <= 1, value >= 1)


class DiscreteUniform(Discrete):
    """
    Discrete uniform distribution.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    Parameters
    ----------
    lower : int
        Lower limit.
    upper : int
        Upper limit (upper > lower).

    """
    def __init__(self, lower, upper, *args, **kwargs):
        super(DiscreteUniform, self).__init__(*args, **kwargs)
        self.lower, self.upper = floor(lower).astype('int32'), floor(upper).astype('int32')
        self.mode = floor((upper - lower) / 2.).astype('int32')

    def random(self, point=None, size=None):
        lower, upper = draw_values([self.lower, self.upper], point=point)
        return nr.random_integers(lower, upper, size=size)

    def logp(self, value):
        upper = self.upper
        lower = self.lower

        return bound(
            -log(upper - lower + 1),

            lower <= value, value <= upper)


class Categorical(Discrete):
    """
    Categorical log-likelihood. The most general discrete distribution.

    .. math::  f(x=i \mid p) = p_i

    for :math:`i \in 0 \ldots k-1`.

    Parameters
    ----------
    p : float
        :math:`p > 0`, :math:`\sum p = 1`

    """
    def __init__(self, p, *args, **kwargs):
        super(Categorical, self).__init__(*args, **kwargs)
        self.k = p.shape[0]
        self.p = as_tensor_variable(p)
        self.mode = argmax(p)

    def random(self, point=None, size=None):
        p = draw_values([self.p], point=point)
        return nr.multinomial(1, p, size=size)

    def logp(self, value):
        p = self.p
        k = self.k

        sumto1 = theano.gradient.zero_grad(
            le(abs(sum(p) - 1), 1e-5)
            )
        return bound(log(p[value]),
            value >= 0,
            value <= (k - 1),
            sumto1)


class ConstantDist(Discrete):
    """
    Constant log-likelihood with parameter c={0}.

    Parameters
    ----------
    value : float or int
        Data value(s)
    """

    def __init__(self, c, *args, **kwargs):
        super(ConstantDist, self).__init__(*args, **kwargs)
        self.mean = self.median = self.mode = self.c = c

    def random(self, point=None, size=None):
        c = draw_values([self.c], point)
        return np.ones(size).astype(int) * c

    def logp(self, value):
        c = self.c
        return bound(0, eq(value, c))


class ZeroInflatedPoisson(Discrete):
    def __init__(self, theta, z, *args, **kwargs):
        super(ZeroInflatedPoisson, self).__init__(*args, **kwargs)
        self.theta = theta
        self.z = z
        self.pois = Poisson.dist(theta)
        self.const = ConstantDist.dist(0)
        self.mode = self.pois.mode

    def random(self, point=None, size=None):
        theta, z = draw_values([self.theta, self.z], point=point)
        #u = nr.uniform(size=size)
        # To do: Finish me
        return None

    def logp(self, value):
        z = self.z
        return switch(z,
                      self.pois.logp(value),
                      self.const.logp(value))
