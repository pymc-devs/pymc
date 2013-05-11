from dist_math import *

__all__  = ['Binomial',  'BetaBin',  'Bernoulli',  'Poisson', 'NegativeBinomial',
'ConstantDist', 'ZeroInflatedPoisson', 'DiscreteUniform', 'Geometric']


@tensordist(discrete)
def Binomial(n, p):
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
    def logp(value):

        return bound(
            logpow(p, value) + logpow(
                1 - p, n - value) + factln(
                    n) - factln(value) - factln(n - value),

            0 <= value, value <= n,
            0 <= p, p <= 1)

    mode = cast(round(n * p), 'int8')

    logp.__doc__ = """
        Binomial log-likelihood with parameters n={0} and p={1}.

        Parameters
        ----------
        value : int
            Number of successes, x > 0
        """.format(n, p)

    return locals()


@tensordist(discrete)
def BetaBin(alpha, beta, n):
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

    def logp(value):

          return bound(gammaln(alpha + beta) -  gammaln(alpha) - gammaln(beta) +
          gammaln(n  + 1) - gammaln(value + 1) - gammaln(n - value + 1) +
          gammaln(alpha + value) + gammaln(n + beta - value) - gammaln(beta + alpha + n),

            0 <= value, value <= n,
            alpha > 0,
            beta > 0)

    mode = cast(round(alpha / (alpha + beta)), 'int8')

    logp.__doc__ = """
        Beta-binomial log-likelihood with parameters alpha={0}, beta={1},
        and n={2}.

        Parameters
        ----------
        value : int
            x=0,1,...,n
        """.format(alpha, beta, n)

    return locals()


@tensordist(discrete)
def Bernoulli(p):
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

    def logp(value):
        return bound(
            switch(value, log(p), log(1 - p)),
            0 <= p, p <= 1)

    mode = cast(round(p), 'int8')

    logp.__doc__ = """
        Bernoulli log-likelihood with parameter p={0}.

        Parameters
        ----------
        value : int
            Series of successes (1) and failures (0). :math:`x=0,1`
        """.format(p)

    return locals()


@tensordist(discrete)
def Poisson(mu):
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
    def logp(value):
        return bound(
            logpow(mu, value) - factln(value) - mu,

            mu > 0, value >= 0)

    logp.__doc__ = """
        Poisson log-likelihood with parameters mu={0}.

        Parameters
        ----------
        x : int
            :math:`x \in {{0,1,2,...}}`
        """.format(mu)

    mode = floor(mu).astype('int32')
    return locals()


@tensordist(discrete)
def NegativeBinomial(mu, alpha):
    """
    Negative binomial log-likelihood.

     The negative binomial distribution  describes a Poisson random variable
     whose rate parameter is gamma distributed. PyMC's chosen parameterization
     is based on this mixture interpretation.

    .. math::
        f(x \mid \mu, \alpha) = \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x

    :Parameters:
      - `mu` : mu > 0
      - `alpha` : alpha > 0

    .. note::
      - :math:`E[x]=\mu`

    """
    def logp(value):
        # Return Poisson when alpha gets very large
        return switch(alpha > 1e10,

            bound(logpow(mu, value) - factln(value) - mu,
            mu > 0, value >= 0),

            bound(gammaln(value + alpha) - factln(value) - gammaln(alpha) +
            logpow(mu / (mu + alpha), value) + logpow(alpha / (mu + alpha), alpha),
            mu > 0, alpha > 0, value >= 0))

    logp.__doc__ = """
        Negative binomial log-likelihood with parameters ({0},{1}).

        Parameters
        ----------
        x : int
            :math:`x \in {{0,1,2,...}}`
        """.format(mu, alpha)

    mode = floor(mu).astype('int32')
    return locals()


@tensordist(discrete)
def Geometric(p):
    """
    Geometric log-likelihood. The probability that the first success in a
    sequence of Bernoulli trials occurs on the x'th trial.

    .. math::
        f(x \mid p) = p(1-p)^{x-1}

    :Parameters:
      - `p` : Probability of success on an individual trial, :math:`p \in [0,1]`

    .. note::
      - :math:`E(X)=1/p`
      - :math:`Var(X)=\frac{1-p}{p^2}`

    """

    def logp(value):
        return bound(log(p) + logpow(1 - p, value - 1),
                     0 <= p, p <= 1, value >= 1)

    logp.__doc__ = """
        Geometric log-likelihood with parameter {0}.

        Parameters
        ----------
        x : int
            :math:`x \in {{1,2,...}}`
        """.format(p)

    mode = 1

    return locals()


@tensordist(discrete)
def DiscreteUniform(lower, upper):
    """
    Discrete uniform distribution.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    :Parameters:
      - `lower` : Lower limit.
      - `upper` : Upper limit (upper > lower).

    """
    lower, upper = floor(lower).astype('int32'), floor(upper).astype('int32')

    def logp(value):

        return bound(
            -log(upper - lower + 1),

            lower <= value, value <= upper)

    logp.__doc__ = """
        Discrete uniform log-likelihood with bounds ({0},{1}).

        Parameters
        ----------
        x : int
            :math:`lower \leq x \leq upper`
        """.format(upper, lower)

    mode = floor((upper - lower) / 2.).astype('int32')
    return locals()


@tensordist(discrete)
def ConstantDist(c):
    def logp(value):
        return bound(0, eq(value, c))

    mean = median = mode = c
    logp.__doc__ = """
        Constant log-likelihood with parameter c={0}.

        Parameters
        ----------
        value : float or int
            Data value(s)
        """.format(c)
    return locals()


@tensordist(discrete)
def ZeroInflatedPoisson(theta, z):
    pois = Poisson(theta)
    const = ConstantDist(0)

    def logp(value):
        return switch(z,
                      pois.logp(value),
                      const.logp(value))
    mode = pois.mode
    return locals()
