from dist_math import *

__all__ = ['Binomial', 'BetaBin', 'Bernoulli', 'Poisson', 'ConstantDist',
           'ZeroInflatedPoisson']


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
        n=x,x+1,\ldots

    .. note::
    - :math:`E(X)=n\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=n\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """

    def logp(value):

        return bound(
            gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) + gammaln(n + 1) - gammaln(value + 1) - gammaln(n - value + 1) + gammaln(alpha + value) + gammaln(n + beta - value) - gammaln(beta + alpha + n),

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
            x=0,1,\ldots,n
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
    x : int
        Series of successes (1) and failures (0). :math:`x=0,1`
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

            mu > 0)

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
def ConstantDist(c):
    def logp(value):
        return bound(
            0,

            eq(value, c))

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
