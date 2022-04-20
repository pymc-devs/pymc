#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import warnings

import aesara.tensor as at
import numpy as np

from aesara.tensor.random.basic import (
    RandomVariable,
    bernoulli,
    betabinom,
    binomial,
    categorical,
    geometric,
    hypergeometric,
    nbinom,
    poisson,
)
from scipy import stats

import pymc as pm

from pymc.aesaraf import floatX, intX, take_along_axis
from pymc.distributions.dist_math import (
    betaln,
    binomln,
    check_parameters,
    factln,
    log_diff_normal_cdf,
    logpow,
    normal_lccdf,
    normal_lcdf,
)
from pymc.distributions.distribution import Discrete
from pymc.distributions.logprob import logp
from pymc.distributions.mixture import Mixture
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.math import sigmoid
from pymc.vartypes import continuous_types

__all__ = [
    "Binomial",
    "BetaBinomial",
    "Bernoulli",
    "DiscreteWeibull",
    "Poisson",
    "NegativeBinomial",
    "Constant",
    "ZeroInflatedPoisson",
    "ZeroInflatedBinomial",
    "ZeroInflatedNegativeBinomial",
    "DiscreteUniform",
    "Geometric",
    "HyperGeometric",
    "Categorical",
    "OrderedLogistic",
    "OrderedProbit",
]


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
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    logit_p : float
        Alternative log odds for the probability of success.
    """
    rv_op = binomial

    @classmethod
    def dist(cls, n, p=None, logit_p=None, *args, **kwargs):
        if p is not None and logit_p is not None:
            raise ValueError("Incompatible parametrization. Can't specify both p and logit_p.")
        elif p is None and logit_p is None:
            raise ValueError("Incompatible parametrization. Must specify either p or logit_p.")

        if logit_p is not None:
            p = at.sigmoid(logit_p)

        n = at.as_tensor_variable(intX(n))
        p = at.as_tensor_variable(floatX(p))
        return super().dist([n, p], **kwargs)

    def moment(rv, size, n, p):
        mean = at.round(n * p)
        if not rv_size_is_none(size):
            mean = at.full(size, mean)
        return mean

    def logp(value, n, p):
        r"""
        Calculate log-probability of Binomial distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        res = at.switch(
            at.or_(at.lt(value, 0), at.gt(value, n)),
            -np.inf,
            binomln(n, value) + logpow(p, value) + logpow(1 - p, n - value),
        )

        return check_parameters(res, 0 <= n, 0 <= p, p <= 1, msg="n >= 0, 0 <= p <= 1")

    def logcdf(value, n, p):
        """
        Compute the log of the cumulative distribution function for Binomial distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        value = at.floor(value)

        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.switch(
                at.lt(value, n),
                at.log(at.betainc(n - value, value + 1, 1 - p)),
                0,
            ),
        )

        return check_parameters(
            res,
            0 <= n,
            0 <= p,
            p <= 1,
            msg="n >= 0, 0 <= p <= 1",
        )


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
        import arviz as az
        plt.style.use('arviz-darkgrid')

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
    n: int
        Number of Bernoulli trials (n >= 0).
    alpha: float
        alpha > 0.
    beta: float
        beta > 0.
    """

    rv_op = betabinom

    @classmethod
    def dist(cls, alpha, beta, n, *args, **kwargs):
        alpha = at.as_tensor_variable(floatX(alpha))
        beta = at.as_tensor_variable(floatX(beta))
        n = at.as_tensor_variable(intX(n))
        return super().dist([n, alpha, beta], **kwargs)

    def moment(rv, size, n, alpha, beta):
        mean = at.round((n * alpha) / (alpha + beta))
        if not rv_size_is_none(size):
            mean = at.full(size, mean)
        return mean

    def logp(value, n, alpha, beta):
        r"""
        Calculate log-probability of BetaBinomial distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        res = at.switch(
            at.or_(at.lt(value, 0), at.gt(value, n)),
            -np.inf,
            binomln(n, value) + betaln(value + alpha, n - value + beta) - betaln(alpha, beta),
        )
        return check_parameters(res, n >= 0, alpha > 0, beta > 0, msg="n >= 0, alpha > 0, beta > 0")

    def logcdf(value, n, alpha, beta):
        """
        Compute the log of the cumulative distribution function for BetaBinomial distribution
        at the specified value.

        Parameters
        ----------
        value: numeric
            Value for which log CDF is calculated.

        Returns
        -------
        TensorVariable
        """
        # logcdf can only handle scalar values at the moment
        if np.ndim(value):
            raise TypeError(
                f"BetaBinomial.logcdf expects a scalar value but received a {np.ndim(value)}-dimensional object."
            )

        safe_lower = at.switch(at.lt(value, 0), value, 0)
        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.switch(
                at.lt(value, n),
                at.logsumexp(
                    logp(
                        BetaBinomial.dist(alpha=alpha, beta=beta, n=n),
                        at.arange(safe_lower, value + 1),
                    ),
                    keepdims=False,
                ),
                0,
            ),
        )
        return check_parameters(res, 0 <= n, 0 < alpha, 0 < beta, msg="n >= 0, alpha > 0, beta > 0")


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
        import arviz as az
        plt.style.use('arviz-darkgrid')
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

    The bernoulli distribution can be parametrized either in terms of p or logit_p.
    The link between the parametrizations is given by

    .. math:: logit(p) = ln(\frac{p}{1-p})

    Parameters
    ----------
    p: float
        Probability of success (0 < p < 1).
    logit_p: float
        Alternative log odds for the probability of success.
    """
    rv_op = bernoulli

    @classmethod
    def dist(cls, p=None, logit_p=None, *args, **kwargs):
        if p is not None and logit_p is not None:
            raise ValueError("Incompatible parametrization. Can't specify both p and logit_p.")
        elif p is None and logit_p is None:
            raise ValueError("Incompatible parametrization. Must specify either p or logit_p.")

        if logit_p is not None:
            p = at.sigmoid(logit_p)

        p = at.as_tensor_variable(floatX(p))
        return super().dist([p], **kwargs)

    def moment(rv, size, p):
        if not rv_size_is_none(size):
            p = at.full(size, p)
        return at.switch(p < 0.5, 0, 1)

    def logp(value, p):
        r"""
        Calculate log-probability of Bernoulli distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        res = at.switch(
            at.or_(at.lt(value, 0), at.gt(value, 1)),
            -np.inf,
            at.switch(value, at.log(p), at.log1p(-p)),
        )

        return check_parameters(res, p >= 0, p <= 1, msg="0 <= p <= 1")

    def logcdf(value, p):
        """
        Compute the log of the cumulative distribution function for Bernoulli distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.switch(
                at.lt(value, 1),
                at.log1p(-p),
                0,
            ),
        )
        return check_parameters(res, 0 <= p, p <= 1, msg="0 <= p <= 1")


class DiscreteWeibullRV(RandomVariable):
    name = "discrete_weibull"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "int64"
    _print_name = ("dWeibull", "\\operatorname{dWeibull}")

    @classmethod
    def rng_fn(cls, rng, q, beta, size):
        p = rng.uniform(size=size)
        return np.ceil(np.power(np.log(1 - p) / np.log(q), 1.0 / beta)) - 1


discrete_weibull = DiscreteWeibullRV()


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
        import arviz as az
        plt.style.use('arviz-darkgrid')

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
    rv_op = discrete_weibull

    @classmethod
    def dist(cls, q, beta, *args, **kwargs):
        q = at.as_tensor_variable(floatX(q))
        beta = at.as_tensor_variable(floatX(beta))
        return super().dist([q, beta], **kwargs)

    def moment(rv, size, q, beta):
        median = at.power(at.log(0.5) / at.log(q), 1 / beta) - 1
        if not rv_size_is_none(size):
            median = at.full(size, median)
        return median

    def logp(value, q, beta):
        r"""
        Calculate log-probability of DiscreteWeibull distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.log(at.power(q, at.power(value, beta)) - at.power(q, at.power(value + 1, beta))),
        )

        return check_parameters(res, 0 < q, q < 1, 0 < beta, msg="0 < q < 1, beta > 0")

    def logcdf(value, q, beta):
        """
        Compute the log of the cumulative distribution function for Discrete Weibull distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """

        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.log1p(-at.power(q, at.power(value + 1, beta))),
        )
        return check_parameters(res, 0 < q, q < 1, 0 < beta, msg="0 < q < 1, beta > 0")


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
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu: float
        Expected number of occurrences during the given interval
        (mu >= 0).

    Notes
    -----
    The Poisson distribution can be derived as a limiting case of the
    binomial distribution.
    """
    rv_op = poisson

    @classmethod
    def dist(cls, mu, *args, **kwargs):
        mu = at.as_tensor_variable(floatX(mu))
        return super().dist([mu], *args, **kwargs)

    def moment(rv, size, mu):
        mu = at.floor(mu)
        if not rv_size_is_none(size):
            mu = at.full(size, mu)
        return mu

    def logp(value, mu):
        r"""
        Calculate log-probability of Poisson distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        logprob = at.switch(
            at.lt(value, 0),
            -np.inf,
            logpow(mu, value) - factln(value) - mu,
        )
        # Return zero when mu and value are both zero
        logprob = at.switch(
            at.eq(mu, 0) * at.eq(value, 0),
            0,
            logprob,
        )
        return check_parameters(logprob, mu >= 0, msg="mu >= 0")

    def logcdf(value, mu):
        """
        Compute the log of the cumulative distribution function for Poisson distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        value = at.floor(value)
        # Avoid C-assertion when the gammaincc function is called with invalid values (#4340)
        safe_mu = at.switch(at.lt(mu, 0), 0, mu)
        safe_value = at.switch(at.lt(value, 0), 0, value)

        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.log(at.gammaincc(safe_value + 1, safe_mu)),
        )

        return check_parameters(res, 0 <= mu, msg="mu >= 0")


class NegativeBinomial(Discrete):
    R"""
    Negative binomial log-likelihood.

    The negative binomial distribution describes a Poisson random variable
    whose rate parameter is gamma distributed.
    Its pmf, parametrized by the parameters alpha and mu of the gamma distribution, is

    .. math::

       f(x \mid \mu, \alpha) =
           \binom{x + \alpha - 1}{x}
           (\alpha/(\mu+\alpha))^\alpha (\mu/(\mu+\alpha))^x

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy import special
        import arviz as az
        plt.style.use('arviz-darkgrid')

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

    The negative binomial distribution can be parametrized either in terms of mu or p,
    and either in terms of alpha or n. The link between the parametrizations is given by

    .. math::

        p &= \frac{\alpha}{\mu + \alpha} \\
        n &= \alpha

    If it is parametrized in terms of n and p, the negative binomial describes the probability to have x failures
    before the n-th success, given the probability p of success in each trial. Its pmf is

    .. math::

        f(x \mid n, p) =
           \binom{x + n - 1}{x}
           (p)^n (1 - p)^x

    Parameters
    ----------
    alpha: float
        Gamma distribution shape parameter (alpha > 0).
    mu: float
        Gamma distribution mean (mu > 0).
    p: float
        Alternative probability of success in each trial (0 < p < 1).
    n: float
        Alternative number of target success trials (n > 0)
    """
    rv_op = nbinom

    @classmethod
    def dist(cls, mu=None, alpha=None, p=None, n=None, *args, **kwargs):
        n, p = cls.get_n_p(mu=mu, alpha=alpha, p=p, n=n)
        n = at.as_tensor_variable(floatX(n))
        p = at.as_tensor_variable(floatX(p))
        return super().dist([n, p], *args, **kwargs)

    @classmethod
    def get_n_p(cls, mu=None, alpha=None, p=None, n=None):
        if n is None:
            if alpha is not None:
                n = alpha
            else:
                raise ValueError("Incompatible parametrization. Must specify either alpha or n.")
        elif alpha is not None:
            raise ValueError("Incompatible parametrization. Can't specify both alpha and n.")

        if p is None:
            if mu is not None:
                p = n / (mu + n)
            else:
                raise ValueError("Incompatible parametrization. Must specify either mu or p.")
        elif mu is not None:
            raise ValueError("Incompatible parametrization. Can't specify both mu and p.")

        return n, p

    def moment(rv, size, n, p):
        mu = at.floor(n * (1 - p) / p)
        if not rv_size_is_none(size):
            mu = at.full(size, mu)
        return mu

    def logp(value, n, p):
        r"""
        Calculate log-probability of NegativeBinomial distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        alpha = n
        mu = alpha * (1 - p) / p

        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            (
                binomln(value + alpha - 1, value)
                + logpow(mu / (mu + alpha), value)
                + logpow(alpha / (mu + alpha), alpha)
            ),
        )

        negbinom = check_parameters(
            res,
            mu > 0,
            alpha > 0,
            msg="mu > 0, alpha > 0",
        )

        # Return Poisson when alpha gets very large.
        return at.switch(at.gt(alpha, 1e10), logp(Poisson.dist(mu=mu), value), negbinom)

    def logcdf(value, n, p):
        """
        Compute the log of the cumulative distribution function for NegativeBinomial distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """
        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.log(at.betainc(n, at.floor(value) + 1, p)),
        )
        return check_parameters(
            res,
            0 < n,
            0 <= p,
            p <= 1,
            msg="0 < n, 0 <= p <= 1",
        )


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
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    p: float
        Probability of success on an individual trial (0 < p <= 1).
    """

    rv_op = geometric

    @classmethod
    def dist(cls, p, *args, **kwargs):
        p = at.as_tensor_variable(floatX(p))
        return super().dist([p], *args, **kwargs)

    def moment(rv, size, p):
        mean = at.round(1.0 / p)
        if not rv_size_is_none(size):
            mean = at.full(size, mean)
        return mean

    def logp(value, p):
        r"""
        Calculate log-probability of Geometric distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """

        res = at.switch(
            at.lt(value, 1),
            -np.inf,
            at.log(p) + logpow(1 - p, value - 1),
        )

        return check_parameters(
            res,
            0 <= p,
            p <= 1,
            msg="0 <= p <= 1",
        )

    def logcdf(value, p):
        """
        Compute the log of the cumulative distribution function for Geometric distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """

        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.log1mexp(at.log1p(-p) * value),
        )
        return check_parameters(
            res,
            0 <= p,
            p <= 1,
            msg="0 <= p <= 1",
        )


class HyperGeometric(Discrete):
    R"""
    Discrete hypergeometric distribution.

    The probability of :math:`x` successes in a sequence of :math:`n` bernoulli
    trials taken without replacement from a population of :math:`N` objects,
    containing :math:`k` good (or successful or Type I) objects.
    The pmf of this distribution is

    .. math:: f(x \mid N, n, k) = \frac{\binom{k}{x}\binom{N-k}{n-x}}{\binom{N}{n}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.arange(1, 15)
        N = 50
        k = 10
        for n in [20, 25]:
            pmf = st.hypergeom.pmf(x, N, k, n)
            plt.plot(x, pmf, '-o', label='n = {}'.format(n))
        plt.plot(x, pmf, '-o', label='N = {}'.format(N))
        plt.plot(x, pmf, '-o', label='k = {}'.format(k))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================
    Support   :math:`x \in \left[\max(0, n - N + k), \min(k, n)\right]`
    Mean      :math:`\dfrac{nk}{N}`
    Variance  :math:`\dfrac{(N-n)nk(N-k)}{(N-1)N^2}`
    ========  =============================

    Parameters
    ----------
    N : tensor_like of integer
        Total size of the population (N > 0)
    k : tensor_like of integer
        Number of successful individuals in the population (0 <= k <= N)
    n : tensor_like of integer
        Number of samples drawn from the population (0 <= n <= N)
    """

    rv_op = hypergeometric

    @classmethod
    def dist(cls, N, k, n, *args, **kwargs):
        good = at.as_tensor_variable(intX(k))
        bad = at.as_tensor_variable(intX(N - k))
        n = at.as_tensor_variable(intX(n))
        return super().dist([good, bad, n], *args, **kwargs)

    def moment(rv, size, good, bad, n):
        N, k = good + bad, good
        mode = at.floor((n + 1) * (k + 1) / (N + 2))
        if not rv_size_is_none(size):
            mode = at.full(size, mode)
        return mode

    def logp(value, good, bad, n):
        r"""
        Calculate log-probability of HyperGeometric distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor
        good : integer, array_like or TensorVariable
            Number of successful individuals in the population. Alias for parameter :math:`k`.
        bad : integer, array_like or TensorVariable
            Number of unsuccessful individuals in the population. Alias for :math:`N-k`.

        Returns
        -------
        TensorVariable
        """

        tot = good + bad
        result = (
            betaln(good + 1, 1)
            + betaln(bad + 1, 1)
            + betaln(tot - n + 1, n + 1)
            - betaln(value + 1, good - value + 1)
            - betaln(n - value + 1, bad - n + value + 1)
            - betaln(tot + 1, 1)
        )
        # value in [max(0, n - N + k), min(k, n)]
        lower = at.switch(at.gt(n - tot + good, 0), n - tot + good, 0)
        upper = at.switch(at.lt(good, n), good, n)

        res = at.switch(
            at.lt(value, lower),
            -np.inf,
            at.switch(
                at.le(value, upper),
                result,
                -np.inf,
            ),
        )

        return check_parameters(res, lower <= upper, msg="lower <= upper")

    def logcdf(value, good, bad, n):
        """
        Compute the log of the cumulative distribution function for HyperGeometric distribution
        at the specified value.

        Parameters
        ----------
        value : numeric
            Value for which log CDF is calculated.
        good : integer
            Number of successful individuals in the population. Alias for parameter :math:`k`.
        bad : integer
            Number of unsuccessful individuals in the population. Alias for :math:`N-k`.
        n : integer
            Number of samples drawn from the population (0 <= n <= N)

        Returns
        -------
        TensorVariable
        """
        # logcdf can only handle scalar values at the moment
        if np.ndim(value):
            raise TypeError(
                f"HyperGeometric.logcdf expects a scalar value but received a {np.ndim(value)}-dimensional object."
            )

        N = good + bad
        # TODO: Use lower upper in locgdf for smarter logsumexp?
        safe_lower = at.switch(at.lt(value, 0), value, 0)

        res = at.switch(
            at.lt(value, 0),
            -np.inf,
            at.switch(
                at.lt(value, n),
                at.logsumexp(
                    HyperGeometric.logp(at.arange(safe_lower, value + 1), good, bad, n),
                    keepdims=False,
                ),
                0,
            ),
        )

        return check_parameters(
            res,
            0 < N,
            0 <= good,
            0 <= n,
            good <= N,
            n <= N,
            msg="N > 0, 0 <= good <= N, 0 <= n <= N",
        )


class DiscreteUniformRV(RandomVariable):
    name = "discrete_uniform"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "int64"
    _print_name = ("DiscreteUniform", "\\operatorname{DiscreteUniform}")

    @classmethod
    def rng_fn(cls, rng, lower, upper, size=None):
        return stats.randint.rvs(lower, upper + 1, size=size, random_state=rng)


discrete_uniform = DiscreteUniformRV()


class DiscreteUniform(Discrete):
    R"""
    Discrete uniform distribution.
    The pmf of this distribution is

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower+1}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    lower: int
        Lower limit.
    upper: int
        Upper limit (upper > lower).
    """

    rv_op = discrete_uniform

    @classmethod
    def dist(cls, lower, upper, *args, **kwargs):
        lower = intX(at.floor(lower))
        upper = intX(at.floor(upper))
        return super().dist([lower, upper], **kwargs)

    def moment(rv, size, lower, upper):
        mode = at.maximum(at.floor((upper + lower) / 2.0), lower)
        if not rv_size_is_none(size):
            mode = at.full(size, mode)
        return mode

    def logp(value, lower, upper):
        r"""
        Calculate log-probability of DiscreteUniform distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        res = at.switch(
            at.or_(at.lt(value, lower), at.gt(value, upper)),
            -np.inf,
            at.fill(value, -at.log(upper - lower + 1)),
        )
        return check_parameters(res, lower <= upper, msg="lower <= upper")

    def logcdf(value, lower, upper):
        """
        Compute the log of the cumulative distribution function for Discrete uniform distribution
        at the specified value.

        Parameters
        ----------
        value: numeric or np.ndarray or aesara.tensor
            Value(s) for which log CDF is calculated. If the log CDF for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor.

        Returns
        -------
        TensorVariable
        """

        res = at.switch(
            at.le(value, lower),
            -np.inf,
            at.switch(
                at.lt(value, upper),
                at.log(at.minimum(at.floor(value), upper) - lower + 1) - at.log(upper - lower + 1),
                0,
            ),
        )

        return check_parameters(res, lower <= upper, msg="lower <= upper")


class Categorical(Discrete):
    R"""
    Categorical log-likelihood.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    logit_p : float
        Alternative log odds for the probability of success.
    """
    rv_op = categorical

    @classmethod
    def dist(cls, p=None, logit_p=None, **kwargs):
        if p is not None and logit_p is not None:
            raise ValueError("Incompatible parametrization. Can't specify both p and logit_p.")
        elif p is None and logit_p is None:
            raise ValueError("Incompatible parametrization. Must specify either p or logit_p.")

        if logit_p is not None:
            p = pm.math.softmax(logit_p, axis=-1)

        if isinstance(p, np.ndarray) or isinstance(p, list):
            if (np.asarray(p) < 0).any():
                raise ValueError(f"Negative `p` parameters are not valid, got: {p}")
            p_sum = np.sum([p], axis=-1)
            if not np.all(np.isclose(p_sum, 1.0)):
                warnings.warn(
                    f"`p` parameters sum to {p_sum}, instead of 1.0. They will be automatically rescaled. You can rescale them directly to get rid of this warning.",
                    UserWarning,
                )
                p = p / at.sum(p, axis=-1, keepdims=True)
        p = at.as_tensor_variable(floatX(p))
        return super().dist([p], **kwargs)

    def moment(rv, size, p):
        mode = at.argmax(p, axis=-1)
        if not rv_size_is_none(size):
            mode = at.full(size, mode)
        return mode

    def logp(value, p):
        r"""
        Calculate log-probability of Categorical distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or `TensorVariable`

        """
        k = at.shape(p)[-1]
        p_ = p
        value_clip = at.clip(value, 0, k - 1)

        if p.ndim > 1:
            if p.ndim > value_clip.ndim:
                value_clip = at.shape_padleft(value_clip, p_.ndim - value_clip.ndim)
            elif p.ndim < value_clip.ndim:
                p = at.shape_padleft(p, value_clip.ndim - p_.ndim)
            pattern = (p.ndim - 1,) + tuple(range(p.ndim - 1))
            a = at.log(
                take_along_axis(
                    p.dimshuffle(pattern),
                    value_clip,
                )
            )
        else:
            a = at.log(p[value_clip])

        res = at.switch(
            at.or_(at.lt(value, 0), at.gt(value, k - 1)),
            -np.inf,
            a,
        )

        return check_parameters(
            res, at.all(p_ >= 0, axis=-1), at.all(p <= 1, axis=-1), msg="0 <= p <=1"
        )


class ConstantRV(RandomVariable):
    name = "constant"
    ndim_supp = 0
    ndims_params = [0]
    _print_name = ("Constant", "\\operatorname{Constant}")

    def make_node(self, rng, size, dtype, c):
        c = at.as_tensor_variable(c)
        return super().make_node(rng, size, c.dtype, c)

    @classmethod
    def rng_fn(cls, rng, c, size=None):
        if size is None:
            return c.copy()
        return np.full(size, c)


constant = ConstantRV()


class Constant(Discrete):
    r"""
    Constant log-likelihood.

    Parameters
    ----------
    c: float or int
        Constant parameter. The dtype of `c` determines the dtype of the distribution.
        This can affect which sampler is assigned to Constant variables, or variables
        that use Constant, such as Mixtures.
    """

    rv_op = constant

    @classmethod
    def dist(cls, c, *args, **kwargs):
        c = at.as_tensor_variable(c)
        if c.dtype in continuous_types:
            c = floatX(c)
        return super().dist([c], **kwargs)

    def moment(rv, size, c):
        if not rv_size_is_none(size):
            c = at.full(size, c)
        return c

    def logp(value, c):
        r"""
        Calculate log-probability of Constant distribution at specified value.

        Parameters
        ----------
        value: numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or Aesara tensor

        Returns
        -------
        TensorVariable
        """
        return at.switch(
            at.eq(value, c),
            at.zeros_like(value),
            -np.inf,
        )

    def logcdf(value, c):
        return at.switch(
            at.lt(value, c),
            -np.inf,
            0,
        )


def _zero_inflated_mixture(*, name, nonzero_p, nonzero_dist, **kwargs):
    """Helper function to create a zero-inflated mixture

    If name is `None`, this function returns an unregistered variable
    """
    nonzero_p = at.as_tensor_variable(floatX(nonzero_p))
    weights = at.stack([1 - nonzero_p, nonzero_p], axis=-1)
    comp_dists = [
        Constant.dist(0),
        nonzero_dist,
    ]
    if name is not None:
        return Mixture(name, weights, comp_dists, **kwargs)
    else:
        return Mixture.dist(weights, comp_dists, **kwargs)


class ZeroInflatedPoisson:
    R"""
    Zero-inflated Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math::

        f(x \mid \psi, \mu) = \left\{ \begin{array}{l}
            (1-\psi) + \psi e^{-\mu}, \text{if } x = 0 \\
            \psi \frac{e^{-\theta}\theta^x}{x!}, \text{if } x=1,2,3,\ldots
            \end{array} \right.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.arange(0, 22)
        psis = [0.7, 0.4]
        mus = [8, 4]
        for psi, mu in zip(psis, mus):
            pmf = st.poisson.pmf(x, mu)
            pmf[0] = (1 - psi) + pmf[0]
            pmf[1:] =  psi * pmf[1:]
            pmf /= pmf.sum()
            plt.plot(x, pmf, '-o', label='$\\psi$ = {}, $\\mu$ = {}'.format(psi, mu))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================
    Support   :math:`x \in \mathbb{N}_0`
    Mean      :math:`\psi\mu`
    Variance  :math:`\mu + \frac{1-\psi}{\psi}\mu^2`
    ========  ==========================

    Parameters
    ----------
    psi: float
        Expected proportion of Poisson variates (0 < psi < 1)
    mu: float
        Expected number of occurrences during the given interval
        (mu >= 0).
    """

    def __new__(cls, name, psi, mu, **kwargs):
        return _zero_inflated_mixture(
            name=name, nonzero_p=psi, nonzero_dist=Poisson.dist(mu=mu), **kwargs
        )

    @classmethod
    def dist(cls, psi, mu, **kwargs):
        return _zero_inflated_mixture(
            name=None, nonzero_p=psi, nonzero_dist=Poisson.dist(mu=mu), **kwargs
        )


class ZeroInflatedBinomial:
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
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    Mean      :math:`\psi n p`
    Variance  :math:`(1-\psi) n p [1 - p(1 - \psi n)].`
    ========  ==========================

    Parameters
    ----------
    psi: float
        Expected proportion of Binomial variates (0 < psi < 1)
    n: int
        Number of Bernoulli trials (n >= 0).
    p: float
        Probability of success in each trial (0 < p < 1).

    """

    def __new__(cls, name, psi, n, p, **kwargs):
        return _zero_inflated_mixture(
            name=name, nonzero_p=psi, nonzero_dist=Binomial.dist(n=n, p=p), **kwargs
        )

    @classmethod
    def dist(cls, psi, n, p, **kwargs):
        return _zero_inflated_mixture(
            name=None, nonzero_p=psi, nonzero_dist=Binomial.dist(n=n, p=p), **kwargs
        )


class ZeroInflatedNegativeBinomial:
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
        import arviz as az
        plt.style.use('arviz-darkgrid')

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

    The zero inflated negative binomial distribution can be parametrized
    either in terms of mu or p, and either in terms of alpha or n.
    The link between the parametrizations is given by

    .. math::

        \mu &= \frac{n(1-p)}{p} \\
        \alpha &= n

    Parameters
    ----------
    psi: float
        Expected proportion of NegativeBinomial variates (0 < psi < 1)
    mu: float
        Poission distribution parameter (mu > 0).
    alpha: float
        Gamma distribution parameter (alpha > 0).
    p: float
        Alternative probability of success in each trial (0 < p < 1).
    n: float
        Alternative number of target success trials (n > 0)
    """

    def __new__(cls, name, psi, mu=None, alpha=None, p=None, n=None, **kwargs):
        return _zero_inflated_mixture(
            name=name,
            nonzero_p=psi,
            nonzero_dist=NegativeBinomial.dist(mu=mu, alpha=alpha, p=p, n=n),
            **kwargs,
        )

    @classmethod
    def dist(cls, psi, mu=None, alpha=None, p=None, n=None, **kwargs):
        return _zero_inflated_mixture(
            name=None,
            nonzero_p=psi,
            nonzero_dist=NegativeBinomial.dist(mu=mu, alpha=alpha, p=p, n=n),
            **kwargs,
        )


class _OrderedLogistic(Categorical):
    r"""
    Underlying class for ordered logistic distributions.
    See docs for the OrderedLogistic wrapper class for more details on how to use it in models.
    """
    rv_op = categorical

    @classmethod
    def dist(cls, eta, cutpoints, *args, **kwargs):
        eta = at.as_tensor_variable(floatX(eta))
        cutpoints = at.as_tensor_variable(cutpoints)

        pa = sigmoid(cutpoints - at.shape_padright(eta))
        p_cum = at.concatenate(
            [
                at.zeros_like(at.shape_padright(pa[..., 0])),
                pa,
                at.ones_like(at.shape_padright(pa[..., 0])),
            ],
            axis=-1,
        )
        p = p_cum[..., 1:] - p_cum[..., :-1]

        return super().dist(p, *args, **kwargs)


class OrderedLogistic:
    R"""
    Wrapper class for Ordered Logistic distributions.

    Useful for regression on ordinal data values whose values range
    from 1 to K as a function of some predictor, :math:`\eta`. The
    cutpoints, :math:`c`, separate which ranges of :math:`\eta` are
    mapped to which of the K observed dependent variables. The number
    of cutpoints is K - 1. It is recommended that the cutpoints are
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
    eta: float
        The predictor.
    cutpoints: array
        The length K - 1 array of cutpoints which break :math:`\eta` into
        ranges. Do not explicitly set the first and last elements of
        :math:`c` to negative and positive infinity.
    compute_p: boolean, default True
        Whether to compute and store in the trace the inferred probabilities of each categories,
        based on the cutpoints' values. Defaults to True.
        Might be useful to disable it if memory usage is of interest.

    Examples
    --------

    .. code-block:: python

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
            idata = pm.sample()

        # Plot the results
        plt.hist(cluster1, 30, alpha=0.5);
        plt.hist(cluster2, 30, alpha=0.5);
        plt.hist(cluster3, 30, alpha=0.5);
        posterior = idata.posterior.stack(sample=("chain", "draw"))
        plt.hist(posterior["cutpoints"][0], 80, alpha=0.2, color='k');
        plt.hist(posterior["cutpoints"][1], 80, alpha=0.2, color='k');
    """

    def __new__(cls, name, *args, compute_p=True, **kwargs):
        out_rv = _OrderedLogistic(name, *args, **kwargs)
        if compute_p:
            pm.Deterministic(f"{name}_probs", out_rv.owner.inputs[3], dims=kwargs.get("dims"))
        return out_rv

    @classmethod
    def dist(cls, *args, **kwargs):
        return _OrderedLogistic.dist(*args, **kwargs)


class _OrderedProbit(Categorical):
    r"""
    Underlying class for ordered probit distributions.
    See docs for the OrderedProbit wrapper class for more details on how to use it in models.
    """
    rv_op = categorical

    @classmethod
    def dist(cls, eta, cutpoints, sigma=1, *args, **kwargs):
        eta = at.as_tensor_variable(floatX(eta))
        cutpoints = at.as_tensor_variable(cutpoints)

        probits = at.shape_padright(eta) - cutpoints
        _log_p = at.concatenate(
            [
                at.shape_padright(normal_lccdf(0, sigma, probits[..., 0])),
                log_diff_normal_cdf(
                    0, at.shape_padright(sigma), probits[..., :-1], probits[..., 1:]
                ),
                at.shape_padright(normal_lcdf(0, sigma, probits[..., -1])),
            ],
            axis=-1,
        )
        _log_p = at.as_tensor_variable(floatX(_log_p))
        p = at.exp(_log_p)

        return super().dist(p, *args, **kwargs)


class OrderedProbit:
    R"""
    Wrapper class for Ordered Probit distributions.

    Useful for regression on ordinal data values whose values range
    from 1 to K as a function of some predictor, :math:`\eta`. The
    cutpoints, :math:`c`, separate which ranges of :math:`\eta` are
    mapped to which of the K observed dependent variables. The number
    of cutpoints is K - 1. It is recommended that the cutpoints are
    constrained to be ordered.

    In order to stabilize the computation, log-likelihood is computed
    in log space using the scaled error function `erfcx`.

    .. math::

       f(k \mid \eta, c) = \left\{
         \begin{array}{l}
           1 - \text{normal_cdf}(0, \sigma, \eta - c_1)
             \,, \text{if } k = 0 \\
           \text{normal_cdf}(0, \sigma, \eta - c_{k - 1}) -
           \text{normal_cdf}(0, \sigma, \eta - c_{k})
             \,, \text{if } 0 < k < K \\
           \text{normal_cdf}(0, \sigma, \eta - c_{K - 1})
             \,, \text{if } k = K \\
         \end{array}
       \right.

    Parameters
    ----------
    eta: float
        The predictor.
    cutpoints: array
        The length K - 1 array of cutpoints which break :math:`\eta` into
        ranges. Do not explicitly set the first and last elements of
        :math:`c` to negative and positive infinity.
    sigma: float, default 1.0
         Standard deviation of the probit function.
    compute_p: boolean, default True
        Whether to compute and store in the trace the inferred probabilities of each categories,
        based on the cutpoints' values. Defaults to True.
        Might be useful to disable it if memory usage is of interest.
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

        # Ordered probit regression
        with pm.Model() as model:
            cutpoints = pm.Normal("cutpoints", mu=[-1,1], sigma=10, shape=2,
                                  transform=pm.distributions.transforms.ordered)
            y_ = pm.OrderedProbit("y", cutpoints=cutpoints, eta=x, observed=y)
            idata = pm.sample()

        # Plot the results
        plt.hist(cluster1, 30, alpha=0.5);
        plt.hist(cluster2, 30, alpha=0.5);
        plt.hist(cluster3, 30, alpha=0.5);
        posterior = idata.posterior.stack(sample=("chain", "draw"))
        plt.hist(posterior["cutpoints"][0], 80, alpha=0.2, color='k');
        plt.hist(posterior["cutpoints"][1], 80, alpha=0.2, color='k');
    """

    def __new__(cls, name, *args, compute_p=True, **kwargs):
        out_rv = _OrderedProbit(name, *args, **kwargs)
        if compute_p:
            pm.Deterministic(f"{name}_probs", out_rv.owner.inputs[3], dims=kwargs.get("dims"))
        return out_rv

    @classmethod
    def dist(cls, *args, **kwargs):
        return _OrderedProbit.dist(*args, **kwargs)
