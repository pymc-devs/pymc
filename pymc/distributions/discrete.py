#   Copyright 2023 The PyMC Developers
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

import numpy as np
import pytensor.tensor as pt

from pytensor.tensor import TensorConstant
from pytensor.tensor.random.basic import (
    RandomVariable,
    ScipyRandomVariable,
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

from pymc.distributions.dist_math import (
    betaln,
    binomln,
    check_icdf_parameters,
    check_icdf_value,
    check_parameters,
    factln,
    log_diff_normal_cdf,
    logpow,
    normal_lccdf,
    normal_lcdf,
)
from pymc.distributions.distribution import Discrete
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.logprob.basic import logcdf, logp
from pymc.math import sigmoid
from pymc.pytensorf import floatX, intX

__all__ = [
    "Binomial",
    "BetaBinomial",
    "Bernoulli",
    "DiscreteWeibull",
    "Poisson",
    "NegativeBinomial",
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
        :context: close-figs

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
    n : tensor_like of int
        Number of Bernoulli trials (n >= 0).
    p : tensor_like of float
        Probability of success in each trial (0 < p < 1).
    logit_p : tensor_like of float
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
            p = pt.sigmoid(logit_p)

        n = pt.as_tensor_variable(intX(n))
        p = pt.as_tensor_variable(floatX(p))
        return super().dist([n, p], **kwargs)

    def moment(rv, size, n, p):
        mean = pt.round(n * p)
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, n, p):
        res = pt.switch(
            pt.or_(pt.lt(value, 0), pt.gt(value, n)),
            -np.inf,
            binomln(n, value) + logpow(p, value) + logpow(1 - p, n - value),
        )

        return check_parameters(
            res,
            n >= 0,
            0 <= p,
            p <= 1,
            msg="n >= 0, 0 <= p <= 1",
        )

    def logcdf(value, n, p):
        value = pt.floor(value)

        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.switch(
                pt.lt(value, n),
                pt.log(pt.betainc(n - value, value + 1, 1 - p)),
                0,
            ),
        )

        return check_parameters(
            res,
            n >= 0,
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
        :context: close-figs

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
    Variance  :math:`\dfrac{n \alpha \beta (\alpha+\beta+n)}{(\alpha+\beta)^2 (\alpha+\beta+1)}`
    ========  =================================================================

    Parameters
    ----------
    n : tensor_like of int
        Number of Bernoulli trials (n >= 0).
    alpha : tensor_like of float
        alpha > 0.
    beta : tensor_like of float
        beta > 0.
    """

    rv_op = betabinom

    @classmethod
    def dist(cls, alpha, beta, n, *args, **kwargs):
        alpha = pt.as_tensor_variable(floatX(alpha))
        beta = pt.as_tensor_variable(floatX(beta))
        n = pt.as_tensor_variable(intX(n))
        return super().dist([n, alpha, beta], **kwargs)

    def moment(rv, size, n, alpha, beta):
        mean = pt.round((n * alpha) / (alpha + beta))
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, n, alpha, beta):
        res = pt.switch(
            pt.or_(pt.lt(value, 0), pt.gt(value, n)),
            -np.inf,
            binomln(n, value) + betaln(value + alpha, n - value + beta) - betaln(alpha, beta),
        )
        return check_parameters(
            res,
            n >= 0,
            alpha > 0,
            beta > 0,
            msg="n >= 0, alpha > 0, beta > 0",
        )

    def logcdf(value, n, alpha, beta):
        # logcdf can only handle scalar values at the moment
        if np.ndim(value):
            raise TypeError(
                f"BetaBinomial.logcdf expects a scalar value but received a {np.ndim(value)}-dimensional object."
            )

        safe_lower = pt.switch(pt.lt(value, 0), value, 0)
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.switch(
                pt.lt(value, n),
                pt.logsumexp(
                    logp(
                        BetaBinomial.dist(alpha=alpha, beta=beta, n=n),
                        pt.arange(safe_lower, value + 1),
                    ),
                    keepdims=False,
                ),
                0,
            ),
        )
        return check_parameters(
            res,
            n >= 0,
            alpha > 0,
            beta > 0,
            msg="n >= 0, alpha > 0, beta > 0",
        )


class Bernoulli(Discrete):
    R"""Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes
    (x=1) and failures (x=0).
    The pmf of this distribution is

    .. math:: f(x \mid p) = p^{x} (1-p)^{1-x}

    .. plot::
        :context: close-figs

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
    p : tensor_like of float
        Probability of success (0 < p < 1).
    logit_p : tensor_like of float
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
            p = pt.sigmoid(logit_p)

        p = pt.as_tensor_variable(floatX(p))
        return super().dist([p], **kwargs)

    def moment(rv, size, p):
        if not rv_size_is_none(size):
            p = pt.full(size, p)
        return pt.switch(p < 0.5, 0, 1)

    def logp(value, p):
        res = pt.switch(
            pt.or_(pt.lt(value, 0), pt.gt(value, 1)),
            -np.inf,
            pt.switch(value, pt.log(p), pt.log1p(-p)),
        )

        return check_parameters(
            res,
            0 <= p,
            p <= 1,
            msg="0 <= p <= 1",
        )

    def logcdf(value, p):
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.switch(
                pt.lt(value, 1),
                pt.log1p(-p),
                0,
            ),
        )
        return check_parameters(
            res,
            0 <= p,
            p <= 1,
            msg="0 <= p <= 1",
        )


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
    R"""Discrete Weibull log-likelihood.

    The discrete Weibull distribution is a flexible model of count data that
    can handle both over- and under-dispersion.
    The pmf of this distribution is

    .. math:: f(x \mid q, \beta) = q^{x^{\beta}} - q^{(x + 1)^{\beta}}

    .. plot::
        :context: close-figs

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

    Parameters
    ----------
    q : tensor_like of float
        Shape parameter (0 < q < 1).
    beta : tensor_like of float
        Shape parameter (beta > 0).

    """
    rv_op = discrete_weibull

    @classmethod
    def dist(cls, q, beta, *args, **kwargs):
        q = pt.as_tensor_variable(floatX(q))
        beta = pt.as_tensor_variable(floatX(beta))
        return super().dist([q, beta], **kwargs)

    def moment(rv, size, q, beta):
        median = pt.power(pt.log(0.5) / pt.log(q), 1 / beta) - 1
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median

    def logp(value, q, beta):
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log(pt.power(q, pt.power(value, beta)) - pt.power(q, pt.power(value + 1, beta))),
        )

        return check_parameters(
            res,
            0 < q,
            q < 1,
            beta > 0,
            msg="0 < q < 1, beta > 0",
        )

    def logcdf(value, q, beta):
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log1p(-pt.power(q, pt.power(value + 1, beta))),
        )
        return check_parameters(
            res,
            0 < q,
            q < 1,
            beta > 0,
            msg="0 < q < 1, beta > 0",
        )


class Poisson(Discrete):
    R"""
    Poisson log-likelihood.

    Often used to model the number of events occurring in a fixed period
    of time when the times at which events occur are independent.
    The pmf of this distribution is

    .. math:: f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}

    .. plot::
        :context: close-figs

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
    mu : tensor_like of float
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
        mu = pt.as_tensor_variable(floatX(mu))
        return super().dist([mu], *args, **kwargs)

    def moment(rv, size, mu):
        mu = pt.floor(mu)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    def logp(value, mu):
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            logpow(mu, value) - factln(value) - mu,
        )
        # Return zero when mu and value are both zero
        res = pt.switch(
            pt.eq(mu, 0) * pt.eq(value, 0),
            0,
            res,
        )
        return check_parameters(
            res,
            mu >= 0,
            msg="mu >= 0",
        )

    def logcdf(value, mu):
        value = pt.floor(value)
        # Avoid C-assertion when the gammaincc function is called with invalid values (#4340)
        safe_mu = pt.switch(pt.lt(mu, 0), 0, mu)
        safe_value = pt.switch(pt.lt(value, 0), 0, value)

        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log(pt.gammaincc(safe_value + 1, safe_mu)),
        )

        return check_parameters(
            res,
            mu >= 0,
            msg="mu >= 0",
        )


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
        :context: close-figs

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
    alpha : tensor_like of float
        Gamma distribution shape parameter (alpha > 0).
    mu : tensor_like of float
        Gamma distribution mean (mu > 0).
    p : tensor_like of float
        Alternative probability of success in each trial (0 < p < 1).
    n : tensor_like of float
        Alternative number of target success trials (n > 0)
    """
    rv_op = nbinom

    @classmethod
    def dist(cls, mu=None, alpha=None, p=None, n=None, *args, **kwargs):
        n, p = cls.get_n_p(mu=mu, alpha=alpha, p=p, n=n)
        n = pt.as_tensor_variable(floatX(n))
        p = pt.as_tensor_variable(floatX(p))
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
        mu = pt.floor(n * (1 - p) / p)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    def logp(value, n, p):
        alpha = n
        mu = alpha * (1 - p) / p

        res = pt.switch(
            pt.lt(value, 0),
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
        return pt.switch(pt.gt(alpha, 1e10), logp(Poisson.dist(mu=mu), value), negbinom)

    def logcdf(value, n, p):
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log(pt.betainc(n, pt.floor(value) + 1, p)),
        )
        return check_parameters(
            res,
            n > 0,
            0 <= p,
            p <= 1,
            msg="n > 0, 0 <= p <= 1",
        )


class Geometric(Discrete):
    R"""
    Geometric log-likelihood.

    The probability that the first success in a sequence of Bernoulli
    trials occurs on the x'th trial.
    The pmf of this distribution is

    .. math:: f(x \mid p) = p(1-p)^{x-1}

    .. plot::
        :context: close-figs

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
    p : tensor_like of float
        Probability of success on an individual trial (0 < p <= 1).
    """

    rv_op = geometric

    @classmethod
    def dist(cls, p, *args, **kwargs):
        p = pt.as_tensor_variable(floatX(p))
        return super().dist([p], *args, **kwargs)

    def moment(rv, size, p):
        mean = pt.round(1.0 / p)
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, p):
        res = pt.switch(
            pt.lt(value, 1),
            -np.inf,
            pt.log(p) + logpow(1 - p, value - 1),
        )

        return check_parameters(
            res,
            0 <= p,
            p <= 1,
            msg="0 <= p <= 1",
        )

    def logcdf(value, p):
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log1mexp(pt.log1p(-p) * value),
        )
        return check_parameters(
            res,
            0 <= p,
            p <= 1,
            msg="0 <= p <= 1",
        )

    def icdf(value, p):
        res = pt.ceil(pt.log1p(-value) / pt.log1p(-p)).astype("int64")
        res_1m = pt.maximum(res - 1, 0)
        dist = pm.Geometric.dist(p=p)
        value_1m = pt.exp(logcdf(dist, res_1m))
        res = pt.switch(value_1m >= value, res_1m, res)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
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
        :context: close-figs

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
            plt.plot(x, pmf, '-o', label='N = {}, k = {}, n = {}'.format(N, k, n))
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
    N : tensor_like of int
        Total size of the population (N > 0)
    k : tensor_like of int
        Number of successful individuals in the population (0 <= k <= N)
    n : tensor_like of int
        Number of samples drawn from the population (0 <= n <= N)
    """

    rv_op = hypergeometric

    @classmethod
    def dist(cls, N, k, n, *args, **kwargs):
        good = pt.as_tensor_variable(intX(k))
        bad = pt.as_tensor_variable(intX(N - k))
        n = pt.as_tensor_variable(intX(n))
        return super().dist([good, bad, n], *args, **kwargs)

    def moment(rv, size, good, bad, n):
        N, k = good + bad, good
        mode = pt.floor((n + 1) * (k + 1) / (N + 2))
        if not rv_size_is_none(size):
            mode = pt.full(size, mode)
        return mode

    def logp(value, good, bad, n):
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
        lower = pt.switch(pt.gt(n - tot + good, 0), n - tot + good, 0)
        upper = pt.switch(pt.lt(good, n), good, n)

        res = pt.switch(
            pt.lt(value, lower),
            -np.inf,
            pt.switch(
                pt.le(value, upper),
                result,
                -np.inf,
            ),
        )

        return check_parameters(
            res,
            lower <= upper,
            msg="lower <= upper",
        )

    def logcdf(value, good, bad, n):
        # logcdf can only handle scalar values at the moment
        if np.ndim(value):
            raise TypeError(
                f"HyperGeometric.logcdf expects a scalar value but received a {np.ndim(value)}-dimensional object."
            )

        N = good + bad
        # TODO: Use lower upper in locgdf for smarter logsumexp?
        safe_lower = pt.switch(pt.lt(value, 0), value, 0)

        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.switch(
                pt.lt(value, n),
                pt.logsumexp(
                    HyperGeometric.logp(pt.arange(safe_lower, value + 1), good, bad, n),
                    keepdims=False,
                ),
                0,
            ),
        )

        return check_parameters(
            res,
            N > 0,
            0 <= good,
            good <= N,
            0 <= n,
            n <= N,
            msg="N > 0, 0 <= good <= N, 0 <= n <= N",
        )


class DiscreteUniformRV(ScipyRandomVariable):
    name = "discrete_uniform"
    ndim_supp = 0
    ndims_params = [0, 0]
    dtype = "int64"
    _print_name = ("DiscreteUniform", "\\operatorname{DiscreteUniform}")

    @classmethod
    def rng_fn_scipy(cls, rng, lower, upper, size=None):
        return stats.randint.rvs(lower, upper + 1, size=size, random_state=rng)


discrete_uniform = DiscreteUniformRV()


class DiscreteUniform(Discrete):
    R"""Discrete uniform distribution.

    The pmf of this distribution is

    .. math:: f(x \mid lower, upper) = \frac{1}{upper-lower+1}

    .. plot::
        :context: close-figs

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
    lower : tensor_like of int
        Lower limit.
    upper : tensor_like of int
        Upper limit (upper > lower).
    """

    rv_op = discrete_uniform

    @classmethod
    def dist(cls, lower, upper, *args, **kwargs):
        lower = intX(pt.floor(lower))
        upper = intX(pt.floor(upper))
        return super().dist([lower, upper], **kwargs)

    def moment(rv, size, lower, upper):
        mode = pt.maximum(pt.floor((upper + lower) / 2.0), lower)
        if not rv_size_is_none(size):
            mode = pt.full(size, mode)
        return mode

    def logp(value, lower, upper):
        res = pt.switch(
            pt.or_(pt.lt(value, lower), pt.gt(value, upper)),
            -np.inf,
            pt.fill(value, -pt.log(upper - lower + 1)),
        )
        return check_parameters(
            res,
            lower <= upper,
            msg="lower <= upper",
        )

    def logcdf(value, lower, upper):
        res = pt.switch(
            pt.le(value, lower),
            -np.inf,
            pt.switch(
                pt.lt(value, upper),
                pt.log(pt.minimum(pt.floor(value), upper) - lower + 1) - pt.log(upper - lower + 1),
                0,
            ),
        )

        return check_parameters(
            res,
            lower <= upper,
            msg="lower <= upper",
        )

    def icdf(value, lower, upper):
        res = pt.ceil(value * (upper - lower + 1)).astype("int64") + lower - 1
        res_1m = pt.maximum(res - 1, lower)
        dist = pm.DiscreteUniform.dist(lower=lower, upper=upper)
        value_1m = pt.exp(logcdf(dist, res_1m))
        res = pt.switch(value_1m >= value, res_1m, res)

        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            lower <= upper,
            msg="lower <= upper",
        )


class Categorical(Discrete):
    R"""
    Categorical log-likelihood.

    The most general discrete distribution. The pmf of this distribution is

    .. math:: f(x \mid p) = p_x

    .. plot::
        :context: close-figs

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
        p > 0 and the elements of p must sum to 1.
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

        p = pt.as_tensor_variable(p)
        if isinstance(p, TensorConstant):
            p_ = np.asarray(p.data)
            if np.any(p_ < 0):
                raise ValueError(f"Negative `p` parameters are not valid, got: {p_}")
            p_sum_ = np.sum([p_], axis=-1)
            if not np.all(np.isclose(p_sum_, 1.0)):
                warnings.warn(
                    f"`p` parameters sum to {p_sum_}, instead of 1.0. "
                    "They will be automatically rescaled. "
                    "You can rescale them directly to get rid of this warning.",
                    UserWarning,
                )
                p_ = p_ / pt.sum(p_, axis=-1, keepdims=True)
                p = pt.as_tensor_variable(p_)
        return super().dist([p], **kwargs)

    def moment(rv, size, p):
        mode = pt.argmax(p, axis=-1)
        if not rv_size_is_none(size):
            mode = pt.full(size, mode)
        return mode

    def logp(value, p):
        k = pt.shape(p)[-1]
        p_ = p
        value_clip = pt.clip(value, 0, k - 1)

        if p.ndim > 1:
            if p.ndim > value_clip.ndim:
                value_clip = pt.shape_padleft(value_clip, p_.ndim - value_clip.ndim)
            elif p.ndim < value_clip.ndim:
                p = pt.shape_padleft(p, value_clip.ndim - p_.ndim)
            pattern = (p.ndim - 1,) + tuple(range(p.ndim - 1))
            a = pt.log(
                pt.take_along_axis(
                    p.dimshuffle(pattern),
                    value_clip,
                )
            )
        else:
            a = pt.log(p[value_clip])

        res = pt.switch(
            pt.or_(pt.lt(value, 0), pt.gt(value, k - 1)),
            -np.inf,
            a,
        )

        return check_parameters(
            res,
            0 <= p_,
            p_ <= 1,
            pt.isclose(pt.sum(p, axis=-1), 1),
            msg="0 <= p <=1, sum(p) = 1",
        )


class _OrderedLogistic(Categorical):
    r"""
    Underlying class for ordered logistic distributions.
    See docs for the OrderedLogistic wrapper class for more details on how to use it in models.
    """
    rv_op = categorical

    @classmethod
    def dist(cls, eta, cutpoints, *args, **kwargs):
        eta = pt.as_tensor_variable(floatX(eta))
        cutpoints = pt.as_tensor_variable(cutpoints)

        pa = sigmoid(cutpoints - pt.shape_padright(eta))
        p_cum = pt.concatenate(
            [
                pt.zeros_like(pt.shape_padright(pa[..., 0])),
                pa,
                pt.ones_like(pt.shape_padright(pa[..., 0])),
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
    eta : tensor_like of float
        The predictor.
    cutpoints : tensor_like of array
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
        eta = pt.as_tensor_variable(floatX(eta))
        cutpoints = pt.as_tensor_variable(cutpoints)

        probits = pt.shape_padright(eta) - cutpoints
        _log_p = pt.concatenate(
            [
                pt.shape_padright(normal_lccdf(0, sigma, probits[..., 0])),
                log_diff_normal_cdf(
                    0, pt.shape_padright(sigma), probits[..., :-1], probits[..., 1:]
                ),
                pt.shape_padright(normal_lcdf(0, sigma, probits[..., -1])),
            ],
            axis=-1,
        )
        _log_p = pt.as_tensor_variable(floatX(_log_p))
        p = pt.exp(_log_p)

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
    eta : tensor_like of float
        The predictor.
    cutpoints : tensor_like array of floats
        The length K - 1 array of cutpoints which break :math:`\eta` into
        ranges. Do not explicitly set the first and last elements of
        :math:`c` to negative and positive infinity.
    sigma : tensor_like of float, default 1.0
         Standard deviation of the probit function.
    compute_p : boolean, default True
        Whether to compute and store in the trace the inferred probabilities of each categories,
        based on the cutpoints' values. Defaults to True.
        Might be useful to disable it if memory usage is of interest.

    Examples
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
