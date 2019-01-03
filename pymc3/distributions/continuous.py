# coding: utf-8
"""
A collection of common probability distributions for stochastic
nodes in PyMC.
"""
import numpy as np
import theano.tensor as tt
from scipy import stats
from scipy.special import expit
from scipy.interpolate import InterpolatedUnivariateSpline
import warnings

from pymc3.theanof import floatX
from . import transforms
from pymc3.util import get_variable_name
from .special import log_i0
from ..math import invlogit, logit, logdiffexp
from .dist_math import (
    alltrue_elemwise, betaln, bound, gammaln, i0e, incomplete_beta, logpow,
    normal_lccdf, normal_lcdf, SplineWrapper, std_cdf, zvalue,
)
from .distribution import (Continuous, draw_values, generate_samples,
                           broadcast_distribution_samples)

__all__ = ['Uniform', 'Flat', 'HalfFlat', 'Normal', 'TruncatedNormal', 'Beta',
           'Kumaraswamy', 'Exponential', 'Laplace', 'StudentT', 'Cauchy',
           'HalfCauchy', 'Gamma', 'Weibull', 'HalfStudentT', 'Lognormal',
           'ChiSquared', 'HalfNormal', 'Wald', 'Pareto', 'InverseGamma',
           'ExGaussian', 'VonMises', 'SkewNormal', 'Triangular', 'Gumbel',
           'Logistic', 'LogitNormal', 'Interpolated', 'Rice']


class PositiveContinuous(Continuous):
    """Base class for positive continuous distributions"""

    def __init__(self, transform=transforms.log, *args, **kwargs):
        super().__init__(transform=transform, *args, **kwargs)


class UnitContinuous(Continuous):
    """Base class for continuous distributions on [0,1]"""

    def __init__(self, transform=transforms.logodds, *args, **kwargs):
        super().__init__(transform=transform, *args, **kwargs)


class BoundedContinuous(Continuous):
    """Base class for bounded continuous distributions"""

    def __init__(self, transform='auto', lower=None, upper=None,
                 *args, **kwargs):

        lower = tt.as_tensor_variable(lower) if lower is not None else None
        upper = tt.as_tensor_variable(upper) if upper is not None else None

        if transform == 'auto':
            if lower is None and upper is None:
                transform = None
            elif lower is not None and upper is None:
                transform = transforms.lowerbound(lower)
            elif lower is None and upper is not None:
                transform = transforms.upperbound(upper)
            else:
                transform = transforms.interval(lower, upper)

        super().__init__(transform=transform, *args, **kwargs)


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
        msg = "The variable specified for {0} has negative support for {1}, ".format(
            label, distname)
        msg += "likely making it unsuitable for this parameter."
        warnings.warn(msg)


def get_tau_sigma(tau=None, sigma=None):
    """
    Find precision and standard deviation. The link between the two
    parameterizations is given by the inverse relationship:

    .. math::
        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    tau : array-like, optional
    sigma : array-like, optional

    Results
    -------
    Returns tuple (tau, sigma)

    Notes
    -----
    If neither tau nor sigma is provided, returns (1., 1.)
    """
    if tau is None:
        if sigma is None:
            sigma = 1.
            tau = 1.
        else:
            tau = sigma**-2.

    else:
        if sigma is not None:
            raise ValueError("Can't pass both tau and sigma")
        else:
            sigma = tau**-.5

    # cast tau and sigma to float in a way that works for both np.arrays
    # and pure python
    tau = 1. * tau
    sigma = 1. * sigma

    return floatX(tau), floatX(sigma)


class Uniform(BoundedContinuous):
    R"""
    Continuous uniform log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid lower, upper) = \frac{1}{upper-lower}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-3, 3, 500)
        ls = [0., -2]
        us = [2., 1]
        for l, u in zip(ls, us):
            y = np.zeros(500)
            y[(x<u) & (x>l)] = 1.0/(u-l)
            plt.plot(x, y, label='lower = {}, upper = {}'.format(l, u))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 1)
        plt.legend(loc=1)
        plt.show()

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

    def __init__(self, lower=0, upper=1, *args, **kwargs):
        self.lower = lower = tt.as_tensor_variable(floatX(lower))
        self.upper = upper = tt.as_tensor_variable(floatX(upper))
        self.mean = (upper + lower) / 2.
        self.median = self.mean

        super().__init__(lower=lower, upper=upper, *args, **kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from Uniform distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """

        lower, upper = draw_values([self.lower, self.upper],
                                   point=point, size=size)
        return generate_samples(stats.uniform.rvs, loc=lower,
                                scale=upper - lower,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Uniform distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        lower = self.lower
        upper = self.upper
        return bound(-tt.log(upper - lower),
                     value >= lower, value <= upper)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        lower = dist.lower
        upper = dist.upper
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Uniform}}(\mathit{{lower}}={},~\mathit{{upper}}={})$'.format(
            name, get_variable_name(lower), get_variable_name(upper))

    def logcdf(self, value):
        return tt.switch(
            tt.or_(tt.lt(value, self.lower), tt.gt(value, self.upper)),
            -np.inf,
            tt.switch(
                tt.eq(value, self.upper),
                0,
                tt.log((value - self.lower)) -
                tt.log((self.upper - self.lower))
            )
        )


class Flat(Continuous):
    """
    Uninformative log-likelihood that returns 0 regardless of
    the passed value.
    """

    def __init__(self, *args, **kwargs):
        self._default = 0
        super().__init__(defaults=('_default',), *args, **kwargs)

    def random(self, point=None, size=None):
        """Raises ValueError as it is not possible to sample from Flat distribution

        Parameters
        ----------
        point : dict, optional
        size : int, optional

        Raises
        -------
        ValueError
        """
        raise ValueError('Cannot sample from Flat distribution')

    def logp(self, value):
        """
        Calculate log-probability of Flat distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        return tt.zeros_like(value)

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Flat}}()$'.format(name)

    def logcdf(self, value):
        return tt.switch(
            tt.eq(value, -np.inf),
            -np.inf,
            tt.switch(
                tt.eq(value, np.inf),
                0,
                tt.log(0.5)
            )
        )


class HalfFlat(PositiveContinuous):
    """Improper flat prior over the positive reals."""

    def __init__(self, *args, **kwargs):
        self._default = 1
        super().__init__(defaults=('_default',), *args, **kwargs)

    def random(self, point=None, size=None):
        """Raises ValueError as it is not possible to sample from HalfFlat distribution

        Parameters
        ----------
        point : dict, optional
        size : int, optional

        Raises
        -------
        ValueError
        """
        raise ValueError('Cannot sample from HalfFlat distribution')

    def logp(self, value):
        """
        Calculate log-probability of HalfFlat distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        return bound(tt.zeros_like(value), value > 0)

    def _repr_latex_(self, name=None, dist=None):
        name = r'\text{%s}' % name
        return r'${} \sim \text{{HalfFlat}}()$'.format(name)

    def logcdf(self, value):
        return tt.switch(
            tt.lt(value, np.inf),
            -np.inf,
            tt.switch(
                tt.eq(value, np.inf),
                0,
                -np.inf
            )
        )


class Normal(Continuous):
    R"""
    Univariate normal log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau) =
           \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    Normal distribution can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-5, 5, 1000)
        mus = [0., 0., 0., -2.]
        sigmas = [0.4, 1., 2., 0.4]
        for mu, sigma in zip(mus, sigmas):
            pdf = st.norm.pdf(x, mu, sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{1}{\tau}` or :math:`\sigma^2`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0) (only required if tau is not specified).
    tau : float
        Precision (tau > 0) (only required if sigma is not specified).

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.Normal('x', mu=0, sigma=10)

        with pm.Model():
            x = pm.Normal('x', mu=0, tau=1/23)
    """

    def __init__(self, mu=0, sigma=None, tau=None, sd=None, **kwargs):
        if sd is not None:
            sigma = sd
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.sigma = self.sd = tt.as_tensor_variable(sigma)
        self.tau = tt.as_tensor_variable(tau)

        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.variance = 1. / self.tau

        assert_negative_support(sigma, 'sigma', 'Normal')
        assert_negative_support(tau, 'tau', 'Normal')

        super().__init__(**kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from Normal distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, tau, _ = draw_values([self.mu, self.tau, self.sigma],
                                 point=point, size=size)
        return generate_samples(stats.norm.rvs, loc=mu, scale=tau**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Normal distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        sigma = self.sigma
        tau = self.tau
        mu = self.mu

        return bound((-tau * (value - mu)**2 + tt.log(tau / np.pi / 2.)) / 2.,
                     sigma > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        sigma = dist.sigma
        mu = dist.mu
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Normal}}(\mathit{{mu}}={},~\mathit{{sigma}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(sigma))

    def logcdf(self, value):
        return normal_lcdf(self.mu, self.sigma, value)


class TruncatedNormal(BoundedContinuous):
    R"""
    Univariate truncated normal log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x;\mu ,\sigma ,a,b)={\frac {\phi ({\frac {x-\mu }{\sigma }})}{
       \sigma \left(\Phi ({\frac {b-\mu }{\sigma }})-\Phi ({\frac {a-\mu }{\sigma }})\right)}}

    Truncated normal distribution can be parameterized either in terms of precision
    or standard deviation. The link between the two parametrizations is
    given by

    .. math::

       \tau = \dfrac{1}{\sigma^2}


    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-10, 10, 1000)
        mus = [0.,  0., 0.]
        sigmas = [3.,5.,7.]
        a1 = [-3, -5, -5]
        b1 = [7, 5, 4]
        for mu, sigma, a, b in zip(mus, sigmas,a1,b1):
            print mu, sigma, a, b
            an, bn = (a - mu) / sigma, (b - mu) / sigma
            pdf = st.truncnorm.pdf(x, an,bn, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, a={}, b={}'.format(mu, sigma, a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [a, b]`
    Mean      :math:`\mu +{\frac {\phi (\alpha )-\phi (\beta )}{Z}}\sigma`
    Variance  :math:`\sigma ^{2}\left[1+{\frac {\alpha \phi (\alpha )-\beta \phi (\beta )}{Z}}-
    \left({\frac {\phi (\alpha )-\phi (\beta )}{Z}}\right)^{2}\right]`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0).
    lower : float (optional)
        Left bound.
    upper : float (optional)
        Right bound.

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.TruncatedNormal('x', mu=0, sigma=10, lower=0)

        with pm.Model():
            x = pm.TruncatedNormal('x', mu=0, sigma=10, upper=1)

        with pm.Model():
            x = pm.TruncatedNormal('x', mu=0, sigma=10, lower=0, upper=1)

    """

    def __init__(self, mu=0, sigma=None, tau=None, lower=None, upper=None,
                 transform='auto', sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.sigma = self.sd = tt.as_tensor_variable(sigma)
        self.tau = tt.as_tensor_variable(tau)
        self.lower = tt.as_tensor_variable(floatX(lower)) if lower is not None else lower
        self.upper = tt.as_tensor_variable(floatX(upper)) if upper is not None else upper
        self.mu = tt.as_tensor_variable(floatX(mu))

        if self.lower is None and self.upper is None:
            self._defaultval = mu
        elif self.lower is None and self.upper is not None:
            self._defaultval = self.upper - 1.
        elif self.lower is not None and self.upper is None:
            self._defaultval = self.lower + 1.
        else:
            self._defaultval = (self.lower + self.upper) / 2

        assert_negative_support(sigma, 'sigma', 'TruncatedNormal')
        assert_negative_support(tau, 'tau', 'TruncatedNormal')

        super().__init__(defaults=('_defaultval',), transform=transform,
                         lower=lower, upper=upper, *args, **kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from TruncatedNormal distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu_v, std_v, a_v, b_v = draw_values(
            [self.mu, self.sigma, self.lower, self.upper], point=point, size=size)
        return generate_samples(stats.truncnorm.rvs,
                                a=(a_v - mu_v)/std_v,
                                b=(b_v - mu_v) / std_v,
                                loc=mu_v,
                                scale=std_v,
                                dist_shape=self.shape,
                                size=size,
                                )

    def logp(self, value):
        """
        Calculate log-probability of TruncatedNormal distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        mu = self.mu
        sigma = self.sigma

        norm = self._normalization()
        logp = Normal.dist(mu=mu, sigma=sigma).logp(value) - norm

        bounds = [sigma > 0]
        if self.lower is not None:
            bounds.append(value >= self.lower)
        if self.upper is not None:
            bounds.append(value <= self.upper)
        return bound(logp, *bounds)

    def _normalization(self):
        mu, sigma = self.mu, self.sigma

        if self.lower is None and self.upper is None:
            return 0.

        if self.lower is not None and self.upper is not None:
            lcdf_a = normal_lcdf(mu, sigma, self.lower)
            lcdf_b = normal_lcdf(mu, sigma, self.upper)
            lsf_a = normal_lccdf(mu, sigma, self.lower)
            lsf_b = normal_lccdf(mu, sigma, self.upper)

            return tt.switch(
                self.lower > 0,
                logdiffexp(lsf_a, lsf_b),
                logdiffexp(lcdf_b, lcdf_a),
            )

        if self.lower is not None:
            return normal_lccdf(mu, sigma, self.lower)
        else:
            return normal_lcdf(mu, sigma, self.upper)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        name = r'\text{%s}' % name
        return (
            r'${} \sim \text{{TruncatedNormal}}('
            '\mathit{{mu}}={},~\mathit{{sigma}}={},a={},b={})$'
            .format(
                name,
                get_variable_name(self.mu),
                get_variable_name(self.sigma),
                get_variable_name(self.lower),
                get_variable_name(self.upper),
            )
        )


class HalfNormal(PositiveContinuous):
    R"""
    Half-normal log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \tau) =
           \sqrt{\frac{2\tau}{\pi}}
           \exp\left(\frac{-x^2 \tau}{2}\right)

       f(x \mid \sigma) =\sigma
           \sqrt{\frac{2}{\pi}}
           \exp\left(\frac{-x^2}{2\sigma^2}\right)

    .. note::

       The parameters ``sigma``/``tau`` (:math:`\sigma`/:math:`\tau`) refer to
       the standard deviation/precision of the unfolded normal distribution, for
       the standard deviation of the half-normal distribution, see below. For
       the half-normal, they are just two parameterisation :math:`\sigma^2
       \equiv \frac{1}{\tau}` of a scale parameter

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 5, 200)
        for sigma in [0.4, 1., 2.]:
            pdf = st.halfnorm.pdf(x, scale=sigma)
            plt.plot(x, pdf, label=r'$\sigma$ = {}'.format(sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\sqrt{\dfrac{2}{\tau \pi}}` or :math:`\dfrac{\sigma \sqrt{2}}{\sqrt{\pi}}`
    Variance  :math:`\dfrac{1}{\tau}\left(1 - \dfrac{2}{\pi}\right)` or :math:`\sigma^2\left(1 - \dfrac{2}{\pi}\right)`
    ========  ==========================================

    Parameters
    ----------
    sigma : float
        Scale parameter :math:`sigma` (``sigma`` > 0) (only required if ``tau`` is not specified).
    tau : float
        Precision :math:`tau` (tau > 0) (only required if sigma is not specified).

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.HalfNormal('x', sigma=10)

        with pm.Model():
            x = pm.HalfNormal('x', tau=1/15)
    """

    def __init__(self, sigma=None, tau=None, sd=None, *args, **kwargs):
        if sd is not None:
            sigma = sd

        super().__init__(*args, **kwargs)
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        self.sigma = self.sd = sigma = tt.as_tensor_variable(sigma)
        self.tau = tau = tt.as_tensor_variable(tau)

        self.mean = tt.sqrt(2 / (np.pi * self.tau))
        self.variance = (1. - 2 / np.pi) / self.tau

        assert_negative_support(tau, 'tau', 'HalfNormal')
        assert_negative_support(sigma, 'sigma', 'HalfNormal')

    def random(self, point=None, size=None):
        """
        Draw random values from HalfNormal distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        sigma = draw_values([self.sigma], point=point)[0]
        return generate_samples(stats.halfnorm.rvs, loc=0., scale=sigma,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of HalfNormal distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        tau = self.tau
        sigma = self.sigma
        return bound(-0.5 * tau * value**2 + 0.5 * tt.log(tau * 2. / np.pi),
                     value >= 0,
                     tau > 0, sigma > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        sigma = dist.sigma
        name = r'\text{%s}' % name
        return r'${} \sim \text{{HalfNormal}}(\mathit{{sigma}}={})$'.format(name,
                                                                         get_variable_name(sigma))

    def logcdf(self, value):
        sigma = self.sigma
        z = zvalue(value, mu=0, sigma=sigma)
        return tt.switch(
            tt.lt(z, -1.0),
            tt.log(tt.erfcx(-z / tt.sqrt(2.))) - tt.sqr(z),
            tt.log1p(-tt.erfc(z / tt.sqrt(2.)))
        )


class Wald(PositiveContinuous):
    R"""
    Wald log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \lambda) =
           \left(\frac{\lambda}{2\pi)}\right)^{1/2} x^{-3/2}
           \exp\left\{
               -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2
           \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 500)
        mus = [1., 1., 1., 3.]
        lams = [1., .2, 3., 1.]
        for mu, lam in zip(mus, lams):
            pdf = st.invgauss.pdf(x, mu/lam, scale=lam)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\lambda$ = {}'.format(mu, lam))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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

    .. [Giner2016] Göknur Giner, Gordon K. Smyth (2016)
       statmod: Probability Calculations for the Inverse Gaussian Distribution
    """

    def __init__(self, mu=None, lam=None, phi=None, alpha=0., *args, **kwargs):
        super().__init__(*args, **kwargs)
        mu, lam, phi = self.get_mu_lam_phi(mu, lam, phi)
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.lam = lam = tt.as_tensor_variable(floatX(lam))
        self.phi = phi = tt.as_tensor_variable(floatX(phi))

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

    def random(self, point=None, size=None):
        """
        Draw random values from Wald distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, lam, alpha = draw_values([self.mu, self.lam, self.alpha],
                                     point=point, size=size)
        mu, lam, alpha = broadcast_distribution_samples([mu, lam, alpha],
                                                        size=size)
        return generate_samples(self._random,
                                mu, lam, alpha,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Wald distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
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

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        lam = dist.lam
        mu = dist.mu
        alpha = dist.alpha
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Wald}}(\mathit{{mu}}={},~\mathit{{lam}}={},~\mathit{{alpha}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(lam),
                                                                get_variable_name(alpha))

    def logcdf(self, value):
        # Distribution parameters
        mu = self.mu
        lam = self.lam
        alpha = self.alpha

        value -= alpha
        q = value / mu
        l = lam * mu
        r = tt.sqrt(value * lam)

        a = normal_lcdf(0, 1, (q - 1.)/r)
        b = 2./l + normal_lcdf(0, 1, -(q + 1.)/r)
        return tt.switch(
            (
                # Left limit
                tt.lt(value, 0) |
                (tt.eq(value, 0) & tt.gt(mu, 0) & tt.lt(lam, np.inf)) |
                (tt.lt(value, mu) & tt.eq(lam, 0))
            ),
            -np.inf,
            tt.switch(
                (
                    # Right limit
                    tt.eq(value, np.inf) |
                    (tt.eq(lam, 0) & tt.gt(value, mu)) |
                    (tt.gt(value, 0) & tt.eq(lam, np.inf)) |
                    # Degenerate distribution
                    (
                        tt.lt(mu, np.inf) &
                        tt.eq(mu, value) &
                        tt.eq(lam, 0)
                    ) |
                    (tt.eq(value, 0) & tt.eq(lam, np.inf))
                ),
                0,
                a + tt.log1p(tt.exp(b - a))
            )
        )


class Beta(UnitContinuous):
    R"""
    Beta log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 1, 200)
        alphas = [.5, 5., 1., 2., 2.]
        betas = [.5, 1., 3., 2., 5.]
        for a, b in zip(alphas, betas):
            pdf = st.beta.pdf(x, a, b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 4.5)
        plt.legend(loc=9)
        plt.show()

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
    sigma : float
        Alternative standard deviation (0 < sigma < sqrt(mu * (1 - mu))).

    Notes
    -----
    Beta distribution is a conjugate prior for the parameter :math:`p` of
    the binomial distribution.
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None,
                 sd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd
        alpha, beta = self.get_alpha_beta(alpha, beta, mu, sigma)
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.beta = beta = tt.as_tensor_variable(floatX(beta))

        self.mean = self.alpha / (self.alpha + self.beta)
        self.variance = self.alpha * self.beta / (
            (self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))

        assert_negative_support(alpha, 'alpha', 'Beta')
        assert_negative_support(beta, 'beta', 'Beta')

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            kappa = mu * (1 - mu) / sigma**2 - 1
            alpha = mu * kappa
            beta = (1 - mu) * kappa
        else:
            raise ValueError('Incompatible parameterization. Either use alpha '
                             'and beta, or mu and sigma to specify distribution.')

        return alpha, beta

    def random(self, point=None, size=None):
        """
        Draw random values from Beta distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point, size=size)
        return generate_samples(stats.beta.rvs, alpha, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Beta distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        beta = self.beta

        logval = tt.log(value)
        log1pval = tt.log1p(-value)
        logp = (tt.switch(tt.eq(alpha, 1), 0, (alpha - 1) * logval)
                + tt.switch(tt.eq(beta, 1), 0, (beta - 1) * log1pval)
                - betaln(alpha, beta))

        return bound(logp,
                     value >= 0, value <= 1,
                     alpha > 0, beta > 0)

    def logcdf(self, value):
        value = floatX(tt.as_tensor(value))
        a = floatX(tt.as_tensor(self.alpha))
        b = floatX(tt.as_tensor(self.beta))
        return tt.switch(
            tt.le(value, 0),
            -np.inf,
            tt.switch(
                tt.ge(value, 1),
                0,
                tt.log(incomplete_beta(a, b, value))
            )
        )

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        alpha = dist.alpha
        beta = dist.beta
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Beta}}(\mathit{{alpha}}={},~\mathit{{beta}}={})$'.format(name,
                                                                get_variable_name(alpha),
                                                                get_variable_name(beta))

class Kumaraswamy(UnitContinuous):
    R"""
    Kumaraswamy log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid a, b) =
           abx^{a-1}(1-x^a)^{b-1}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 1, 200)
        a_s = [.5, 5., 1., 2., 2.]
        b_s = [.5, 1., 3., 2., 5.]
        for a, b in zip(a_s, b_s):
            pdf = a * b * x ** (a - 1) * (1 - x ** a) ** (b - 1)
            plt.plot(x, pdf, label=r'$a$ = {}, $b$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 3.)
        plt.legend(loc=9)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (0, 1)`
    Mean      :math:`b B(1 + \tfrac{1}{a}, b)`
    Variance  :math:`b B(1 + \tfrac{2}{a}, b) - (b B(1 + \tfrac{1}{a}, b))^2`
    ========  ==============================================================

    Parameters
    ----------
    a : float
        a > 0.
    b : float
        b > 0.
    """

    def __init__(self, a, b, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.a = a = tt.as_tensor_variable(floatX(a))
        self.b = b = tt.as_tensor_variable(floatX(b))

        ln_mean = tt.log(b) + tt.gammaln(1 + 1 / a) + tt.gammaln(b) - tt.gammaln(1 + 1 / a + b)
        self.mean = tt.exp(ln_mean)
        ln_2nd_raw_moment = tt.log(b) + tt.gammaln(1 + 2 / a) + tt.gammaln(b) - tt.gammaln(1 + 2 / a + b)
        self.variance = tt.exp(ln_2nd_raw_moment) - self.mean ** 2

        assert_negative_support(a, 'a', 'Kumaraswamy')
        assert_negative_support(b, 'b', 'Kumaraswamy')

    def _random(self, a, b, size=None):
        u = np.random.uniform(size=size)
        return (1 - (1 - u) ** (1 / b)) ** (1 / a)

    def random(self, point=None, size=None):
        """
        Draw random values from Kumaraswamy distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        a, b = draw_values([self.a, self.b],
                           point=point, size=size)
        a, b = broadcast_distribution_samples([a, b], size=size)
        return generate_samples(self._random, a, b,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Kumaraswamy distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        a = self.a
        b = self.b

        logp = tt.log(a) + tt.log(b) + (a - 1) * tt.log(value) + (b - 1) * tt.log(1 - value ** a)

        return bound(logp,
                     value >= 0, value <= 1,
                     a > 0, b > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        a = dist.a
        b = dist.b
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Kumaraswamy}}(\mathit{{a}}={},~\mathit{{b}}={})$'.format(name,
                                                                                          get_variable_name(a),
                                                                                          get_variable_name(b))


class Exponential(PositiveContinuous):
    R"""
    Exponential log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \lambda) = \lambda \exp\left\{ -\lambda x \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 100)
        for lam in [0.5, 1., 2.]:
            pdf = st.expon.pdf(x, scale=1.0/lam)
            plt.plot(x, pdf, label=r'$\lambda$ = {}'.format(lam))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.lam = lam = tt.as_tensor_variable(floatX(lam))
        self.mean = 1. / self.lam
        self.median = self.mean * tt.log(2)
        self.mode = tt.zeros_like(self.lam)

        self.variance = self.lam**-2

        assert_negative_support(lam, 'lam', 'Exponential')

    def random(self, point=None, size=None):
        """
        Draw random values from Exponential distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        lam = draw_values([self.lam], point=point, size=size)[0]
        return generate_samples(np.random.exponential, scale=1. / lam,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Exponential distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        lam = self.lam
        return bound(tt.log(lam) - lam * value, value >= 0, lam > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        lam = dist.lam
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Exponential}}(\mathit{{lam}}={})$'.format(name,
                                                                get_variable_name(lam))

    def logcdf(self, value):
        """
        Compute the log CDF for the Exponential distribution

        References
        ----------
        .. [Machler2012] Martin Mächler (2012).
            "Accurately computing log(1-exp(-|a|)) Assessed by the Rmpfr
            package"
        """
        value = floatX(tt.as_tensor(value))
        lam = self.lam
        a = lam * value
        return tt.switch(
            tt.le(value, 0.0),
            -np.inf,
            tt.switch(
                tt.le(a, tt.log(2.0)),
                tt.log(-tt.expm1(-a)),
                tt.log1p(-tt.exp(-a)),
            )
        )


class Laplace(Continuous):
    R"""
    Laplace log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, b) =
           \frac{1}{2b} \exp \left\{ - \frac{|x - \mu|}{b} \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-10, 10, 1000)
        mus = [0., 0., 0., -5.]
        bs = [1., 2., 4., 4.]
        for mu, b in zip(mus, bs):
            pdf = st.laplace.pdf(x, loc=mu, scale=b)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $b$ = {}'.format(mu, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.b = b = tt.as_tensor_variable(floatX(b))
        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(floatX(mu))

        self.variance = 2 * self.b**2

        assert_negative_support(b, 'b', 'Laplace')

    def random(self, point=None, size=None):
        """
        Draw random values from Laplace distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, b = draw_values([self.mu, self.b], point=point, size=size)
        return generate_samples(np.random.laplace, mu, b,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Laplace distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        mu = self.mu
        b = self.b

        return -tt.log(2 * b) - abs(value - mu) / b

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        b = dist.b
        mu = dist.mu
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Laplace}}(\mathit{{mu}}={},~\mathit{{b}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(b))

    def logcdf(self, value):
        a = self.mu
        b = self.b
        y = (value - a) / b
        return tt.switch(
            tt.le(value, a),
            tt.log(0.5) + y,
            tt.switch(
                tt.gt(y, 1),
                tt.log1p(-0.5 * tt.exp(-y)),
                tt.log(1 - 0.5 * tt.exp(-y))
            )
        )


class Lognormal(PositiveContinuous):
    R"""
    Log-normal log-likelihood.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau) =
           \frac{1}{x} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 100)
        mus = [0., 0., 0.]
        sigmas = [.25, .5, 1.]
        for mu, sigma in zip(mus, sigmas):
            pdf = st.lognorm.pdf(x, sigma, scale=np.exp(mu))
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =========================================================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\exp\{\mu + \frac{1}{2\tau}\}`
    Variance  :math:`(\exp\{\frac{1}{\tau}\} - 1) \times \exp\{2\mu + \frac{1}{\tau}\}`
    ========  =========================================================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Standard deviation. (sigma > 0). (only required if tau is not specified).
    tau : float
        Scale parameter (tau > 0). (only required if sigma is not specified).

    Example
    -------
    .. code-block:: python

        # Example to show that we pass in only `sigma` or `tau` but not both.
        with pm.Model():
            x = pm.Lognormal('x', mu=2, sigma=30)

        with pm.Model():
            x = pm.Lognormal('x', mu=2, tau=1/100)
    """

    def __init__(self, mu=0, sigma=None, tau=None, sd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd

        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.tau = tau = tt.as_tensor_variable(tau)
        self.sigma = self.sd = sigma = tt.as_tensor_variable(sigma)

        self.mean = tt.exp(self.mu + 1. / (2 * self.tau))
        self.median = tt.exp(self.mu)
        self.mode = tt.exp(self.mu - 1. / self.tau)
        self.variance = (tt.exp(1. / self.tau) - 1) * tt.exp(2 * self.mu + 1. / self.tau)

        assert_negative_support(tau, 'tau', 'Lognormal')
        assert_negative_support(sigma, 'sigma', 'Lognormal')

    def _random(self, mu, tau, size=None):
        samples = np.random.normal(size=size)
        return np.exp(mu + (tau**-0.5) * samples)

    def random(self, point=None, size=None):
        """
        Draw random values from Lognormal distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, tau = draw_values([self.mu, self.tau], point=point, size=size)
        mu, tau = broadcast_distribution_samples([mu, tau], size=size)
        return generate_samples(self._random, mu, tau,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Lognormal distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        mu = self.mu
        tau = self.tau
        return bound(-0.5 * tau * (tt.log(value) - mu)**2
                     + 0.5 * tt.log(tau / (2. * np.pi))
                     - tt.log(value),
                     tau > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        tau = dist.tau
        mu = dist.mu
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Lognormal}}(\mathit{{mu}}={},~\mathit{{tau}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(tau))

    def logcdf(self, value):
        mu = self.mu
        sigma = self.sigma
        z = zvalue(tt.log(value), mu=mu, sigma=sigma)

        return tt.switch(
            tt.le(value, 0),
            -np.inf,
            tt.switch(
                tt.lt(z, -1.0),
                tt.log(tt.erfcx(-z / tt.sqrt(2.)) / 2.) -
                tt.sqr(z) / 2,
                tt.log1p(-tt.erfc(z / tt.sqrt(2.)) / 2.)
            )
        )


class StudentT(Continuous):
    R"""
    Student's T log-likelihood.

    Describes a normal variable whose precision is gamma distributed.
    If only nu parameter is passed, this specifies a standard (central)
    Student's T.

    The pdf of this distribution is

    .. math::

       f(x|\mu,\lambda,\nu) =
           \frac{\Gamma(\frac{\nu + 1}{2})}{\Gamma(\frac{\nu}{2})}
           \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
           \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-8, 8, 200)
        mus = [0., 0., -2., -2.]
        sigmas = [1., 1., 1., 2.]
        dfs = [1., 5., 5., 5.]
        for mu, sigma, df in zip(mus, sigmas, dfs):
            pdf = st.t.pdf(x, df, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, $\nu$ = {}'.format(mu, sigma, df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    ========  ========================

    Parameters
    ----------
    nu : float
        Degrees of freedom, also known as normality parameter (nu > 0).
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases. (only required if lam is not specified)
    lam : float
        Scale parameter (lam > 0). Converges to the precision as nu
        increases. (only required if sigma is not specified)

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.StudentT('x', nu=15, mu=0, sigma=10)

        with pm.Model():
            x = pm.StudentT('x', nu=15, mu=0, lam=1/23)
    """

    def __init__(self, nu, mu=0, lam=None, sigma=None, sd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(StudentT, self).__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd
        self.nu = nu = tt.as_tensor_variable(floatX(nu))
        lam, sigma = get_tau_sigma(tau=lam, sigma=sigma)
        self.lam = lam = tt.as_tensor_variable(lam)
        self.sigma = self.sd = sigma = tt.as_tensor_variable(sigma)
        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(mu)

        self.variance = tt.switch((nu > 2) * 1,
                                  (1 / self.lam) * (nu / (nu - 2)),
                                  np.inf)

        assert_negative_support(lam, 'lam (sigma)', 'StudentT')
        assert_negative_support(nu, 'nu', 'StudentT')

    def random(self, point=None, size=None):
        """
        Draw random values from StudentT distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        nu, mu, lam = draw_values([self.nu, self.mu, self.lam],
                                  point=point, size=size)
        return generate_samples(stats.t.rvs, nu, loc=mu, scale=lam**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of StudentT distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        nu = self.nu
        mu = self.mu
        lam = self.lam
        sigma = self.sigma

        return bound(gammaln((nu + 1.0) / 2.0)
                     + .5 * tt.log(lam / (nu * np.pi))
                     - gammaln(nu / 2.0)
                     - (nu + 1.0) / 2.0 * tt.log1p(lam * (value - mu)**2 / nu),
                     lam > 0, nu > 0, sigma > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        nu = dist.nu
        mu = dist.mu
        lam = dist.lam
        name = r'\text{%s}' % name
        return r'${} \sim \text{{StudentT}}(\mathit{{nu}}={},~\mathit{{mu}}={},~\mathit{{lam}}={})$'.format(name,
                                                                get_variable_name(nu),
                                                                get_variable_name(mu),
                                                                get_variable_name(lam))

    def logcdf(self, value):
        nu = self.nu
        mu = self.mu
        sigma = self.sigma
        t = (value - mu)/sigma
        sqrt_t2_nu = tt.sqrt(t**2 + nu)
        x = (t + sqrt_t2_nu)/(2.0 * sqrt_t2_nu)
        return tt.log(incomplete_beta(nu/2., nu/2., x))


class Pareto(Continuous):
    R"""
    Pareto log-likelihood.

    Often used to characterize wealth distribution, or other examples of the
    80/20 rule.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 4, 1000)
        alphas = [1., 2., 5., 5.]
        ms = [1., 1., 1., 2.]
        for alpha, m in zip(alphas, ms):
            pdf = st.pareto.pdf(x, alpha, scale=m)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, m = {}'.format(alpha, m))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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

    def __init__(self, alpha, m, transform='lowerbound', *args, **kwargs):
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.m = m = tt.as_tensor_variable(floatX(m))

        self.mean = tt.switch(tt.gt(alpha, 1), alpha *
                              m / (alpha - 1.), np.inf)
        self.median = m * 2.**(1. / alpha)
        self.variance = tt.switch(
            tt.gt(alpha, 2),
            (alpha * m**2) / ((alpha - 2.) * (alpha - 1.)**2),
            np.inf)

        assert_negative_support(alpha, 'alpha', 'Pareto')
        assert_negative_support(m, 'm', 'Pareto')

        if transform == 'lowerbound':
            transform = transforms.lowerbound(self.m)
        super().__init__(transform=transform, *args, **kwargs)

    def _random(self, alpha, m, size=None):
        u = np.random.uniform(size=size)
        return m * (1. - u)**(-1. / alpha)

    def random(self, point=None, size=None):
        """
        Draw random values from Pareto distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        alpha, m = draw_values([self.alpha, self.m],
                               point=point, size=size)
        alpha, m = broadcast_distribution_samples([alpha, m], size=size)
        return generate_samples(self._random, alpha, m,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Pareto distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        m = self.m
        return bound(tt.log(alpha) + logpow(m, alpha)
                     - logpow(value, alpha + 1),
                     value >= m, alpha > 0, m > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        alpha = dist.alpha
        m = dist.m
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Pareto}}(\mathit{{alpha}}={},~\mathit{{m}}={})$'.format(name,
                                                                get_variable_name(alpha),
                                                                get_variable_name(m))

    def logcdf(self, value):
        m = self.m
        alpha = self.alpha
        arg = (m / value) ** alpha
        return tt.switch(
            tt.lt(value, m),
            -np.inf,
            tt.switch(
                tt.le(arg, 1e-5),
                tt.log1p(-arg),
                tt.log(1 - arg)
            )
        )


class Cauchy(Continuous):
    R"""
    Cauchy log-likelihood.

    Also known as the Lorentz or the Breit-Wigner distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-5, 5, 500)
        alphas = [0., 0., 0., -2.]
        betas = [.5, 1., 2., 1.]
        for a, b in zip(alphas, betas):
            pdf = st.cauchy.pdf(x, loc=a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.median = self.mode = self.alpha = tt.as_tensor_variable(floatX(alpha))
        self.beta = tt.as_tensor_variable(floatX(beta))

        assert_negative_support(beta, 'beta', 'Cauchy')

    def _random(self, alpha, beta, size=None):
        u = np.random.uniform(size=size)
        return alpha + beta * np.tan(np.pi * (u - 0.5))

    def random(self, point=None, size=None):
        """
        Draw random values from Cauchy distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point, size=size)
        alpha, beta = broadcast_distribution_samples([alpha, beta], size=size)
        return generate_samples(self._random, alpha, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Cauchy distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        beta = self.beta
        return bound(- tt.log(np.pi) - tt.log(beta)
                     - tt.log1p(((value - alpha) / beta)**2),
                     beta > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        alpha = dist.alpha
        beta = dist.beta
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Cauchy}}(\mathit{{alpha}}={},~\mathit{{beta}}={})$'.format(name,
                                                                get_variable_name(alpha),
                                                                get_variable_name(beta))

    def logcdf(self, value):
        return tt.log(
            0.5 + tt.arctan((value - self.alpha) / self.beta) / np.pi
        )


class HalfCauchy(PositiveContinuous):
    R"""
    Half-Cauchy log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \beta) = \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 5, 200)
        for b in [0.5, 1.0, 2.0]:
            pdf = st.cauchy.pdf(x, scale=b)
            plt.plot(x, pdf, label=r'$\beta$ = {}'.format(b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in [0, \infty)`
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
        super().__init__(*args, **kwargs)
        self.mode = tt.as_tensor_variable(0)
        self.median = self.beta = tt.as_tensor_variable(floatX(beta))

        assert_negative_support(beta, 'beta', 'HalfCauchy')

    def _random(self, beta, size=None):
        u = np.random.uniform(size=size)
        return beta * np.abs(np.tan(np.pi * (u - 0.5)))

    def random(self, point=None, size=None):
        """
        Draw random values from HalfCauchy distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        beta = draw_values([self.beta], point=point, size=size)[0]
        return generate_samples(self._random, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of HalfCauchy distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        beta = self.beta
        return bound(tt.log(2) - tt.log(np.pi) - tt.log(beta)
                     - tt.log1p((value / beta)**2),
                     value >= 0, beta > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        beta = dist.beta
        name = r'\text{%s}' % name
        return r'${} \sim \text{{HalfCauchy}}(\mathit{{beta}}={})$'.format(name,
                                                                get_variable_name(beta))

    def logcdf(self, value):
        return tt.switch(
            tt.le(value, 0),
            -np.inf,
            tt.log(
                2 * tt.arctan(value / self.beta) / np.pi
            ))


class Gamma(PositiveContinuous):
    R"""
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables,
    each of which has mean beta.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 20, 200)
        alphas = [1., 2., 3., 7.5]
        betas = [.5, .5, 1., 1.]
        for a, b in zip(alphas, betas):
            pdf = st.gamma.pdf(x, a, scale=1.0/b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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
    sigma : float
        Alternative scale parameter (sigma > 0).
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None,
                 sd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd

        alpha, beta = self.get_alpha_beta(alpha, beta, mu, sigma)
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.beta = beta = tt.as_tensor_variable(floatX(beta))
        self.mean = alpha / beta
        self.mode = tt.maximum((alpha - 1) / beta, 0)
        self.variance = alpha / beta**2

        assert_negative_support(alpha, 'alpha', 'Gamma')
        assert_negative_support(beta, 'beta', 'Gamma')

    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            alpha = mu**2 / sigma**2
            beta = mu / sigma**2
        else:
            raise ValueError('Incompatible parameterization. Either use '
                             'alpha and beta, or mu and sigma to specify '
                             'distribution.')

        return alpha, beta

    def random(self, point=None, size=None):
        """
        Draw random values from Gamma distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point, size=size)
        return generate_samples(stats.gamma.rvs, alpha, scale=1. / beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Gamma distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        beta = self.beta
        return bound(
            -gammaln(alpha) + logpow(
                beta, alpha) - beta * value + logpow(value, alpha - 1),
            value >= 0,
            alpha > 0,
            beta > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        beta = dist.beta
        alpha = dist.alpha
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Gamma}}(\mathit{{alpha}}={},~\mathit{{beta}}={})$'.format(name,
                                                                get_variable_name(alpha),
                                                                get_variable_name(beta))


class InverseGamma(PositiveContinuous):
    R"""
    Inverse gamma log-likelihood, the reciprocal of the gamma distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 500)
        alphas = [1., 2., 3., 3.]
        betas = [1., 1., 1., .5]
        for a, b in zip(alphas, betas):
            pdf = st.invgamma.pdf(x, a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ======================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
    Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha - 2)}`
              for :math:`\alpha > 2`
    ========  ======================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    mu : float
        Alternative shape parameter (mu > 0).
    sigma : float
        Alternative scale parameter (sigma > 0).
    """

    def __init__(self, alpha=None, beta=None, mu=None, sigma=None, sd=None,
                 *args, **kwargs):
        super().__init__(*args, defaults=('mode',), **kwargs)

        if sd is not None:
            sigma = sd

        alpha, beta = InverseGamma._get_alpha_beta(alpha, beta, mu, sigma)
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.beta = beta = tt.as_tensor_variable(floatX(beta))

        self.mean = self._calculate_mean()
        self.mode = beta / (alpha + 1.)
        self.variance = tt.switch(tt.gt(alpha, 2),
                                  (beta**2) / ((alpha - 2) * (alpha - 1.)**2),
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

    @staticmethod
    def _get_alpha_beta(alpha, beta, mu, sigma):
        if (alpha is not None):
            if (beta is not None):
                pass
            else:
                beta = 1
        elif (mu is not None) and (sigma is not None):
            alpha = (2 * sigma**2 + mu**2)/sigma**2
            beta = mu * (mu**2 + sigma**2) / sigma**2
        else:
            raise ValueError('Incompatible parameterization. Either use '
                             'alpha and (optionally) beta, or mu and sigma to specify '
                             'distribution.')

        return alpha, beta

    def random(self, point=None, size=None):
        """
        Draw random values from InverseGamma distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point, size=size)
        return generate_samples(stats.invgamma.rvs, a=alpha, scale=beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of InverseGamma distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        beta = self.beta
        return bound(logpow(beta, alpha) - gammaln(alpha) - beta / value
                     + logpow(value, -alpha - 1),
                     value > 0, alpha > 0, beta > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        beta = dist.beta
        alpha = dist.alpha
        name = r'\text{%s}' % name
        return r'${} \sim \text{{InverseGamma}}(\mathit{{alpha}}={},~\mathit{{beta}}={})$'.format(name,
                                                                get_variable_name(alpha),
                                                                get_variable_name(beta))


class ChiSquared(Gamma):
    R"""
    :math:`\chi^2` log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) = \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 15, 200)
        for df in [1, 2, 3, 6, 9]:
            pdf = st.chi2.pdf(x, df)
            plt.plot(x, pdf, label=r'$\nu$ = {}'.format(df))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 0.6)
        plt.legend(loc=1)
        plt.show()

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
        self.nu = nu = tt.as_tensor_variable(floatX(nu))
        super().__init__(alpha=nu / 2., beta=0.5, *args, **kwargs)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        nu = dist.nu
        name = r'\text{%s}' % name
        return r'${} \sim \Chi^2(\mathit{{nu}}={})$'.format(name,
                                                            get_variable_name(nu))


class Weibull(PositiveContinuous):
    R"""
    Weibull log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\alpha x^{\alpha - 1}
           \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 3, 200)
        alphas = [.5, 1., 1.5, 5., 5.]
        betas = [1., 1., 1., 1.,  2]
        for a, b in zip(alphas, betas):
            pdf = st.weibull_min.pdf(x, a, scale=b)
            plt.plot(x, pdf, label=r'$\alpha$ = {}, $\beta$ = {}'.format(a, b))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 2.5)
        plt.legend(loc=1)
        plt.show()

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
        super().__init__(*args, **kwargs)
        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))
        self.beta = beta = tt.as_tensor_variable(floatX(beta))
        self.mean = beta * tt.exp(gammaln(1 + 1. / alpha))
        self.median = beta * tt.exp(gammaln(tt.log(2)))**(1. / alpha)
        self.variance = (beta**2) * \
            tt.exp(gammaln(1 + 2. / alpha - self.mean**2))
        self.mode = tt.switch(alpha >= 1,
                              beta * ((alpha - 1)/alpha) ** (1 / alpha),
                              0)  # Reference: https://en.wikipedia.org/wiki/Weibull_distribution

        assert_negative_support(alpha, 'alpha', 'Weibull')
        assert_negative_support(beta, 'beta', 'Weibull')

    def random(self, point=None, size=None):
        """
        Draw random values from Weibull distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        alpha, beta = draw_values([self.alpha, self.beta],
                                  point=point, size=size)
        alpha, beta = broadcast_distribution_samples([alpha, beta], size=size)

        def _random(a, b, size=None):
            return b * (-np.log(np.random.uniform(size=size)))**(1 / a)

        return generate_samples(_random, alpha, beta,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Weibull distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        alpha = self.alpha
        beta = self.beta
        return bound(tt.log(alpha) - tt.log(beta)
                     + (alpha - 1) * tt.log(value / beta)
                     - (value / beta)**alpha,
                     value >= 0, alpha > 0, beta > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        beta = dist.beta
        alpha = dist.alpha
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Weibull}}(\mathit{{alpha}}={},~\mathit{{beta}}={})$'.format(name,
                                                                get_variable_name(alpha),
                                                                get_variable_name(beta))

    def logcdf(self, value):
        '''
        Compute the log CDF for the Weibull distribution

        References
        ----------
        .. [Machler2012] Martin Mächler (2012).
            "Accurately computing log(1-exp(-|a|)) Assessed by the Rmpfr
            package"
        '''
        alpha = self.alpha
        beta = self.beta
        a = (value / beta)**alpha
        return tt.switch(
            tt.le(value, 0.0),
            -np.inf,
            tt.switch(
                tt.le(a, tt.log(2.0)),
                tt.log(-tt.expm1(-a)),
                tt.log1p(-tt.exp(-a)))
        )


class HalfStudentT(PositiveContinuous):
    R"""
    Half Student's T log-likelihood

    The pdf of this distribution is

    .. math::

        f(x \mid \sigma,\nu) =
            \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
            {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
            \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0, 5, 200)
        sigmas = [1., 1., 2., 1.]
        nus = [.5, 1., 1., 30.]
        for sigma, nu in zip(sigmas, nus):
            pdf = st.t.pdf(x, df=nu, loc=0, scale=sigma)
            plt.plot(x, pdf, label=r'$\sigma$ = {}, $\nu$ = {}'.format(sigma, nu))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ========================
    Support   :math:`x \in [0, \infty)`
    ========  ========================

    Parameters
    ----------
    nu : float
        Degrees of freedom, also known as normality parameter (nu > 0).
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases. (only required if lam is not specified)
    lam : float
        Scale parameter (lam > 0). Converges to the precision as nu
        increases. (only required if sigma is not specified)

    Examples
    --------
    .. code-block:: python

        # Only pass in one of lam or sigma, but not both.
        with pm.Model():
            x = pm.HalfStudentT('x', sigma=10, nu=10)

        with pm.Model():
            x = pm.HalfStudentT('x', lam=4, nu=10)
    """

    def __init__(self, nu=1, sigma=None, lam=None, sd=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd

        self.mode = tt.as_tensor_variable(0)
        lam, sigma = get_tau_sigma(lam, sigma)
        self.median = tt.as_tensor_variable(sigma)
        self.sigma = self.sd = tt.as_tensor_variable(sigma)
        self.lam = tt.as_tensor_variable(lam)
        self.nu = nu = tt.as_tensor_variable(floatX(nu))

        assert_negative_support(sigma, 'sigma', 'HalfStudentT')
        assert_negative_support(lam, 'lam', 'HalfStudentT')
        assert_negative_support(nu, 'nu', 'HalfStudentT')

    def random(self, point=None, size=None):
        """
        Draw random values from HalfStudentT distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        nu, sigma = draw_values([self.nu, self.sigma], point=point, size=size)
        return np.abs(generate_samples(stats.t.rvs, nu, loc=0, scale=sigma,
                                       dist_shape=self.shape,
                                       size=size))

    def logp(self, value):
        """
        Calculate log-probability of HalfStudentT distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        nu = self.nu
        sigma = self.sigma
        lam = self.lam

        return bound(tt.log(2) + gammaln((nu + 1.0) / 2.0)
                     - gammaln(nu / 2.0)
                     - .5 * tt.log(nu * np.pi * sigma**2)
                     - (nu + 1.0) / 2.0 * tt.log1p(value ** 2 / (nu * sigma**2)),
                     sigma > 0, lam > 0, nu > 0, value >= 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        nu = dist.nu
        sigma = dist.sigma
        name = r'\text{%s}' % name
        return r'${} \sim \text{{HalfStudentT}}(\mathit{{nu}}={},~\mathit{{sigma}}={})$'.format(name,
                                                                get_variable_name(nu),
                                                                get_variable_name(sigma))


class ExGaussian(Continuous):
    R"""
    Exponentially modified Gaussian log-likelihood.

    Results from the convolution of a normal distribution with an exponential
    distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \sigma, \tau) =
           \frac{1}{\nu}\;
           \exp\left\{\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right\}
           \Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)

    where :math:`\Phi` is the cumulative distribution function of the
    standard normal distribution.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-6, 9, 200)
        mus = [0., -2., 0., -3.]
        sigmas = [1., 1., 3., 1.]
        nus = [1., 1., 1., 4.]
        for mu, sigma, nu in zip(mus, sigmas, nus):
            pdf = st.exponnorm.pdf(x, nu/sigma, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, $\nu$ = {}'.format(mu, sigma, nu))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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

    def __init__(self, mu=0., sigma=None, nu=None, sd=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if sd is not None:
            sigma = sd

        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.sigma = self.sd = sigma = tt.as_tensor_variable(floatX(sigma))
        self.nu = nu = tt.as_tensor_variable(floatX(nu))
        self.mean = mu + nu
        self.variance = (sigma**2) + (nu**2)

        assert_negative_support(sigma, 'sigma', 'ExGaussian')
        assert_negative_support(nu, 'nu', 'ExGaussian')

    def random(self, point=None, size=None):
        """
        Draw random values from ExGaussian distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, sigma, nu = draw_values([self.mu, self.sigma, self.nu],
                                    point=point, size=size)
        mu, sigma, nu = broadcast_distribution_samples([mu, sigma, nu],
                                                       size=size)

        def _random(mu, sigma, nu, size=None):
            return (np.random.normal(mu, sigma, size=size)
                    + np.random.exponential(scale=nu, size=size))

        return generate_samples(_random, mu, sigma, nu,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of ExGaussian distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
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

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        sigma = dist.sigma
        mu = dist.mu
        nu = dist.nu
        name = r'\text{%s}' % name
        return r'${} \sim \text{{ExGaussian}}(\mathit{{mu}}={},~\mathit{{sigma}}={},~\mathit{{nu}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(sigma),
                                                                get_variable_name(nu))

    def logcdf(self, value):
        """
        Compute the log CDF for the ExGaussian distribution

        References
        ----------
        .. [Rigby2005] R.A. Rigby (2005).
           "Generalized additive models for location, scale and shape"
           http://dx.doi.org/10.1111/j.1467-9876.2005.00510.x
        """
        mu = self.mu
        sigma = self.sigma
        sigma_2 = sigma**2
        nu = self.nu
        z = value - mu - sigma_2/nu
        return tt.switch(
            tt.gt(nu, 0.05 * sigma),
            tt.log(std_cdf((value - mu)/sigma) -
                   std_cdf(z/sigma) * tt.exp(
                       ((mu + (sigma_2/nu))**2 -
                        (mu**2) -
                        2 * value * ((sigma_2)/nu))/(2 * sigma_2))),
            normal_lcdf(mu, sigma, value))


class VonMises(Continuous):
    R"""
    Univariate VonMises log-likelihood.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \kappa) =
            \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    where :math:`I_0` is the modified Bessel function of order 0.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-np.pi, np.pi, 200)
        mus = [0., 0., 0.,  -2.5]
        kappas = [.01, 0.5,  4., 2.]
        for mu, kappa in zip(mus, kappas):
            pdf = st.vonmises.pdf(x, kappa, loc=mu)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\kappa$ = {}'.format(mu, kappa))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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
        if transform == 'circular':
            transform = transforms.Circular()
        super().__init__(transform=transform, *args, **kwargs)
        self.mean = self.median = self.mode = self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.kappa = kappa = tt.as_tensor_variable(floatX(kappa))

        assert_negative_support(kappa, 'kappa', 'VonMises')

    def random(self, point=None, size=None):
        """
        Draw random values from VonMises distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, kappa = draw_values([self.mu, self.kappa],
                                point=point, size=size)
        return generate_samples(stats.vonmises.rvs, loc=mu, kappa=kappa,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of VonMises distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        mu = self.mu
        kappa = self.kappa
        return bound(kappa * tt.cos(mu - value) - (tt.log(2 * np.pi) + log_i0(kappa)),
                     kappa > 0, value >= -np.pi, value <= np.pi)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        kappa = dist.kappa
        mu = dist.mu
        name = r'\text{%s}' % name
        return r'${} \sim \text{{VonMises}}(\mathit{{mu}}={},~\mathit{{kappa}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(kappa))



class SkewNormal(Continuous):
    R"""
    Univariate skew-normal log-likelihood.

     The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau, \alpha) =
       2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-4, 4, 200)
        for alpha in [-6, 0, 6]:
            pdf = st.skewnorm.pdf(x, alpha, loc=0, scale=1)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}, $\alpha$ = {}'.format(0, 1, alpha))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

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
    sigma : float
        Scale parameter (sigma > 0).
    tau : float
        Alternative scale parameter (tau > 0).
    alpha : float
        Skewness parameter.

    Notes
    -----
    When alpha=0 we recover the Normal distribution and mu becomes the mean,
    tau the precision and sigma the standard deviation. In the limit of alpha
    approaching plus/minus infinite we get a half-normal distribution.

    """

    def __init__(self, mu=0.0, sigma=None, tau=None, alpha=1, sd=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        if sd is not None:
            sigma = sd

        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        self.tau = tt.as_tensor_variable(tau)
        self.sigma = self.sd = tt.as_tensor_variable(sigma)

        self.alpha = alpha = tt.as_tensor_variable(floatX(alpha))

        self.mean = mu + self.sigma * (2 / np.pi)**0.5 * alpha / (1 + alpha**2)**0.5
        self.variance = self.sigma**2 * (1 - (2 * alpha**2) / ((1 + alpha**2) * np.pi))

        assert_negative_support(tau, 'tau', 'SkewNormal')
        assert_negative_support(sigma, 'sigma', 'SkewNormal')

    def random(self, point=None, size=None):
        """
        Draw random values from SkewNormal distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, tau, _, alpha = draw_values(
            [self.mu, self.tau, self.sigma, self.alpha], point=point, size=size)
        return generate_samples(stats.skewnorm.rvs,
                                a=alpha, loc=mu, scale=tau**-0.5,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of SkewNormal distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        tau = self.tau
        sigma = self.sigma
        mu = self.mu
        alpha = self.alpha
        return bound(
            tt.log(1 +
                   tt.erf(((value - mu) * tt.sqrt(tau) * alpha) / tt.sqrt(2)))
            + (-tau * (value - mu)**2
               + tt.log(tau / np.pi / 2.)) / 2.,
            tau > 0, sigma > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        sigma = dist.sigma
        mu = dist.mu
        alpha = dist.alpha
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Skew-Normal}}(\mathit{{mu}}={},~\mathit{{sigma}}={},~\mathit{{alpha}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(sigma),
                                                                get_variable_name(alpha))


class Triangular(BoundedContinuous):
    R"""
    Continuous Triangular log-likelihood

    The pdf of this distribution is

    .. math::

       \begin{cases}
         0 & \text{for } x < a, \\
         \frac{2(x-a)}{(b-a)(c-a)} & \text{for } a \le x < c, \\[4pt]
         \frac{2}{b-a}             & \text{for } x = c, \\[4pt]
         \frac{2(b-x)}{(b-a)(b-c)} & \text{for } c < x \le b, \\[4pt]
         0 & \text{for } b < x.
        \end{cases}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-2, 10, 500)
        lowers = [0., -1, 2]
        cs = [2., 0., 6.5]
        uppers = [4., 1, 8]
        for lower, c, upper in zip(lowers, cs, uppers):
            scale = upper - lower
            c_ = (c - lower) / scale
            pdf = st.triang.pdf(x, loc=lower, c=c_, scale=scale)
            plt.plot(x, pdf, label='lower = {}, c = {}, upper = {}'.format(lower,
                                                                           c,
                                                                           upper))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ============================================================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{lower + upper + c}{3}`
    Variance  :math:`\dfrac{upper^2 + lower^2 +c^2 - lower*upper - lower*c - upper*c}{18}`
    ========  ============================================================================

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
        self.median = self.mean = self.c = c = tt.as_tensor_variable(floatX(c))
        self.lower = lower = tt.as_tensor_variable(floatX(lower))
        self.upper = upper = tt.as_tensor_variable(floatX(upper))

        super().__init__(lower=lower, upper=upper, *args, **kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from Triangular distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        c, lower, upper = draw_values([self.c, self.lower, self.upper],
                                      point=point, size=size)
        scale = upper - lower
        c_ = (c - lower) / scale
        return generate_samples(stats.triang.rvs, c=c_, loc=lower, scale=scale,
                                size=size, dist_shape=self.shape, random_state=None)

    def logp(self, value):
        """
        Calculate log-probability of Triangular distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        c = self.c
        lower = self.lower
        upper = self.upper
        return tt.switch(alltrue_elemwise([lower <= value, value < c]),
                         tt.log(2 * (value - lower) / ((upper - lower) * (c - lower))),
                         tt.switch(tt.eq(value, c),
                                   tt.log(2 / (upper - lower)),
                                   tt.switch(alltrue_elemwise([c < value, value <= upper]),
                                             tt.log(2 * (upper - value) / ((upper - lower) * (upper - c))),
                                             np.inf)))

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        lower = dist.lower
        upper = dist.upper
        c = dist.c
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Triangular}}(\mathit{{c}}={},~\mathit{{lower}}={},~\mathit{{upper}}={})$'.format(name,
                                                                get_variable_name(c),
                                                                get_variable_name(lower),
                                                                get_variable_name(upper))

    def logcdf(self, value):
        l = self.lower
        u = self.upper
        c = self.c
        return tt.switch(
            tt.le(value, l),
            -np.inf,
            tt.switch(
                tt.le(value, c),
                tt.log(((value - l) ** 2) / ((u - l) * (c - l))),
                tt.switch(
                    tt.lt(value, u),
                    tt.log1p(-((u - value) ** 2) / ((u - l) * (u - c))),
                    0
                )
            )
        )


class Gumbel(Continuous):
    R"""
        Univariate Gumbel log-likelihood

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \beta) = \frac{1}{\beta}e^{-(z + e^{-z})}

    where

    .. math::

        z = \frac{x - \mu}{\beta}.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [0., 4., -1.]
        betas = [2., 2., 4.]
        for mu, beta in zip(mus, betas):
            pdf = st.gumbel_r.pdf(x, loc=mu, scale=beta)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\beta$ = {}'.format(mu, beta))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()


    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu + \beta\gamma`, where \gamma is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^2}{6} \beta^2`
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location parameter.
    beta : float
        Scale parameter (beta > 0).
    """

    def __init__(self, mu=0, beta=1.0, **kwargs):
        self.mu = tt.as_tensor_variable(floatX(mu))
        self.beta = tt.as_tensor_variable(floatX(beta))

        assert_negative_support(beta, 'beta', 'Gumbel')

        self.mean = self.mu + self.beta * np.euler_gamma
        self.median = self.mu - self.beta * tt.log(tt.log(2))
        self.mode = self.mu
        self.variance = (np.pi ** 2 / 6.0) * self.beta ** 2

        super().__init__(**kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from Gumbel distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, sigma = draw_values([self.mu, self.beta], point=point, size=size)
        return generate_samples(stats.gumbel_r.rvs, loc=mu, scale=sigma,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Gumbel distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        scaled = (value - self.mu) / self.beta
        return bound(-scaled - tt.exp(-scaled) - tt.log(self.beta), self.beta > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        beta = dist.beta
        mu = dist.mu
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Gumbel}}(\mathit{{mu}}={},~\mathit{{beta}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(beta))

    def logcdf(self, value):
        beta = self.beta
        mu = self.mu

        return -tt.exp(-(value - mu)/beta)


class Rice(PositiveContinuous):
    R"""
    Rice distribution.

    .. math::

       f(x\mid \nu ,\sigma )=
       {\frac  {x}{\sigma ^{2}}}\exp
       \left({\frac  {-(x^{2}+\nu ^{2})}{2\sigma ^{2}}}\right)I_{0}\left({\frac  {x\nu }{\sigma ^{2}}}\right),

    ========  ==============================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\sigma {\sqrt  {\pi /2}}\,\,L_{{1/2}}(-\nu ^{2}/2\sigma ^{2})`
    Variance  :math:`2\sigma ^{2}+\nu ^{2}-{\frac  {\pi \sigma ^{2}}{2}}L_{{1/2}}^{2}\left({\frac  {-\nu ^{2}}{2\sigma ^{2}}}\right)`
    ========  ==============================================================


    Parameters
    ----------
    nu : float
        noncentrality parameter.
    sigma : float
        scale parameter.
    b : float
        shape parameter (alternative to nu).

    Notes
    -----
    The distribution :math:`\mathrm{Rice}\left(|\nu|,\sigma\right)` is the
    distribution of :math:`R=\sqrt{X^2+Y^2}` where :math:`X\sim N(\nu \cos{\theta}, \sigma^2)`,
    :math:`Y\sim N(\nu \sin{\theta}, \sigma^2)` are independent and for any
    real :math:`\theta`.

    The distribution is defined with either nu or b.
    The link between the two parametrizations is given by

    .. math::

       b = \dfrac{\nu}{\sigma}

    """

    def __init__(self, nu=None, sigma=None, b=None, sd=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if sd is not None:
            sigma = sd

        nu, b, sigma = self.get_nu_b(nu, b, sigma)
        self.nu = nu = tt.as_tensor_variable(floatX(nu))
        self.sigma = self.sd = sigma = tt.as_tensor_variable(floatX(sigma))
        self.b = b = tt.as_tensor_variable(floatX(b))
        self.mean = sigma * np.sqrt(np.pi / 2) * tt.exp((-nu**2 / (2 * sigma**2)) / 2) * ((1 - (-nu**2 / (2 * sigma**2)))
                                 * tt.i0(-(-nu**2 / (2 * sigma**2)) / 2) - (-nu**2 / (2 * sigma**2)) * tt.i1(-(-nu**2 / (2 * sigma**2)) / 2))
        self.variance = 2 * sigma**2 + nu**2 - (np.pi * sigma**2 / 2) * (tt.exp((-nu**2 / (2 * sigma**2)) / 2) * ((1 - (-nu**2 / (
            2 * sigma**2))) * tt.i0(-(-nu**2 / (2 * sigma**2)) / 2) - (-nu**2 / (2 * sigma**2)) * tt.i1(-(-nu**2 / (2 * sigma**2)) / 2)))**2

    def get_nu_b(self, nu, b, sigma):
        if sigma is None:
            sigma = 1.
        if nu is None and b is not None:
            nu = b * sigma
            return nu, b, sigma
        elif nu is not None and b is None:
            b = nu / sigma
            return nu, b, sigma
        raise ValueError('Rice distribution must specify either nu'
                         ' or b.')

    def random(self, point=None, size=None):
        """
        Draw random values from Rice distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        nu, sigma = draw_values([self.nu, self.sigma],
                             point=point, size=size)
        return generate_samples(stats.rice.rvs, b=nu / sigma, scale=sigma, loc=0,
                                dist_shape=self.shape, size=size)

    def logp(self, value):
        """
        Calculate log-probability of Rice distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        nu = self.nu
        sigma = self.sigma
        b = self.b
        x = value / sigma
        return bound(tt.log(x * tt.exp((-(x - b) * (x - b)) / 2) * i0e(x * b) / sigma),
                     sigma >= 0,
                     nu >= 0,
                     value > 0,
                     )


class Logistic(Continuous):
    R"""
    Logistic log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, s) =
           \frac{\exp\left(-\frac{x - \mu}{s}\right)}{s \left(1 + \exp\left(-\frac{x - \mu}{s}\right)\right)^2}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(-5, 5, 200)
        mus = [0., 0., 0., -2.]
        ss = [.4, 1., 2., .4]
        for mu, s in zip(mus, ss):
            pdf = st.logistic.pdf(x, loc=mu, scale=s)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $s$ = {}'.format(mu, s))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu`
    Variance  :math:`\frac{s^2 \pi^2}{3}`
    ========  ==========================================


    Parameters
    ----------
    mu : float
        Mean.
    s : float
        Scale (s > 0).
    """

    def __init__(self, mu=0., s=1., *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mu = tt.as_tensor_variable(floatX(mu))
        self.s = tt.as_tensor_variable(floatX(s))

        self.mean = self.mode = mu
        self.variance = s**2 * np.pi**2 / 3.

    def logp(self, value):
        """
        Calculate log-probability of Logistic distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        mu = self.mu
        s = self.s

        return bound(
            -(value - mu) / s - tt.log(s) - 2 * tt.log1p(tt.exp(-(value - mu) / s)), s > 0)

    def random(self, point=None, size=None):
        """
        Draw random values from Logistic distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, s = draw_values([self.mu, self.s], point=point, size=size)

        return generate_samples(
            stats.logistic.rvs,
            loc=mu, scale=s,
            dist_shape=self.shape,
            size=size)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        mu = dist.mu
        s = dist.s
        name = r'\text{%s}' % name
        return r'${} \sim \text{{Logistic}}(\mathit{{mu}}={},~\mathit{{s}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(s))

    def logcdf(self, value):
        """
        Compute the log CDF for the Logistic distribution

        References
        ----------
        .. [Machler2012] Martin Mächler (2012).
            "Accurately computing log(1-exp(-|a|)) Assessed by the Rmpfr
            package"
        """
        mu = self.mu
        s = self.s
        a = -(value - mu)/s
        return - tt.switch(
            tt.le(a, -37),
            tt.exp(a),
            tt.switch(
                tt.le(a, 18),
                tt.log1p(tt.exp(a)),
                tt.switch(
                    tt.le(a, 33.3),
                    tt.exp(-a) + a,
                    a)))


class LogitNormal(UnitContinuous):
    R"""
    Logit-Normal log-likelihood.

    The pdf of this distribution is

    .. math::
       f(x \mid \mu, \tau) =
           \frac{1}{x(1-x)} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (logit(x)-\mu)^2 \right\}


    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy.special import logit
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(0.0001, 0.9999, 500)
        mus = [0., 0., 0., 1.]
        sigmas = [0.3, 1., 2., 1.]
        for mu, sigma in  zip(mus, sigmas):
            pdf = st.norm.pdf(logit(x), loc=mu, scale=sigma) * 1/(x * (1-x))
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
            plt.legend(loc=1)
        plt.show()

    ========  ==========================================
    Support   :math:`x \in (0, 1)`
    Mean      no analytical solution
    Variance  no analytical solution
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    tau : float
        Scale parameter (tau > 0).
    """

    def __init__(self, mu=0, sigma=None, tau=None, sd=None, **kwargs):
        if sd is not None:
            sigma = sd
        self.mu = mu = tt.as_tensor_variable(floatX(mu))
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        self.sigma = self.sd = tt.as_tensor_variable(sigma)
        self.tau = tau = tt.as_tensor_variable(tau)

        self.median = invlogit(mu)
        assert_negative_support(sigma, 'sigma', 'LogitNormal')
        assert_negative_support(tau, 'tau', 'LogitNormal')

        super().__init__(**kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from LogitNormal distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        mu, _, sigma = draw_values(
            [self.mu, self.tau, self.sigma], point=point, size=size)
        return expit(generate_samples(stats.norm.rvs, loc=mu, scale=sigma, dist_shape=self.shape,
                                      size=size))

    def logp(self, value):
        """
        Calculate log-probability of LogitNormal distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        sigma = self.sigma
        mu = self.mu
        tau = self.tau
        return bound(-0.5 * tau * (logit(value) - mu) ** 2
                     + 0.5 * tt.log(tau / (2. * np.pi))
                     - tt.log(value * (1 - value)), value > 0, value < 1, tau > 0)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        sigma = dist.sigma
        mu = dist.mu
        name = r'\text{%s}' % name
        return r'${} \sim \text{{LogitNormal}}(\mathit{{mu}}={},~\mathit{{sigma}}={})$'.format(name,
                                                                get_variable_name(mu),
                                                                get_variable_name(sigma))


class Interpolated(BoundedContinuous):
    R"""
    Univariate probability distribution defined as a linear interpolation
    of probability density function evaluated on some lattice of points.

    The lattice can be uneven, so the steps between different points can have
    different size and it is possible to vary the precision between regions
    of the support.

    The probability density function values don not have to be normalized, as the
    interpolated density is any way normalized to make the total probability
    equal to $1$.

    Both parameters `x_points` and values `pdf_points` are not variables, but
    plain array-like objects, so they are constant and cannot be sampled.

    ========  ===========================================
    Support   :math:`x \in [x\_points[0], x\_points[-1]]`
    ========  ===========================================

    Parameters
    ----------
    x_points : array-like
        A monotonically growing list of values
    pdf_points : array-like
        Probability density function evaluated on lattice `x_points`
    """

    def __init__(self, x_points, pdf_points, *args, **kwargs):
        self.lower = lower = tt.as_tensor_variable(x_points[0])
        self.upper = upper = tt.as_tensor_variable(x_points[-1])

        super().__init__(lower=lower, upper=upper, *args, **kwargs)

        interp = InterpolatedUnivariateSpline(
            x_points, pdf_points, k=1, ext='zeros')
        Z = interp.integral(x_points[0], x_points[-1])

        self.Z = tt.as_tensor_variable(Z)
        self.interp_op = SplineWrapper(interp)
        self.x_points = x_points
        self.pdf_points = pdf_points / Z
        self.cdf_points = interp.antiderivative()(x_points) / Z

        self.median = self._argcdf(0.5)

    def _argcdf(self, p):
        pdf = self.pdf_points
        cdf = self.cdf_points
        x = self.x_points

        index = np.searchsorted(cdf, p) - 1
        slope = (pdf[index + 1] - pdf[index]) / (x[index + 1] - x[index])

        return x[index] + np.where(
            np.abs(slope) <= 1e-8,
            np.where(
                np.abs(pdf[index]) <= 1e-8,
                np.zeros(index.shape),
                (p - cdf[index]) / pdf[index]
            ),
            (-pdf[index] + np.sqrt(pdf[index] ** 2 + 2 * slope * (p - cdf[index]))) / slope
        )

    def _random(self, size=None):
        return self._argcdf(np.random.uniform(size=size))

    def random(self, size=None):
        """
        Draw random values from Interpolated distribution.

        Parameters
        ----------
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        return generate_samples(self._random,
                                dist_shape=self.shape,
                                size=size)

    def logp(self, value):
        """
        Calculate log-probability of Interpolated distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value(s) for which log-probability is calculated. If the log probabilities for multiple
            values are desired the values must be provided in a numpy array or theano tensor

        Returns
        -------
        TensorVariable
        """
        return tt.log(self.interp_op(value) / self.Z)
