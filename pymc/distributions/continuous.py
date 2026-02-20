#   Copyright 2024 - present The PyMC Developers
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

# Contains code from AePPL, Copyright (c) 2021-2022, Aesara Developers.

"""A collection of common probability distributions for stochastic nodes in PyMC."""

import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.raise_op import Assert
from pytensor.tensor import gamma as gammafn
from pytensor.tensor import gammaln, get_underlying_scalar_constant_value
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.extra_ops import broadcast_shape
from pytensor.tensor.math import betaincinv, gammaincinv, tanh
from pytensor.tensor.random.basic import (
    BetaRV,
    _gamma,
    cauchy,
    exponential,
    gumbel,
    halfnormal,
    invgamma,
    laplace,
    logistic,
    lognormal,
    normal,
    pareto,
    t,
    triangular,
    uniform,
    vonmises,
)
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.utils import normalize_size_param
from pytensor.tensor.variable import TensorConstant, TensorVariable

from pymc.distributions.custom import CustomDist
from pymc.logprob.abstract import _logprob_helper
from pymc.logprob.basic import TensorLike, icdf
from pymc.pytensorf import normalize_rng_param

try:
    from polyagamma import polyagamma_cdf, polyagamma_pdf, random_polyagamma
except ImportError:  # pragma: no cover

    def random_polyagamma(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")

    def polyagamma_pdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")

    def polyagamma_cdf(*args, **kwargs):
        raise RuntimeError("polyagamma package is not installed!")


from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline

from pymc.distributions import transforms
from pymc.distributions.dist_math import (
    SplineWrapper,
    check_icdf_parameters,
    check_icdf_value,
    check_parameters,
    clipped_beta_rvs,
    i0e,
    log_diff_normal_cdf,
    log_normal,
    logpow,
    normal_lccdf,
    normal_lcdf,
    zvalue,
)
from pymc.distributions.distribution import DIST_PARAMETER_TYPES, Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import implicit_size_from_params, rv_size_is_none
from pymc.distributions.transforms import _default_transform
from pymc.math import invlogit, logdiffexp

__all__ = [
    "AsymmetricLaplace",
    "Beta",
    "Cauchy",
    "ChiSquared",
    "ExGaussian",
    "Exponential",
    "Flat",
    "Gamma",
    "Gumbel",
    "HalfCauchy",
    "HalfFlat",
    "HalfNormal",
    "HalfStudentT",
    "Interpolated",
    "InverseGamma",
    "Kumaraswamy",
    "Laplace",
    "LogNormal",
    "Logistic",
    "LogitNormal",
    "Moyal",
    "Normal",
    "Pareto",
    "PolyaGamma",
    "Rice",
    "SkewNormal",
    "SkewStudentT",
    "StudentT",
    "Triangular",
    "TruncatedNormal",
    "Uniform",
    "VonMises",
    "Wald",
    "Weibull",
]


class PositiveContinuous(Continuous):
    """Base class for positive continuous distributions."""


class UnitContinuous(Continuous):
    """Base class for continuous distributions on [0,1]."""


class CircularContinuous(Continuous):
    """Base class for circular continuous distributions."""


class BoundedContinuous(Continuous):
    """Base class for bounded continuous distributions."""

    # Indices of the arguments that define the lower and upper bounds of the distribution
    bound_args_indices: tuple[int | None, int | None] | None = None


@_default_transform.register(PositiveContinuous)
def pos_cont_transform(op, rv):
    return transforms.log


@_default_transform.register(UnitContinuous)
def unit_cont_transform(op, rv):
    return transforms.logodds


@_default_transform.register(CircularContinuous)
def circ_cont_transform(op, rv):
    return transforms.circular


@_default_transform.register(BoundedContinuous)
def bounded_cont_transform(op, rv, bound_args_indices=None):
    if bound_args_indices is None:
        raise ValueError(f"Must specify bound_args_indices for {op} bounded distribution")

    def transform_params(*args):
        lower, upper = None, None
        if bound_args_indices[0] is not None:
            lower = args[bound_args_indices[0]]
        if bound_args_indices[1] is not None:
            upper = args[bound_args_indices[1]]

        if lower is not None:
            lower = pt.as_tensor_variable(lower)
            try:
                if get_underlying_scalar_constant_value(lower) == -np.inf:
                    lower = None
            except NotScalarConstantError:
                pass

        if upper is not None:
            upper = pt.as_tensor_variable(upper)
            try:
                if get_underlying_scalar_constant_value(upper) == np.inf:
                    upper = None
            except NotScalarConstantError:
                pass

        return lower, upper

    return transforms.Interval(bounds_fn=transform_params)


def get_tau_sigma(
    tau: TensorLike | None = None, sigma: TensorLike | None = None
) -> tuple[TensorVariable, TensorVariable]:
    r"""
    Find precision and standard deviation.

    The link between the two parameterizations is given by the inverse
    relationship:

    .. math::
        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    tau: array-like, optional
    sigma: array-like, optional

    Results
    -------
    Returns tuple (tau, sigma)

    Notes
    -----
    If neither tau nor sigma is provided, returns (1., 1.)
    """
    if tau is not None and sigma is not None:
        raise ValueError("Can't pass both tau and sigma")
    if tau is None and sigma is None:
        sigma = pt.as_tensor_variable(1.0)
        tau = pt.as_tensor_variable(1.0)
    elif tau is None:
        assert sigma is not None  # Just for type checker
        sigma = pt.as_tensor_variable(sigma)
        # Keep tau negative, if sigma was negative, so that it will
        # fail when used
        tau = (sigma**-2.0) * pt.sign(sigma)
    else:
        tau = pt.as_tensor_variable(tau)
        # Keep sigma negative, if tau was negative, so that it will
        # fail when used
        sigma = pt.abs(tau) ** -0.5 * pt.sign(tau)

    return tau, sigma


class Uniform(BoundedContinuous):
    r"""
    Continuous uniform distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid lower, upper) = \frac{1}{upper-lower}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    lower : tensor_like of float, default 0
        Lower limit.
    upper : tensor_like of float, default 1
        Upper limit.
    """

    rv_op = uniform
    bound_args_indices = (2, 3)  # Lower, Upper

    @classmethod
    def dist(cls, lower=0, upper=1, **kwargs):
        lower = pt.as_tensor_variable(lower)
        upper = pt.as_tensor_variable(upper)
        return super().dist([lower, upper], **kwargs)

    def support_point(rv, size, lower, upper):
        lower, upper = pt.broadcast_arrays(lower, upper)
        support_point = (lower + upper) / 2
        if not rv_size_is_none(size):
            support_point = pt.full(size, support_point)
        return support_point

    def logp(value, lower, upper):
        res = pt.switch(
            pt.bitwise_and(pt.ge(value, lower), pt.le(value, upper)),
            pt.fill(value, -pt.log(upper - lower)),
            -np.inf,
        )

        return check_parameters(
            res,
            lower <= upper,
            msg="lower <= upper",
        )

    def logcdf(value, lower, upper):
        res = pt.switch(
            pt.lt(value, lower),
            -np.inf,
            pt.switch(
                pt.lt(value, upper),
                pt.log(value - lower) - pt.log(upper - lower),
                0,
            ),
        )

        return check_parameters(
            res,
            lower <= upper,
            msg="lower <= upper",
        )

    def icdf(value, lower, upper):
        res = lower + (upper - lower) * value
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, lower < upper)


@_default_transform.register(Uniform)
def uniform_default_transform(op, rv):
    return bounded_cont_transform(op, rv, Uniform.bound_args_indices)


class FlatRV(RandomVariable):
    name = "flat"
    signature = "->()"
    dtype = "floatX"
    _print_name = ("Flat", "\\operatorname{Flat}")

    @classmethod
    def rng_fn(cls, rng, size):
        raise NotImplementedError("Cannot sample from flat variable")


flat = FlatRV()


class Flat(Continuous):
    """Uninformative distribution that returns 0 regardless of the passed value."""

    rv_op = flat

    @classmethod
    def dist(cls, **kwargs):
        res = super().dist([], **kwargs)
        return res

    def support_point(rv, size):
        return pt.zeros(() if rv_size_is_none(size) else size)

    def logp(value):
        return pt.zeros_like(value)

    def logcdf(value):
        return pt.switch(
            pt.eq(value, -np.inf), -np.inf, pt.switch(pt.eq(value, np.inf), 0, pt.log(0.5))
        )


class HalfFlatRV(RandomVariable):
    name = "half_flat"
    signature = "->()"
    dtype = "floatX"
    _print_name = ("HalfFlat", "\\operatorname{HalfFlat}")

    @classmethod
    def rng_fn(cls, rng, size):
        raise NotImplementedError("Cannot sample from half_flat variable")


halfflat = HalfFlatRV()


class HalfFlat(PositiveContinuous):
    """Improper flat prior over the positive reals."""

    rv_op = halfflat

    @classmethod
    def dist(cls, **kwargs):
        res = super().dist([], **kwargs)
        return res

    def support_point(rv, size):
        return pt.ones(() if rv_size_is_none(size) else size)

    def logp(value):
        return pt.switch(pt.lt(value, 0), -np.inf, pt.zeros_like(value))

    def logcdf(value):
        return pt.switch(pt.lt(value, np.inf), -np.inf, pt.switch(pt.eq(value, np.inf), 0, -np.inf))


class Normal(Continuous):
    r"""
    Univariate normal distribution.

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
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float, default 0
        Mean.
    sigma : tensor_like of float, optional
        Standard deviation (sigma > 0) (only required if tau is not specified).
        Defaults to 1 if neither sigma nor tau is specified.
    tau : tensor_like of float, optional
        Precision (tau > 0) (only required if sigma is not specified).

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.Normal("x", mu=0, sigma=10)

        with pm.Model():
            x = pm.Normal("x", mu=0, tau=1 / 23)
    """

    rv_op = normal

    @classmethod
    def dist(cls, mu=0, sigma=None, tau=None, **kwargs):
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        sigma = pt.as_tensor_variable(sigma)
        mu = pt.as_tensor_variable(mu)
        return super().dist([mu, sigma], **kwargs)

    def support_point(rv, size, mu, sigma):
        mu, _ = pt.broadcast_arrays(mu, sigma)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    def logp(value, mu, sigma):
        res = -0.5 * pt.pow((value - mu) / sigma, 2) - pt.log(pt.sqrt(2.0 * np.pi)) - pt.log(sigma)
        return check_parameters(
            res,
            sigma > 0,
            msg="sigma > 0",
        )

    def logcdf(value, mu, sigma):
        return check_parameters(
            normal_lcdf(mu, sigma, value),
            sigma > 0,
            msg="sigma > 0",
        )

    def logccdf(value, mu, sigma):
        return check_parameters(
            normal_lccdf(mu, sigma, value),
            sigma > 0,
            msg="sigma > 0",
        )

    def icdf(value, mu, sigma):
        res = mu + sigma * -np.sqrt(2.0) * pt.erfcinv(2 * value)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            sigma > 0,
            msg="sigma > 0",
        )


class TruncatedNormalRV(RandomVariable):
    name = "truncated_normal"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("TruncatedNormal", "\\operatorname{TruncatedNormal}")

    @classmethod
    def rng_fn(
        cls,
        rng: np.random.RandomState,
        mu: np.ndarray | float,
        sigma: np.ndarray | float,
        lower: np.ndarray | float,
        upper: np.ndarray | float,
        size: list[int] | int | None,
    ) -> np.ndarray:
        # Upcast to float64. (Caller will downcast to desired dtype if needed)
        #   (Work-around for https://github.com/scipy/scipy/issues/15928)
        return stats.truncnorm.rvs(
            a=((lower - mu) / sigma).astype("float64"),
            b=((upper - mu) / sigma).astype("float64"),
            loc=(mu).astype("float64"),
            scale=(sigma).astype("float64"),
            size=size,
            random_state=rng,
        )


truncated_normal = TruncatedNormalRV()


class TruncatedNormal(BoundedContinuous):
    r"""
    Univariate truncated normal distribution.

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
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 10, 1000)
        mus = [0.,  0., 0.]
        sigmas = [3.,5.,7.]
        a1 = [-3, -5, -5]
        b1 = [7, 5, 4]
        for mu, sigma, a, b in zip(mus, sigmas,a1,b1):
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
    Variance  :math:`\sigma ^{2}\left[1+{\frac {\alpha \phi (\alpha )-\beta \phi (\beta )}{Z}}-\left({\frac {\phi (\alpha )-\phi (\beta )}{Z}}\right)^{2}\right]`
    ========  ==========================================

    Parameters
    ----------
    mu : tensor_like of float, default 0
        Mean.
    sigma : tensor_like of float, optional
        Standard deviation (sigma > 0) (only required if tau is not specified).
        Defaults to 1 if neither sigma nor tau is specified.
    tau : tensor_like of float, optional
        Precision (tau > 0) (only required if sigma is not specified).
    lower : tensor_like of float, default - numpy.inf
        Left bound.
    upper : tensor_like of float, default numpy.inf
        Right bound.

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.TruncatedNormal("x", mu=0, sigma=10, lower=0)

        with pm.Model():
            x = pm.TruncatedNormal("x", mu=0, sigma=10, upper=1)

        with pm.Model():
            x = pm.TruncatedNormal("x", mu=0, sigma=10, lower=0, upper=1)

    """

    rv_op = truncated_normal
    bound_args_indices = (4, 5)  # indexes for lower and upper args

    @classmethod
    def dist(
        cls,
        mu: DIST_PARAMETER_TYPES | None = 0,
        sigma: DIST_PARAMETER_TYPES | None = None,
        *,
        tau: DIST_PARAMETER_TYPES | None = None,
        lower: DIST_PARAMETER_TYPES | None = None,
        upper: DIST_PARAMETER_TYPES | None = None,
        **kwargs,
    ) -> RandomVariable:
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        sigma = pt.as_tensor_variable(sigma)
        mu = pt.as_tensor_variable(mu)

        lower = pt.as_tensor_variable(lower) if lower is not None else pt.constant(-np.inf)
        upper = pt.as_tensor_variable(upper) if upper is not None else pt.constant(np.inf)
        return super().dist([mu, sigma, lower, upper], **kwargs)

    def support_point(rv, size, mu, sigma, lower, upper):
        mu, _, lower, upper = pt.broadcast_arrays(mu, sigma, lower, upper)
        support_point = pt.switch(
            pt.eq(lower, -np.inf),
            pt.switch(
                pt.eq(upper, np.inf),
                # lower = -inf, upper = inf
                mu,
                # lower = -inf, upper = x
                upper - 1,
            ),
            pt.switch(
                pt.eq(upper, np.inf),
                # lower = x, upper = inf
                lower + 1,
                # lower = x, upper = x
                (lower + upper) / 2,
            ),
        )

        if not rv_size_is_none(size):
            support_point = pt.full(size, support_point)

        return support_point

    def logp(value, mu, sigma, lower, upper):
        is_lower_bounded = not (
            isinstance(lower, TensorConstant) and np.all(np.isneginf(lower.value))
        )
        is_upper_bounded = not (isinstance(upper, TensorConstant) and np.all(np.isinf(upper.value)))

        if is_lower_bounded and is_upper_bounded:
            norm = log_diff_normal_cdf(mu, sigma, upper, lower)
        elif is_lower_bounded:
            norm = normal_lccdf(mu, sigma, lower)
        elif is_upper_bounded:
            norm = normal_lcdf(mu, sigma, upper)
        else:
            norm = 0.0

        logp = _logprob_helper(Normal.dist(mu, sigma), value) - norm

        if is_lower_bounded:
            logp = pt.switch(value < lower, -np.inf, logp)

        if is_upper_bounded:
            logp = pt.switch(value > upper, -np.inf, logp)

        if is_lower_bounded and is_upper_bounded:
            logp = check_parameters(
                logp,
                pt.le(lower, upper),
                msg="lower_bound <= upper_bound",
            )

        return logp

    def logcdf(value, mu, sigma, lower, upper):
        logcdf = log_diff_normal_cdf(mu, sigma, value, lower) - log_diff_normal_cdf(
            mu, sigma, upper, lower
        )

        is_lower_bounded = not (
            isinstance(lower, TensorConstant) and np.all(np.isneginf(lower.value))
        )
        is_upper_bounded = not (isinstance(upper, TensorConstant) and np.all(np.isinf(upper.value)))

        if is_lower_bounded:
            logcdf = pt.switch(value < lower, -np.inf, logcdf)

        if is_upper_bounded:
            logcdf = pt.switch(value <= upper, logcdf, 0.0)

        if is_lower_bounded and is_upper_bounded:
            logcdf = check_parameters(
                logcdf,
                pt.le(lower, upper),
                msg="lower_bound <= upper_bound",
            )

        return logcdf


@_default_transform.register(TruncatedNormal)
def truncated_normal_default_transform(op, rv):
    return bounded_cont_transform(op, rv, TruncatedNormal.bound_args_indices)


class HalfNormal(PositiveContinuous):
    r"""
    Half-normal distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \tau) =
           \sqrt{\frac{2\tau}{\pi}}
           \exp\left(\frac{-x^2 \tau}{2}\right)

       f(x \mid \sigma) =
           \sqrt{\frac{2}{\pi\sigma^2}}
           \exp\left(\frac{-x^2}{2\sigma^2}\right)

    .. note::

       The parameters ``sigma``/``tau`` (:math:`\sigma`/:math:`\tau`) refer to
       the standard deviation/precision of the unfolded normal distribution, for
       the standard deviation of the half-normal distribution, see below. For
       the half-normal, they are just two parameterisation :math:`\sigma^2
       \equiv \frac{1}{\tau}` of a scale parameter.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    sigma : tensor_like of float, optional
        Scale parameter :math:`\sigma` (``sigma`` > 0) (only required if ``tau`` is not specified).
        Defaults to 1.
    tau : tensor_like of float, optional
        Precision :math:`\tau` (tau > 0) (only required if sigma is not specified).
        Defaults to 1.

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.HalfNormal("x", sigma=10)

        with pm.Model():
            x = pm.HalfNormal("x", tau=1 / 15)
    """

    rv_op = halfnormal

    @classmethod
    def dist(
        cls,
        sigma: DIST_PARAMETER_TYPES | None = None,
        tau: DIST_PARAMETER_TYPES | None = None,
        *args,
        **kwargs,
    ):
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        return super().dist([0.0, sigma], **kwargs)

    def support_point(rv, size, loc, sigma):
        support_point = loc + sigma
        if not rv_size_is_none(size):
            support_point = pt.full(size, support_point)
        return support_point

    def logp(value, loc, sigma):
        res = -0.5 * pt.pow((value - loc) / sigma, 2) + pt.log(pt.sqrt(2.0 / np.pi)) - pt.log(sigma)
        res = pt.switch(pt.ge(value, loc), res, -np.inf)
        return check_parameters(
            res,
            sigma > 0,
            msg="sigma > 0",
        )

    def logcdf(value, loc, sigma):
        z = zvalue(value, mu=loc, sigma=sigma)
        logcdf = pt.switch(
            pt.lt(value, loc),
            -np.inf,
            pt.log1p(-pt.erfc(z / pt.sqrt(2.0))),
        )

        return check_parameters(
            logcdf,
            sigma > 0,
            msg="sigma > 0",
        )

    def icdf(value, loc, sigma):
        res = icdf(Normal.dist(loc, sigma), (value + 1.0) / 2.0)
        res = check_icdf_value(res, value)
        return res


class WaldRV(RandomVariable):
    name = "wald"
    signature = "(),(),()->()"
    dtype = "floatX"
    _print_name = ("Wald", "\\operatorname{Wald}")

    @classmethod
    def rng_fn(cls, rng, mu, lam, alpha, size) -> np.ndarray:
        return np.asarray(rng.wald(mu, lam, size=size) + alpha)


wald = WaldRV()


class Wald(PositiveContinuous):
    r"""
    Wald distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \lambda) =
           \left(\frac{\lambda}{2\pi}\right)^{1/2} x^{-3/2}
           \exp\left\{
               -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2
           \right\}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float, optional
        Mean of the distribution (mu > 0).
    lam : tensor_like of float, optional
        Relative precision (lam > 0).
    phi : tensor_like of float, optional
        Alternative shape parameter (phi > 0).
    alpha : tensor_like of float, default 0
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

    rv_op = wald

    @classmethod
    def dist(
        cls,
        mu: DIST_PARAMETER_TYPES | None = None,
        lam: DIST_PARAMETER_TYPES | None = None,
        phi: DIST_PARAMETER_TYPES | None = None,
        alpha: DIST_PARAMETER_TYPES | None = 0.0,
        **kwargs,
    ):
        mu, lam, phi = cls.get_mu_lam_phi(mu, lam, phi)
        alpha = pt.as_tensor_variable(alpha)
        mu = pt.as_tensor_variable(mu)
        lam = pt.as_tensor_variable(lam)
        return super().dist([mu, lam, alpha], **kwargs)

    def support_point(rv, size, mu, lam, alpha):
        mu, _, _ = pt.broadcast_arrays(mu, lam, alpha)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    @staticmethod
    def get_mu_lam_phi(mu, lam, phi):
        if mu is None:
            if lam is not None and phi is not None:
                return lam / phi, lam, phi
        else:
            if lam is None:
                if phi is None:
                    return mu, 1.0, 1.0 / mu
                else:
                    return mu, mu * phi, phi
            else:
                if phi is None:
                    return mu, lam, lam / mu

        raise ValueError(
            "Wald distribution must specify either mu only, mu and lam, mu and phi, or lam and phi."
        )

    def logp(value, mu, lam, alpha):
        centered_value = value - alpha
        logp = pt.switch(
            pt.le(centered_value, 0),
            -np.inf,
            (
                logpow(lam / (2.0 * np.pi), 0.5)
                - logpow(centered_value, 1.5)
                - (0.5 * lam / centered_value * ((centered_value - mu) / mu) ** 2)
            ),
        )

        return check_parameters(
            logp,
            mu > 0,
            lam > 0,
            alpha >= 0,
            msg="mu > 0, lam > 0, alpha >= 0",
        )

    def logcdf(value, mu, lam, alpha):
        value -= alpha
        q = value / mu
        ell = lam * mu
        r = pt.sqrt(value * lam)

        a = normal_lcdf(0, 1, (q - 1.0) / r)
        b = 2.0 / ell + normal_lcdf(0, 1, -(q + 1.0) / r)

        logcdf = pt.switch(
            pt.le(value, 0),
            -np.inf,
            pt.switch(
                pt.lt(value, np.inf),
                a + pt.log1pexp(b - a),
                0,
            ),
        )

        return check_parameters(
            logcdf,
            mu > 0,
            lam > 0,
            alpha >= 0,
            msg="mu > 0, lam > 0, alpha >= 0",
        )


class BetaClippedRV(BetaRV):
    @classmethod
    def rng_fn(cls, rng, alpha, beta, size) -> np.ndarray:
        return np.asarray(clipped_beta_rvs(alpha, beta, size=size, random_state=rng))


beta = BetaClippedRV()


class Beta(UnitContinuous):
    r"""
    Beta distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}

    where :math:`B` is the Beta function.

    For more information, see https://en.wikipedia.org/wiki/Beta_distribution.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    beta, mean and standard deviation or mean and sample size. The link between the three
    parametrizations is given by

    .. math::

       \alpha &= \mu \kappa \\
       \beta  &= (1 - \mu) \kappa

       \text{where } \kappa = \frac{\mu(1-\mu)}{\sigma^2} - 1

       \alpha &= \mu * \nu \\
       \beta &= (1 - \mu) * \nu

    Parameters
    ----------
    alpha : tensor_like of float, optional
        ``alpha`` > 0. If not specified, then calculated using ``mu`` and ``sigma``.
    beta : tensor_like of float, optional
        ``beta`` > 0. If not specified, then calculated using ``mu`` and ``sigma``.
    mu : tensor_like of float, optional
        Alternative mean (0 < ``mu`` < 1).
    sigma : tensor_like of float, optional
        Alternative standard deviation (0 < ``sigma`` < sqrt(``mu`` * (1 - ``mu``))).
    nu : tensor_like of float, optional
        Alternative "sample size" of a Beta distribution (``nu`` > 0).

    Notes
    -----
    Beta distribution is a conjugate prior for the parameter :math:`p` of
    the binomial distribution.
    """

    rv_op = pytensor.tensor.random.beta

    @classmethod
    def dist(
        cls,
        alpha: DIST_PARAMETER_TYPES | None = None,
        beta: DIST_PARAMETER_TYPES | None = None,
        mu: DIST_PARAMETER_TYPES | None = None,
        sigma: DIST_PARAMETER_TYPES | None = None,
        nu: DIST_PARAMETER_TYPES | None = None,
        *args,
        **kwargs,
    ):
        alpha, beta = cls.get_alpha_beta(alpha, beta, mu, sigma, nu)
        alpha = pt.as_tensor_variable(alpha)
        beta = pt.as_tensor_variable(beta)

        return super().dist([alpha, beta], **kwargs)

    def support_point(rv, size, alpha, beta):
        mean = alpha / (alpha + beta)
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    @classmethod
    def get_alpha_beta(self, alpha=None, beta=None, mu=None, sigma=None, nu=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            kappa = mu * (1 - mu) / sigma**2 - 1
            alpha = mu * kappa
            beta = (1 - mu) * kappa
        elif (mu is not None) and (nu is not None):
            alpha = mu * nu
            beta = (1 - mu) * nu
        else:
            raise ValueError(
                "Incompatible parameterization. Either use alpha "
                "and beta, mu and sigma or mu and nu to specify "
                "distribution."
            )

        return alpha, beta

    def logp(value, alpha, beta):
        res = (
            pt.switch(pt.eq(alpha, 1.0), 0.0, (alpha - 1.0) * pt.log(value))
            + pt.switch(pt.eq(beta, 1.0), 0.0, (beta - 1.0) * pt.log1p(-value))
            - (pt.gammaln(alpha) + pt.gammaln(beta) - pt.gammaln(alpha + beta))
        )
        res = pt.switch(pt.bitwise_and(pt.ge(value, 0.0), pt.le(value, 1.0)), res, -np.inf)
        return check_parameters(
            res,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )

    def logcdf(value, alpha, beta):
        logcdf = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.switch(
                pt.lt(value, 1),
                pt.log(pt.betainc(alpha, beta, value)),
                0,
            ),
        )

        return check_parameters(
            logcdf,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )

    def icdf(value, alpha, beta):
        res = betaincinv(alpha, beta, value)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )


class KumaraswamyRV(SymbolicRandomVariable):
    name = "kumaraswamy"
    extended_signature = "[rng],[size],(),()->[rng],()"
    _print_name = ("Kumaraswamy", "\\operatorname{Kumaraswamy}")

    @classmethod
    def rv_op(cls, a, b, *, size=None, rng=None):
        a = pt.as_tensor(a)
        b = pt.as_tensor(b)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        if rv_size_is_none(size):
            size = implicit_size_from_params(a, b, ndims_params=cls.ndims_params)

        next_rng, u = uniform(size=size, rng=rng).owner.outputs
        draws = (1 - (1 - u) ** (1 / b)) ** (1 / a)

        return cls(
            inputs=[rng, size, a, b],
            outputs=[next_rng, draws],
        )(rng, size, a, b)


class Kumaraswamy(UnitContinuous):
    r"""
    Kumaraswamy distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid a, b) =
           abx^{a-1}(1-x^a)^{b-1}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    a : tensor_like of float
        a > 0.
    b : tensor_like of float
        b > 0.
    """

    rv_type = KumaraswamyRV
    rv_op = KumaraswamyRV.rv_op

    @classmethod
    def dist(cls, a: DIST_PARAMETER_TYPES, b: DIST_PARAMETER_TYPES, *args, **kwargs):
        return super().dist([a, b], *args, **kwargs)

    def support_point(rv, size, a, b):
        mean = pt.exp(pt.log(b) + pt.gammaln(1 + 1 / a) + pt.gammaln(b) - pt.gammaln(1 + 1 / a + b))
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, a, b):
        res = pt.log(a) + pt.log(b) + (a - 1) * pt.log(value) + (b - 1) * pt.log(1 - value**a)
        res = pt.switch(
            pt.or_(pt.lt(value, 0), pt.gt(value, 1)),
            -np.inf,
            res,
        )
        return check_parameters(
            res,
            a > 0,
            b > 0,
            msg="a > 0, b > 0",
        )

    def logcdf(value, a, b):
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.switch(
                pt.lt(value, 1),
                pt.log1mexp(b * pt.log1p(-(value**a))),
                0,
            ),
        )

        return check_parameters(
            res,
            a > 0,
            b > 0,
            msg="a > 0, b > 0",
        )

    def icdf(value, a, b):
        res = pt.exp(pt.reciprocal(a) * pt.log1mexp(pt.reciprocal(b) * pt.log1p(-value)))
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            a > 0,
            b > 0,
            msg="a > 0, b > 0",
        )


class Exponential(PositiveContinuous):
    r"""
    Exponential distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \lambda) = \lambda \exp\left\{ -\lambda x \right\}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    lam : tensor_like of float
        Rate or inverse scale (``lam`` > 0).
    scale: tensor_like of float
        Alternative parameter (scale = 1/lam).
    """

    rv_op = exponential

    @classmethod
    def dist(cls, lam=None, *, scale=None, **kwargs):
        if lam is None and scale is None:
            scale = 1.0
        elif lam is not None and scale is not None:
            raise ValueError("Incompatible parametrization. Can't specify both lam and scale.")
        elif lam is not None:
            scale = pt.reciprocal(lam)

        scale = pt.as_tensor_variable(scale)
        # PyTensor exponential op is parametrized in terms of mu (1/lam)
        return super().dist([scale], **kwargs)

    def support_point(rv, size, mu):
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    def logp(value, mu):
        res = -pt.log(mu) - value / mu
        res = pt.switch(pt.ge(value, 0.0), res, -np.inf)
        return check_parameters(
            res,
            mu >= 0,
            msg="mu >= 0",
        )

    def logcdf(value, mu):
        lam = pt.reciprocal(mu)
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log1mexp(-lam * value),
        )

        return check_parameters(
            res,
            lam >= 0,
            msg="lam >= 0",
        )

    def icdf(value, mu):
        res = -mu * pt.log(1 - value)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            mu >= 0,
            msg="mu >= 0",
        )


class Laplace(Continuous):
    r"""
    Laplace distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, b) =
           \frac{1}{2b} \exp \left\{ - \frac{|x - \mu|}{b} \right\}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float
        Location parameter.
    b : tensor_like of float
        Scale parameter (b > 0).
    """

    rv_op = laplace

    @classmethod
    def dist(cls, mu, b, *args, **kwargs):
        b = pt.as_tensor_variable(b)
        mu = pt.as_tensor_variable(mu)

        return super().dist([mu, b], *args, **kwargs)

    def support_point(rv, size, mu, b):
        mu, _ = pt.broadcast_arrays(mu, b)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    def logp(value, mu, b):
        res = -pt.log(2 * b) - pt.abs(value - mu) / b
        return check_parameters(
            res,
            b > 0,
            msg="b > 0",
        )

    def logcdf(value, mu, b):
        y = (value - mu) / b

        res = pt.switch(
            pt.le(value, mu),
            pt.log(0.5) + y,
            pt.switch(
                pt.gt(y, 1),
                pt.log1p(-0.5 * pt.exp(-y)),
                pt.log(1 - 0.5 * pt.exp(-y)),
            ),
        )

        return check_parameters(
            res,
            b > 0,
            msg="b > 0",
        )

    def icdf(value, mu, b):
        res = pt.switch(
            pt.le(value, 0.5), mu + b * np.log(2 * value), mu - b * np.log(2 - 2 * value)
        )
        res = check_icdf_value(res, value)
        return check_icdf_parameters(res, b > 0, msg="b > 0")


class AsymmetricLaplaceRV(SymbolicRandomVariable):
    name = "asymmetriclaplace"
    extended_signature = "[rng],[size],(),(),()->[rng],()"
    _print_name = ("AsymmetricLaplace", "\\operatorname{AsymmetricLaplace}")

    @classmethod
    def rv_op(cls, b, kappa, mu, *, size=None, rng=None):
        b = pt.as_tensor(b)
        kappa = pt.as_tensor(kappa)
        mu = pt.as_tensor(mu)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        if rv_size_is_none(size):
            size = implicit_size_from_params(b, kappa, mu, ndims_params=cls.ndims_params)

        next_rng, u = uniform(size=size, rng=rng).owner.outputs
        switch = kappa**2 / (1 + kappa**2)
        non_positive_x = mu + kappa * pt.log(u * (1 / switch)) / b
        positive_x = mu - pt.log((1 - u) * (1 + kappa**2)) / (kappa * b)
        draws = non_positive_x * (u <= switch) + positive_x * (u > switch)

        return cls(
            inputs=[rng, size, b, kappa, mu],
            outputs=[next_rng, draws],
        )(rng, size, b, kappa, mu)


class AsymmetricLaplace(Continuous):
    r"""
    Asymmetric-Laplace distribution.

    The pdf of this distribution is

    .. math::
        {f(x|\\b,\kappa,\mu) =
            \left({\frac{\\b}{\kappa + 1/\kappa}}\right)\,e^{-(x-\mu)\\b\,s\kappa ^{s}}}

    where

    .. math::

        s = sgn(x-\mu)

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu-\frac{\\\kappa-1/\kappa}b`
    Variance  :math:`\frac{1+\kappa^{4}}{b^2\kappa^2 }`
    ========  ========================

    AsymmetricLaplace distribution can be parameterized either in terms of kappa
    or q. The link between the two parametrizations is given by

    .. math::

       \kappa = \sqrt(\frac{q}{1-q})

    Parameters
    ----------
    kappa : tensor_like of float
        Symmetry parameter (kappa > 0).
    mu : tensor_like of float
        Location parameter.
    b : tensor_like of float
        Scale parameter (b > 0).
    q : tensor_like of float
        Symmetry parameter (0 < q < 1).

    Notes
    -----
    The parametrization in terms of q is useful for quantile regression with q being the quantile
    of interest.
    """

    rv_type = AsymmetricLaplaceRV
    rv_op = AsymmetricLaplaceRV.rv_op

    @classmethod
    def dist(cls, kappa=None, mu=None, b=None, q=None, *args, **kwargs):
        kappa = cls.get_kappa(kappa, q)
        return super().dist([b, kappa, mu], *args, **kwargs)

    @classmethod
    def get_kappa(cls, kappa=None, q=None):
        if kappa is not None and q is not None:
            raise ValueError(
                "Incompatible parameterization. Either use kappa or q to specify the distribution."
            )
        elif q is not None:
            if isinstance(q, Variable):
                q = check_parameters(q, q > 0, q < 1, msg="0 < q < 1")
            else:
                assert np.all((np.asarray(q) > 0) | (np.asarray(q) < 1))
            kappa = (q / (1 - q)) ** 0.5

        return kappa

    def support_point(rv, size, b, kappa, mu):
        mean = mu - (kappa - 1 / kappa) / b

        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, b, kappa, mu):
        value = value - mu
        res = pt.log(b / (kappa + pt.reciprocal(kappa))) + (
            -value * b * pt.sign(value) * (kappa ** pt.sign(value))
        )

        return check_parameters(
            res,
            b > 0,
            kappa > 0,
            msg="b > 0, kappa > 0",
        )


class LogNormal(PositiveContinuous):
    r"""
    Log-normal distribution.

    Distribution of any random variable whose logarithm is normally
    distributed. A variable might be modeled as log-normal if it can
    be thought of as the multiplicative product of many small
    independent factors.

    Note: Class name Lognormal is deprecated, use LogNormal now!

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau) =
           \frac{1}{x} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (\ln(x)-\mu)^2 \right\}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float, default 0
        Location parameter.
    sigma : tensor_like of float, optional
        Standard deviation. (sigma > 0). (only required if tau is not specified).
        Defaults to 1.
    tau : tensor_like of float, optional
        Scale parameter (tau > 0). (only required if sigma is not specified).
        Defaults to 1.

    Examples
    --------
    .. code-block:: python

        # Example to show that we pass in only ``sigma`` or ``tau`` but not both.
        with pm.Model():
            x = pm.LogNormal("x", mu=2, sigma=30)

        with pm.Model():
            x = pm.LogNormal("x", mu=2, tau=1 / 100)
    """

    rv_op = lognormal

    @classmethod
    def dist(cls, mu=0, sigma=None, tau=None, *args, **kwargs):
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)

        mu = pt.as_tensor_variable(mu)
        sigma = pt.as_tensor_variable(sigma)

        return super().dist([mu, sigma], *args, **kwargs)

    def support_point(rv, size, mu, sigma):
        mean = pt.exp(mu + 0.5 * sigma**2)
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, mu, sigma):
        res = (
            -0.5 * pt.pow((pt.log(value) - mu) / sigma, 2)
            - 0.5 * pt.log(2.0 * np.pi)
            - pt.log(sigma)
            - pt.log(value)
        )
        res = pt.switch(pt.gt(value, 0.0), res, -np.inf)
        return check_parameters(
            res,
            sigma > 0,
            msg="sigma > 0",
        )

    def logcdf(value, mu, sigma):
        res = pt.switch(
            pt.le(value, 0),
            -np.inf,
            normal_lcdf(mu, sigma, pt.log(value)),
        )

        return check_parameters(
            res,
            sigma > 0,
            msg="sigma > 0",
        )

    def icdf(value, mu, sigma):
        res = pt.exp(icdf(Normal.dist(mu, sigma), value))
        return res


Lognormal = LogNormal


class StudentT(Continuous):
    r"""
    Student's T distribution.

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
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    nu : tensor_like of float
        Degrees of freedom, also known as normality parameter (nu > 0).
    mu : tensor_like of float, default 0
        Location parameter.
    sigma : tensor_like of float, optional
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases (only required if lam is not specified). Defaults to 1.
    lam : tensor_like of float, optional
        Scale parameter (lam > 0). Converges to the precision as nu
        increases (only required if sigma is not specified). Defaults to 1.

    Examples
    --------
    .. code-block:: python

        with pm.Model():
            x = pm.StudentT("x", nu=15, mu=0, sigma=10)

        with pm.Model():
            x = pm.StudentT("x", nu=15, mu=0, lam=1 / 23)
    """

    rv_op = t

    @classmethod
    def dist(cls, nu, mu=0, *, sigma=None, lam=None, **kwargs):
        nu = pt.as_tensor_variable(nu)
        lam, sigma = get_tau_sigma(tau=lam, sigma=sigma)
        sigma = pt.as_tensor_variable(sigma)

        return super().dist([nu, mu, sigma], **kwargs)

    def support_point(rv, size, nu, mu, sigma):
        mu, _, _ = pt.broadcast_arrays(mu, nu, sigma)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    def logp(value, nu, mu, sigma):
        lam, _ = get_tau_sigma(sigma=sigma)

        res = (
            gammaln((nu + 1.0) / 2.0)
            + 0.5 * pt.log(lam / (nu * np.pi))
            - gammaln(nu / 2.0)
            - (nu + 1.0) / 2.0 * pt.log1p(lam * (value - mu) ** 2 / nu)
        )

        return check_parameters(
            res,
            lam > 0,
            nu > 0,
            msg="lam > 0, nu > 0",
        )

    def logcdf(value, nu, mu, sigma):
        _, sigma = get_tau_sigma(sigma=sigma)

        t = (value - mu) / sigma
        sqrt_t2_nu = pt.sqrt(t**2 + nu)
        x = (t + sqrt_t2_nu) / (2.0 * sqrt_t2_nu)

        res = pt.log(pt.betainc(nu / 2.0, nu / 2.0, x))

        return check_parameters(
            res,
            nu > 0,
            sigma > 0,
            msg="nu > 0, sigma > 0",
        )

    def icdf(value, nu, mu, sigma):
        res = pt.switch(
            pt.lt(value, 0.5),
            -pt.sqrt(nu) * pt.sqrt((1.0 / betaincinv(nu * 0.5, 0.5, 2.0 * value)) - 1.0),
            pt.sqrt(nu) * pt.sqrt((1.0 / betaincinv(nu * 0.5, 0.5, 2.0 * (1 - value))) - 1.0),
        )
        res = mu + res * sigma
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            nu > 0,
            sigma > 0,
            msg="nu > 0, sigma > 0",
        )


class SkewStudentTRV(RandomVariable):
    name = "skewstudentt"
    signature = "(),(),(),()->()"
    dtype = "floatX"
    _print_name = ("SkewStudentT", "\\operatorname{SkewStudentT}")

    @classmethod
    def rng_fn(cls, rng, a, b, mu, sigma, size=None) -> np.ndarray:
        return np.asarray(
            stats.jf_skew_t.rvs(a=a, b=b, loc=mu, scale=sigma, size=size, random_state=rng)
        )


skewstudentt = SkewStudentTRV()


class SkewStudentT(Continuous):
    r"""
    Skewed Student's T distribution distribution.

    This follows Jones and Faddy (2003)

    The pdf of this distribution is

    .. math::

        f(t)=f(t ; a, b)=C_{a, b}^{-1}\left\{1+\frac{t}{\left(a+b+t^2\right)^{1 / 2}}\right\}^{a+1 / 2}\left\{1-\frac{t}{\left(a+b+t^2\right)^{1 / 2}}\right\}^{b+1 / 2}

    where

    .. math::

        C_{a, b}=2^{a+b-1} B(a, b)(a+b)^{1 / 2}


    ========  =============================================================
    Support   :math:`x \in [\infty, \infty)`
    Mean      :math:`E(T)=\frac{(a-b) \sqrt{(a+b)}}{2} \frac{\Gamma\left(a-\frac{1}{2}\right) \Gamma\left(b-\frac{1}{2}\right)}{\Gamma(a) \Gamma(b)}`
    ========  =============================================================

    Parameters
    ----------
    a : tensor_like of float
        First kurtosis parameter (a > 0).
    b : tensor_like of float
        Second kurtosis parameter (b > 0).
    mu : tensor_like of float
        Location parameter.
    sigma : tensor_like of float
        Scale parameter (sigma > 0). Converges to the standard deviation as a and b
        become close (only required if lam is not specified). Defaults to 1.
    lam : tensor_like of float, optional
        Scale parameter (lam > 0). Converges to the precision as a and b
        become close (only required if sigma is not specified). Defaults to 1.

    """

    rv_op = skewstudentt

    @classmethod
    def dist(cls, a, b, *, mu=0, sigma=None, lam=None, **kwargs):
        a = pt.as_tensor_variable(a)
        b = pt.as_tensor_variable(b)
        lam, sigma = get_tau_sigma(tau=lam, sigma=sigma)
        sigma = pt.as_tensor_variable(sigma)

        return super().dist([a, b, mu, sigma], **kwargs)

    def support_point(rv, size, a, b, mu, sigma):
        a, b, mu, _ = pt.broadcast_arrays(a, b, mu, sigma)
        Et = mu + (a - b) * pt.sqrt(a + b) * gammafn(a - 0.5) * gammafn(b - 0.5) / (
            2 * gammafn(a) * gammafn(b)
        )
        if not rv_size_is_none(size):
            Et = pt.full(size, Et)
        return Et

    def logp(value, a, b, mu, sigma):
        _, sigma = get_tau_sigma(sigma=sigma)

        x = (value - mu) / sigma

        a_ = (a + 0.5) * pt.log(1 + x / pt.sqrt(a + b + x**2))
        b_ = (b + 0.5) * pt.log(1 - x / pt.sqrt(a + b + x**2))
        c = (a + b - 1) * pt.log(2) + pt.special.betaln(a, b) + 0.5 * pt.log(a + b)

        res = a_ + b_ - c - pt.log(sigma)

        return check_parameters(
            res,
            a > 0,
            b > 0,
            sigma > 0,
            msg="a > 0, b > 0, sigma > 0",
        )

    def logcdf(value, a, b, mu, sigma):
        _, sigma = get_tau_sigma(sigma=sigma)

        x = (value - mu) / sigma

        y = (1 + x / pt.sqrt(a + b + x**2)) * 0.5
        res = pt.log(pt.betainc(a, b, y))

        return check_parameters(
            res,
            a > 0,
            b > 0,
            sigma > 0,
            msg="a > 0, b > 0, sigma > 0",
        )

    def icdf(value, a, b, mu, sigma):
        _, sigma = get_tau_sigma(sigma=sigma)

        bval = betaincinv(a, b, value)
        num = (2 * bval - 1) * pt.sqrt(a + b)
        denom = 2 * pt.sqrt(bval * (1 - bval))
        res = num / denom

        res = mu + res * sigma
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            a > 0,
            b > 0,
            sigma > 0,
            msg="a > 0, b > 0, sigma > 0",
        )


class Pareto(BoundedContinuous):
    r"""
    Pareto distribution.

    Often used to characterize wealth distribution, or other examples of the
    80/20 rule.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, m) = \frac{\alpha m^{\alpha}}{x^{\alpha+1}}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    alpha : tensor_like of float
        Shape parameter (alpha > 0).
    m : tensor_like of float
        Scale parameter (m > 0).
    """

    rv_op = pareto
    bound_args_indices = (3, None)  # lower-bounded by `m`

    @classmethod
    def dist(cls, alpha, m, **kwargs):
        alpha = pt.as_tensor_variable(alpha)
        m = pt.as_tensor_variable(m)

        return super().dist([alpha, m], **kwargs)

    def support_point(rv, size, alpha, m):
        median = m * 2 ** (1 / alpha)
        if not rv_size_is_none(size):
            median = pt.full(size, median)
        return median

    def logp(value, alpha, m):
        res = pt.log(alpha) + logpow(m, alpha) - logpow(value, alpha + 1.0)
        res = pt.switch(pt.ge(value, m), res, -np.inf)
        return check_parameters(
            res,
            alpha > 0,
            m > 0,
            msg="alpha > 0, m > 0",
        )

    def logcdf(value, alpha, m):
        arg = (m / value) ** alpha

        res = pt.switch(
            pt.lt(value, m),
            -np.inf,
            pt.switch(
                pt.le(arg, 1e-5),
                pt.log1p(-arg),
                pt.log(1 - arg),
            ),
        )

        return check_parameters(
            res,
            alpha > 0,
            m > 0,
            msg="alpha > 0, m > 0",
        )

    def icdf(value, alpha, m):
        res = m * pt.pow(1 - value, -1 / alpha)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            alpha > 0,
            m > 0,
            msg="alpha > 0, m > 0",
        )


@_default_transform.register(Pareto)
def pareto_default_transform(op, rv):
    return bounded_cont_transform(op, rv, Pareto.bound_args_indices)


class Cauchy(Continuous):
    r"""
    Cauchy distribution.

    Also known as the Lorentz or the Breit-Wigner distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    alpha : tensor_like of float
        Location parameter.
    beta : tensor_like of float
        Scale parameter > 0.
    """

    rv_op = cauchy

    @classmethod
    def dist(cls, alpha, beta, *args, **kwargs):
        alpha = pt.as_tensor_variable(alpha)
        beta = pt.as_tensor_variable(beta)

        return super().dist([alpha, beta], **kwargs)

    def support_point(rv, size, alpha, beta):
        alpha, _ = pt.broadcast_arrays(alpha, beta)
        if not rv_size_is_none(size):
            alpha = pt.full(size, alpha)
        return alpha

    def logp(value, alpha, beta):
        res = -pt.log(np.pi) - pt.log(beta) - pt.log1p(pt.pow((value - alpha) / beta, 2))
        return check_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )

    def logcdf(value, alpha, beta):
        res = pt.log(0.5 + pt.arctan((value - alpha) / beta) / np.pi)
        return check_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )

    def icdf(value, alpha, beta):
        res = alpha + beta * pt.tan(np.pi * (value - 0.5))
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )


class HalfCauchyRV(SymbolicRandomVariable):
    name = "halfcauchy"
    extended_signature = "[rng],[size],()->[rng],()"
    _print_name = ("HalfCauchy", "\\operatorname{HalfCauchy}")

    @classmethod
    def rv_op(cls, beta, *, size=None, rng=None):
        bt = pt.as_tensor(beta)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        next_rng, cauchy_draws = cauchy(loc=0, scale=beta, size=size, rng=rng).owner.outputs
        draws = pt.abs(cauchy_draws)

        return cls(inputs=[rng, size, beta], outputs=[next_rng, draws])(rng, size, beta)


class HalfCauchy(PositiveContinuous):
    r"""
    Half-Cauchy distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \beta) = \frac{2}{\pi \beta [1 + (\frac{x}{\beta})^2]}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    beta : tensor_like of float
        Scale parameter (beta > 0).
    """

    rv_op = HalfCauchyRV.rv_op
    rv_type = HalfCauchyRV

    @classmethod
    def dist(cls, beta, *args, **kwargs):
        beta = pt.as_tensor_variable(beta)
        return super().dist([beta], **kwargs)

    def support_point(rv, size, beta):
        if not rv_size_is_none(size):
            beta = pt.full(size, beta)
        return beta

    def logp(value, beta):
        res = pt.log(2) + _logprob_helper(Cauchy.dist(alpha=0, beta=beta), value)
        res = pt.switch(value >= 0, res, -np.inf)
        return check_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )

    def logcdf(value, beta):
        res = pt.switch(
            value < 0,
            -np.inf,
            pt.log(2 * pt.arctan(value / beta) / np.pi),
        )

        return check_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )

    def icdf(value, beta):
        res = beta * pt.tan(np.pi * (value) / 2.0)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )


class Gamma(PositiveContinuous):
    r"""
    Gamma distribution.

    Represents the sum of alpha exponentially distributed random variables,
    each of which has rate beta.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    Here, the gamma distribution is parameterized by shape (alpha) and rate (beta).

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    alpha : tensor_like of float, optional
        Shape parameter (alpha > 0).
    beta : tensor_like of float, optional
        Rate parameter (beta > 0).
    mu : tensor_like of float, optional
        Alternative shape parameter (mu > 0).
    sigma : tensor_like of float, optional
        Alternative scale parameter (sigma > 0).
    """

    # gamma is temporarily a deprecation wrapper in PyTensor
    rv_op = _gamma

    @classmethod
    def dist(cls, alpha=None, beta=None, mu=None, sigma=None, **kwargs):
        alpha, beta = cls.get_alpha_beta(alpha, beta, mu, sigma)
        alpha = pt.as_tensor_variable(alpha)
        beta = pt.as_tensor_variable(beta)
        # PyTensor gamma op is parametrized in terms of scale (1/beta)
        scale = pt.reciprocal(beta)
        return super().dist([alpha, scale], **kwargs)

    @classmethod
    def get_alpha_beta(cls, alpha=None, beta=None, mu=None, sigma=None):
        if (alpha is not None) and (beta is not None):
            pass
        elif (mu is not None) and (sigma is not None):
            # Use sign of sigma to not let negative sigma fly by
            alpha = (mu**2 / sigma**2) * pt.sign(sigma)
            beta = mu / sigma**2
        else:
            raise ValueError(
                "Incompatible parameterization. Either use "
                "alpha and beta, or mu and sigma to specify "
                "distribution."
            )

        return alpha, beta

    def support_point(rv, size, alpha, scale):
        mean = alpha * scale
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, alpha, scale):
        beta = pt.reciprocal(scale)
        res = -pt.gammaln(alpha) + logpow(beta, alpha) - beta * value + logpow(value, alpha - 1)
        res = pt.switch(pt.ge(value, 0.0), res, -np.inf)
        return check_parameters(
            res,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )

    def logcdf(value, alpha, scale):
        beta = pt.reciprocal(scale)
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log(pt.gammainc(alpha, beta * value)),
        )
        return check_parameters(res, 0 < alpha, 0 < beta, msg="alpha > 0, beta > 0")

    def icdf(value, alpha, scale):
        res = scale * gammaincinv(alpha, value)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            alpha > 0,
            scale > 0,
            msg="alpha > 0, beta > 0",
        )


class InverseGamma(PositiveContinuous):
    r"""
    Inverse gamma distribution, the reciprocal of the gamma distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-\alpha - 1}
           \exp\left(\frac{-\beta}{x}\right)

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    alpha : tensor_like of float, optional
        Shape parameter (alpha > 0).
    beta : tensor_like of float, optional
        Scale parameter (beta > 0).
    mu : tensor_like of float, optional
        Alternative shape parameter (mu > 0).
    sigma : tensor_like of float, optional
        Alternative scale parameter (sigma > 0).
    """

    rv_op = invgamma

    @classmethod
    def dist(cls, alpha=None, beta=None, mu=None, sigma=None, *args, **kwargs):
        alpha, beta = cls._get_alpha_beta(alpha, beta, mu, sigma)
        alpha = pt.as_tensor_variable(alpha)
        beta = pt.as_tensor_variable(beta)

        return super().dist([alpha, beta], **kwargs)

    def support_point(rv, size, alpha, beta):
        mean = beta / (alpha - 1.0)
        mode = beta / (alpha + 1.0)
        support_point = pt.switch(alpha > 1, mean, mode)
        if not rv_size_is_none(size):
            support_point = pt.full(size, support_point)
        return support_point

    @classmethod
    def _get_alpha_beta(cls, alpha, beta, mu, sigma):
        if alpha is not None:
            if beta is not None:
                pass
            else:
                beta = 1.0
        elif (mu is not None) and (sigma is not None):
            # Use sign of sigma to not let negative sigma fly by
            alpha = ((2 * sigma**2 + mu**2) / sigma**2) * pt.sign(sigma)
            beta = mu * (mu**2 + sigma**2) / sigma**2
        else:
            raise ValueError(
                "Incompatible parameterization. Either use "
                "alpha and (optionally) beta, or mu and sigma to specify "
                "distribution."
            )

        return alpha, beta

    def logp(value, alpha, beta):
        res = -pt.gammaln(alpha) + logpow(beta, alpha) - beta / value + logpow(value, -alpha - 1)
        res = pt.switch(pt.ge(value, 0.0), res, -np.inf)
        return check_parameters(
            res,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )

    def logcdf(value, alpha, beta):
        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log(pt.gammaincc(alpha, beta / value)),
        )

        return check_parameters(
            res,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )

    def icdf(value, alpha, beta):
        return icdf(1 / Gamma.dist(alpha, beta), value)


class ChiSquared:
    r"""
    :math:`\chi^2` distribution.

    This is the distribution from the sum of the squares of :math:`\nu` independent standard normal random variables or a special
    case of the gamma distribution with :math:`\alpha = \nu/2` and :math:`\beta = 1/2`.

    The pdf of this distribution is

    .. math::

       f(x \mid \nu) = \frac{x^{(\nu-2)/2}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)}

    Read more about the :math:`\chi^2` distribution at https://en.wikipedia.org/wiki/Chi-squared_distribution

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    nu : tensor_like of float
        Degrees of freedom (nu > 0).

    Notes
    -----
    This is implemented as a special case of the Gamma distribution.
    :math:`\chi^2(\nu) = \text{Gamma}(\alpha=\nu/2, \beta=1/2)`
    """

    def __new__(cls, name, nu, **kwargs):
        return Gamma(name, alpha=nu / 2, beta=1 / 2, **kwargs)

    @classmethod
    def dist(cls, nu, **kwargs):
        return Gamma.dist(alpha=nu / 2, beta=1 / 2, **kwargs)


class WeibullBetaRV(SymbolicRandomVariable):
    name = "weibull"
    extended_signature = "[rng],[size],(),()->[rng],()"
    _print_name = ("Weibull", "\\operatorname{Weibull}")

    @classmethod
    def rv_op(cls, alpha, beta, *, rng=None, size=None):
        alpha = pt.as_tensor(alpha)
        beta = pt.as_tensor(beta)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        if rv_size_is_none(size):
            size = implicit_size_from_params(alpha, beta, ndims_params=cls.ndims_params)

        next_rng, raw_weibull = pt.random.weibull(alpha, size=size, rng=rng).owner.outputs
        draws = beta * raw_weibull
        return cls(
            inputs=[rng, size, alpha, beta],
            outputs=[next_rng, draws],
        )(rng, size, alpha, beta)


class Weibull(PositiveContinuous):
    r"""
    Weibull distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \alpha, \beta) =
           \frac{\alpha x^{\alpha - 1}
           \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    Variance  :math:`\beta^2 \Gamma(1 + \frac{2}{\alpha} - \mu^2/\beta^2)`
    ========  ====================================================

    Parameters
    ----------
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Scale parameter (beta > 0).
    """

    rv_type = WeibullBetaRV
    rv_op = WeibullBetaRV.rv_op

    @classmethod
    def dist(cls, alpha, beta, *args, **kwargs):
        alpha = pt.as_tensor_variable(alpha)
        beta = pt.as_tensor_variable(beta)

        return super().dist([alpha, beta], *args, **kwargs)

    def support_point(rv, size, alpha, beta):
        mean = beta * pt.gamma(1 + 1 / alpha)
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logcdf(value, alpha, beta):
        a = (value / beta) ** alpha

        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            pt.log1mexp(-a),
        )

        return check_parameters(
            res,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )

    def logp(value, alpha, beta):
        res = (
            pt.log(alpha)
            - pt.log(beta)
            + (alpha - 1.0) * pt.log(value / beta)
            - pt.pow(value / beta, alpha)
        )
        res = pt.switch(pt.ge(value, 0.0), res, -np.inf)
        return check_parameters(
            res,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )

    def icdf(value, alpha, beta):
        res = beta * (-pt.log(1 - value)) ** (1 / alpha)
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            alpha > 0,
            beta > 0,
            msg="alpha > 0, beta > 0",
        )


class HalfStudentTRV(SymbolicRandomVariable):
    name = "halfstudentt"
    extended_signature = "[rng],[size],(),()->[rng],()"
    _print_name = ("HalfStudentT", "\\operatorname{HalfStudentT}")

    @classmethod
    def rv_op(cls, nu, sigma, *, size=None, rng=None):
        nu = pt.as_tensor(nu)
        sigma = pt.as_tensor(sigma)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        next_rng, t_draws = t(df=nu, scale=sigma, size=size, rng=rng).owner.outputs
        draws = pt.abs(t_draws)

        return cls(inputs=[rng, size, nu, sigma], outputs=[next_rng, draws])(rng, size, nu, sigma)


class HalfStudentT(PositiveContinuous):
    r"""
    Half Student's T distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \sigma,\nu) =
            \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
            {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
            \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    nu : tensor_like of float
        Degrees of freedom, also known as normality parameter (nu > 0).
    sigma : tensor_like of float, optional
        Scale parameter (sigma > 0). Converges to the standard deviation as nu
        increases (only required if lam is not specified). Defaults to 1.
    lam : tensor_like of float, optional
        Scale parameter (lam > 0). Converges to the precision as nu
        increases (only required if sigma is not specified). Defaults to 1.

    Examples
    --------
    .. code-block:: python

        # Only pass in one of lam or sigma, but not both.
        with pm.Model():
            x = pm.HalfStudentT("x", sigma=10, nu=10)

        with pm.Model():
            x = pm.HalfStudentT("x", lam=4, nu=10)
    """

    rv_type = HalfStudentTRV
    rv_op = HalfStudentTRV.rv_op

    @classmethod
    def dist(cls, nu, sigma=None, lam=None, *args, **kwargs):
        lam, sigma = get_tau_sigma(lam, sigma)
        return super().dist([nu, sigma], *args, **kwargs)

    def support_point(rv, size, nu, sigma):
        sigma, _ = pt.broadcast_arrays(sigma, nu)
        if not rv_size_is_none(size):
            sigma = pt.full(size, sigma)
        return sigma

    def logp(value, nu, sigma):
        res = (
            pt.log(2)
            + gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * pt.log(nu * np.pi * sigma**2)
            - (nu + 1.0) / 2.0 * pt.log1p(value**2 / (nu * sigma**2))
        )

        res = pt.switch(
            pt.lt(value, 0),
            -np.inf,
            res,
        )

        return check_parameters(
            res,
            sigma > 0,
            nu > 0,
            msg="sigma > 0, nu > 0",
        )

    def icdf(value, nu, sigma):
        # Map half-t quantiles to full StudentT quantiles:
        # F_half^{-1}(u) = F_t^{-1}((u + 1)/2; nu, mu=0, sigma)
        res = icdf(StudentT.dist(nu, sigma=sigma), (value + 1.0) / 2.0)
        res = check_icdf_value(res, value)
        return res


class ExGaussianRV(SymbolicRandomVariable):
    name = "exgaussian"
    extended_signature = "[rng],[size],(),(),()->[rng],()"
    _print_name = ("ExGaussian", "\\operatorname{ExGaussian}")

    @classmethod
    def rv_op(cls, mu, sigma, nu, *, size=None, rng=None):
        mu = pt.as_tensor(mu)
        sigma = pt.as_tensor(sigma)
        nu = pt.as_tensor(nu)
        rng = normalize_rng_param(rng)
        size = normalize_size_param(size)

        if rv_size_is_none(size):
            size = implicit_size_from_params(mu, sigma, nu, ndims_params=cls.ndims_params)

        next_rng, normal_draws = normal(loc=mu, scale=sigma, size=size, rng=rng).owner.outputs
        final_rng, exponential_draws = exponential(scale=nu, size=size, rng=next_rng).owner.outputs
        draws = normal_draws + exponential_draws

        return cls(inputs=[rng, size, mu, sigma, nu], outputs=[final_rng, draws])(
            rng, size, mu, sigma, nu
        )


class ExGaussian(Continuous):
    r"""
    Exponentially modified Gaussian distribution.

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
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float, default 0
        Mean of the normal distribution.
    sigma : tensor_like of float
        Standard deviation of the normal distribution (sigma > 0).
    nu : tensor_like of float
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

    rv_type = ExGaussianRV
    rv_op = ExGaussianRV.rv_op

    @classmethod
    def dist(cls, mu=0.0, sigma=1.0, *, nu, **kwargs):
        return super().dist([mu, sigma, nu], **kwargs)

    def support_point(rv, size, mu, sigma, nu):
        mu, nu, _ = pt.broadcast_arrays(mu, nu, sigma)
        support_point = mu + nu
        if not rv_size_is_none(size):
            support_point = pt.full(size, support_point)
        return support_point

    def logp(value, mu, sigma, nu):
        # Alogithm is adapted from dexGAUS.R from gamlss
        res = pt.switch(
            pt.gt(nu, 0.05 * sigma),
            (
                -pt.log(nu)
                + (mu - value) / nu
                + 0.5 * (sigma / nu) ** 2
                + normal_lcdf(mu + (sigma**2) / nu, sigma, value)
            ),
            log_normal(value, mean=mu, sigma=sigma),
        )
        return check_parameters(
            res,
            sigma > 0,
            nu > 0,
            msg="nu > 0, sigma > 0",
        )

    def logcdf(value, mu, sigma, nu):
        # Alogithm is adapted from pexGAUS.R from gamlss
        res = pt.switch(
            pt.gt(nu, 0.05 * sigma),
            logdiffexp(
                normal_lcdf(mu, sigma, value),
                (
                    (mu - value) / nu
                    + 0.5 * (sigma / nu) ** 2
                    + normal_lcdf(mu + (sigma**2) / nu, sigma, value)
                ),
            ),
            normal_lcdf(mu, sigma, value),
        )

        return check_parameters(
            res,
            sigma > 0,
            nu > 0,
            msg="sigma > 0, nu > 0",
        )


class VonMises(CircularContinuous):
    r"""
    Univariate VonMises distribution.

    The pdf of this distribution is

    .. math::

        f(x \mid \mu, \kappa) =
            \frac{e^{\kappa\cos(x-\mu)}}{2\pi I_0(\kappa)}

    where :math:`I_0` is the modified Bessel function of order 0.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float, default 0.0
        Mean.
    kappa : tensor_like of float, default 1.0
        Concentration (:math:`\frac{1}{\kappa}` is analogous to :math:`\sigma^2`).
    """

    rv_op = vonmises

    @classmethod
    def dist(cls, mu=0.0, kappa=1.0, *args, **kwargs):
        mu = pt.as_tensor_variable(mu)
        kappa = pt.as_tensor_variable(kappa)
        return super().dist([mu, kappa], *args, **kwargs)

    def support_point(rv, size, mu, kappa):
        mu, _ = pt.broadcast_arrays(mu, kappa)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    def logp(value, mu, kappa):
        res = kappa * pt.cos(mu - value) - pt.log(2 * np.pi) - pt.log(pt.i0(kappa))
        res = pt.switch(pt.bitwise_and(pt.ge(value, -np.pi), pt.le(value, np.pi)), res, -np.inf)
        return check_parameters(
            res,
            kappa > 0,
            msg="kappa > 0",
        )


class SkewNormalRV(RandomVariable):
    name = "skewnormal"
    signature = "(),(),()->()"
    dtype = "floatX"
    _print_name = ("SkewNormal", "\\operatorname{SkewNormal}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, alpha, size=None) -> np.ndarray:
        return np.asarray(
            stats.skewnorm.rvs(a=alpha, loc=mu, scale=sigma, size=size, random_state=rng)
        )


skewnormal = SkewNormalRV()


class SkewNormal(Continuous):
    r"""
    Univariate skew-normal distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \tau, \alpha) =
       2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float, default 0
        Location parameter.
    sigma : tensor_like of float, optional
        Scale parameter (sigma > 0).
        Defaults to 1.
    tau : tensor_like of float, optional
        Alternative scale parameter (tau > 0).
        Defaults to 1.
    alpha : tensor_like of float, default 1
        Skewness parameter.

    Notes
    -----
    When alpha=0 we recover the Normal distribution and mu becomes the mean,
    tau the precision and sigma the standard deviation. In the limit of alpha
    approaching plus/minus infinite we get a half-normal distribution.

    """

    rv_op = skewnormal

    @classmethod
    def dist(cls, alpha=1, mu=0.0, sigma=None, tau=None, *args, **kwargs):
        tau, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        alpha = pt.as_tensor_variable(alpha)
        mu = pt.as_tensor_variable(mu)
        tau = pt.as_tensor_variable(tau)
        sigma = pt.as_tensor_variable(sigma)

        return super().dist([mu, sigma, alpha], *args, **kwargs)

    def support_point(rv, size, mu, sigma, alpha):
        mean = mu + sigma * (2 / np.pi) ** 0.5 * alpha / (1 + alpha**2) ** 0.5
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, mu, sigma, alpha):
        tau, _ = get_tau_sigma(sigma=sigma)

        res = (
            pt.log(1 + pt.erf(((value - mu) * pt.sqrt(tau) * alpha) / pt.sqrt(2)))
            + (-tau * (value - mu) ** 2 + pt.log(tau / np.pi / 2.0)) / 2.0
        )

        return check_parameters(
            res,
            tau > 0,
            msg="tau > 0",
        )


class Triangular(BoundedContinuous):
    r"""
    Continuous Triangular distribution.

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
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    lower : tensor_like of float, default 0
        Lower limit.
    c : tensor_like of float, default 0.5
        Mode.
    upper : tensor_like of float, default 1
        Upper limit.
    """

    rv_op = triangular
    bound_args_indices = (2, 4)  # lower, upper

    @classmethod
    def dist(cls, lower=0, upper=1, c=0.5, *args, **kwargs):
        lower = pt.as_tensor_variable(lower)
        upper = pt.as_tensor_variable(upper)
        c = pt.as_tensor_variable(c)

        return super().dist([lower, c, upper], *args, **kwargs)

    def support_point(rv, size, lower, c, upper):
        mean = (lower + upper + c) / 3
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, lower, c, upper):
        res = pt.switch(
            pt.lt(value, c),
            pt.log(2 * (value - lower) / ((upper - lower) * (c - lower))),
            pt.log(2 * (upper - value) / ((upper - lower) * (upper - c))),
        )
        res = pt.switch(pt.bitwise_and(pt.le(lower, value), pt.le(value, upper)), res, -np.inf)
        return check_parameters(
            res,
            lower <= c,
            c <= upper,
            msg="lower <= c <= upper",
        )

    def logcdf(value, lower, c, upper):
        res = pt.switch(
            pt.le(value, lower),
            -np.inf,
            pt.switch(
                pt.le(value, c),
                pt.log(((value - lower) ** 2) / ((upper - lower) * (c - lower))),
                pt.switch(
                    pt.lt(value, upper),
                    pt.log1p(-((upper - value) ** 2) / ((upper - lower) * (upper - c))),
                    0,
                ),
            ),
        )

        return check_parameters(
            res,
            lower <= c,
            c <= upper,
            msg="lower <= c <= upper",
        )

    def icdf(value, lower, c, upper):
        res = pt.switch(
            pt.lt(value, ((c - lower) / (upper - lower))),
            lower + np.sqrt((upper - lower) * (c - lower) * value),
            upper - np.sqrt((upper - lower) * (upper - c) * (1 - value)),
        )
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            lower <= c,
            c <= upper,
            msg="lower <= c <= upper",
        )


@_default_transform.register(Triangular)
def triangular_default_transform(op, rv):
    return bounded_cont_transform(op, rv, Triangular.bound_args_indices)


class Gumbel(Continuous):
    r"""
    Univariate right-skewed Gumbel distribution.

    This distribution is typically used for modeling maximum (or extreme) values.
    Those looking to find the extreme minimum provided by the left-skewed Gumbel should
    invert the sign of all x and mu values.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \beta) = \frac{1}{\beta}e^{-(z + e^{-z})}

    where

    .. math::

        z = \frac{x - \mu}{\beta}.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    Mean      :math:`\mu + \beta\gamma`, where :math:`\gamma` is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^2}{6} \beta^2`
    ========  ==========================================

    Parameters
    ----------
    mu : tensor_like of float
        Location parameter.
    beta : tensor_like of float
        Scale parameter (beta > 0).
    """

    rv_op = gumbel

    @classmethod
    def dist(cls, mu, beta, **kwargs):
        mu = pt.as_tensor_variable(mu)
        beta = pt.as_tensor_variable(beta)

        return super().dist([mu, beta], **kwargs)

    def support_point(rv, size, mu, beta):
        mean = mu + beta * np.euler_gamma
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, mu, beta):
        z = (value - mu) / beta
        res = -z - pt.exp(-z) - pt.log(beta)
        return check_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )

    def logcdf(value, mu, beta):
        res = -pt.exp(-(value - mu) / beta)

        return check_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )

    def icdf(value, mu, beta):
        res = mu - beta * pt.log(-pt.log(value))
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            beta > 0,
            msg="beta > 0",
        )


class RiceRV(RandomVariable):
    name = "rice"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Rice", "\\operatorname{Rice}")

    @classmethod
    def rng_fn(cls, rng, b, sigma, size=None) -> np.ndarray:
        return np.asarray(stats.rice.rvs(b=b, scale=sigma, size=size, random_state=rng))


rice = RiceRV()


class Rice(PositiveContinuous):
    r"""
    Rice distribution.

    .. math::

       f(x\mid \nu ,\sigma )=
       {\frac  {x}{\sigma ^{2}}}\exp
       \left({\frac  {-(x^{2}+\nu ^{2})}{2\sigma ^{2}}}\right)I_{0}\left({\frac  {x\nu }{\sigma ^{2}}}\right),

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0, 8, 500)
        nus = [0., 0., 4., 4.]
        sigmas = [1., 2., 1., 2.]
        for nu, sigma in  zip(nus, sigmas):
            pdf = st.rice.pdf(x, nu / sigma, scale=sigma)
            plt.plot(x, pdf, label=r'$\nu$ = {}, $\sigma$ = {}'.format(nu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\sigma {\sqrt  {\pi /2}}\,\,L_{{1/2}}(-\nu ^{2}/2\sigma ^{2})`
    Variance  :math:`2\sigma ^{2}+\nu ^{2}-{\frac  {\pi \sigma ^{2}}{2}}L_{{1/2}}^{2}\left({\frac  {-\nu ^{2}}{2\sigma ^{2}}}\right)`
    ========  ==============================================================


    Parameters
    ----------
    nu : tensor_like of float, optional
        Noncentrality parameter (only required if b is not specified).
    sigma : tensor_like of float, default 1
        scale parameter.
    b : tensor_like of float, optional
        Shape parameter (alternative to nu, only required if nu is not specified).

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

    rv_op = rice

    @classmethod
    def dist(cls, nu=None, sigma=None, b=None, *args, **kwargs):
        nu, b, sigma = cls.get_nu_b(nu, b, sigma)
        b = pt.as_tensor_variable(b)
        sigma = pt.as_tensor_variable(sigma)

        return super().dist([b, sigma], *args, **kwargs)

    @classmethod
    def get_nu_b(cls, nu, b, sigma):
        if sigma is None:
            sigma = 1.0
        if nu is None and b is not None:
            nu = b * sigma
            return nu, b, sigma
        elif nu is not None and b is None:
            b = nu / sigma
            return nu, b, sigma
        raise ValueError("Rice distribution must specify either nu or b.")

    def support_point(rv, size, nu, sigma):
        nu_sigma_ratio = -(nu**2) / (2 * sigma**2)
        mean = (
            sigma
            * np.sqrt(np.pi / 2)
            * pt.exp(nu_sigma_ratio / 2)
            * (
                (1 - nu_sigma_ratio) * pt.i0(-nu_sigma_ratio / 2)
                - nu_sigma_ratio * pt.i1(-nu_sigma_ratio / 2)
            )
        )

        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, b, sigma):
        x = value / sigma

        res = pt.switch(
            pt.le(value, 0),
            -np.inf,
            pt.log(x * pt.exp((-(x - b) * (x - b)) / 2) * i0e(x * b) / sigma),
        )

        return check_parameters(
            res,
            sigma >= 0,
            b >= 0,
            msg="sigma >= 0, b >= 0",
        )


class Logistic(Continuous):
    r"""
    Logistic distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, s) =
           \frac{\exp\left(-\frac{x - \mu}{s}\right)}{s \left(1 + \exp\left(-\frac{x - \mu}{s}\right)\right)^2}

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float, default 0
        Mean.
    s : tensor_like of float, default 1
        Scale (s > 0).
    """

    rv_op = logistic

    @classmethod
    def dist(cls, mu=0.0, s=1.0, *args, **kwargs):
        mu = pt.as_tensor_variable(mu)
        s = pt.as_tensor_variable(s)
        return super().dist([mu, s], *args, **kwargs)

    def support_point(rv, size, mu, s):
        mu, _ = pt.broadcast_arrays(mu, s)
        if not rv_size_is_none(size):
            mu = pt.full(size, mu)
        return mu

    def logp(value, mu, s):
        z = (value - mu) / s
        res = -z - pt.log(s) - 2.0 * pt.log1p(pt.exp(-z))
        return check_parameters(
            res,
            s > 0,
            msg="s > 0",
        )

    def logcdf(value, mu, s):
        res = -pt.log1pexp(-(value - mu) / s)

        return check_parameters(
            res,
            s > 0,
            msg="s > 0",
        )

    def icdf(value, mu, s):
        res = mu + s * pt.log(value / (1 - value))
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            s > 0,
            msg="s > 0",
        )


class LogitNormal:
    r"""
    Logit-Normal distribution.

    The pdf of this distribution is

    .. math::
       f(x \mid \mu, \tau) =
           \frac{1}{x(1-x)} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (logit(x)-\mu)^2 \right\}


    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        from scipy.special import logit
        import arviz as az
        plt.style.use('arviz-darkgrid')
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
    mu : tensor_like of float, default 0
        Location parameter.
    sigma : tensor_like of float, optional
        Scale parameter (sigma > 0).
        Defaults to 1.
    tau : tensor_like of float, optional
        Scale parameter (tau > 0).
        Defaults to 1.
    """

    @staticmethod
    def logitnormal_dist(mu, sigma, size):
        return invlogit(Normal.dist(mu=mu, sigma=sigma, size=size))

    def __new__(cls, name, mu=0, sigma=None, tau=None, **kwargs):
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        return CustomDist(
            name,
            mu,
            sigma,
            dist=cls.logitnormal_dist,
            class_name="LogitNormal",
            **kwargs,
        )

    @classmethod
    def dist(cls, mu=0, sigma=None, tau=None, **kwargs):
        _, sigma = get_tau_sigma(tau=tau, sigma=sigma)
        return CustomDist.dist(
            mu, sigma, dist=cls.logitnormal_dist, class_name="LogitNormal", **kwargs
        )


def _interpolated_argcdf(p, pdf, cdf, x):
    if np.prod(cdf.shape[:-1]) != 1 or np.prod(pdf.shape[:-1]) != 1 or np.prod(x.shape[:-1]) != 1:
        raise NotImplementedError(
            "Function not implemented for batched points. "
            "Open an issue in https://github.com/pymc-devs/pymc if you need this functionality"
        )
    cdf = cdf.squeeze(tuple(range(cdf.ndim - 1)))
    pdf = pdf.squeeze(tuple(range(pdf.ndim - 1)))
    x = x.squeeze(tuple(range(x.ndim - 1)))

    index = np.searchsorted(cdf, p) - 1
    slope = (pdf[index + 1] - pdf[index]) / (x[index + 1] - x[index])

    # First term (constant) of the Taylor expansion around slope = 0
    small_slopes = np.where(
        np.abs(pdf[index]) <= 1e-8, np.zeros(index.shape), (p - cdf[index]) / pdf[index]
    )

    # This warning happens when we divide by slope = 0: we can ignore it
    # because the other result will be returned
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*invalid value encountered in.*", RuntimeWarning)
        large_slopes = (
            -pdf[index] + np.sqrt(pdf[index] ** 2 + 2 * slope * (p - cdf[index]))
        ) / slope

    return x[index] + np.where(np.abs(slope) <= 1e-8, small_slopes, large_slopes)


class InterpolatedRV(RandomVariable):
    name = "interpolated"
    signature = "(x),(x),(x)->()"
    dtype = "floatX"
    _print_name = ("Interpolated", "\\operatorname{Interpolated}")

    @classmethod
    def rng_fn(cls, rng, x, pdf, cdf, size=None) -> np.ndarray:
        p = rng.uniform(size=size)
        return np.asarray(_interpolated_argcdf(p, pdf, cdf, x))


interpolated = InterpolatedRV()


class Interpolated(BoundedContinuous):
    r"""
    Univariate linear interpolation of pdf evaluated on some lattice of points.

    The lattice can be uneven, so the steps between different points can have
    different size and it is possible to vary the precision between regions
    of the support.

    The probability density function values don not have to be normalized, as the
    interpolated density is any way normalized to make the total probability
    equal to $1$.

    Both parameters ``x_points`` and values ``pdf_points`` are not variables, but
    plain array-like objects, so they are constant and cannot be sampled.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import pymc as pm
        import arviz as az
        from scipy.stats import gamma
        plt.style.use('arviz-darkgrid')
        rv = gamma(1.99)
        x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 1000)
        points = np.linspace(x[0], x[-1], 50)
        pdf = rv.pdf(points)
        interpolated = pm.Interpolated.dist(points, pdf)
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, rv.pdf(x), 'C0', linestyle = '--',  label='Original Gamma pdf', alpha=0.8, lw=2)
        ax.plot(points, pdf, color='black', marker='o', label='Lattice Points', alpha=0.5, linestyle='')
        ax.plot(x, np.exp(pm.logp(interpolated, x).eval()), 'C1', label='Interpolated pdf', alpha=0.8, lw=3)
        r = pm.draw(interpolated, draws=1000)
        ax.hist(r, density=True, alpha=0.4, align ='mid', color='grey')
        ax.legend(loc='best', frameon=False)
        plt.show()

    ========  ===========================================
    Support   :math:`x \in [x\_points[0], x\_points[-1]]`
    ========  ===========================================

    Parameters
    ----------
    x_points : array_like
        A monotonically growing list of values. Must be non-symbolic.
    pdf_points : array_like
        Probability density function evaluated on lattice ``x_points``. Must
        be non-symbolic.
    """

    rv_op = interpolated

    @classmethod
    def dist(cls, x_points, pdf_points, *args, **kwargs):
        interp = InterpolatedUnivariateSpline(x_points, pdf_points, k=1, ext="zeros")

        Z = interp.integral(x_points[0], x_points[-1])
        cdf_points = interp.antiderivative()(x_points) / Z
        pdf_points = pdf_points / Z

        x_points = pt.constant(x_points)
        pdf_points = pt.constant(pdf_points)
        cdf_points = pt.constant(cdf_points)

        # lower = pt.as_tensor_variable(x_points[0])
        # upper = pt.as_tensor_variable(x_points[-1])
        # median = _interpolated_argcdf(0.5, pdf_points, cdf_points, x_points)

        return super().dist([x_points, pdf_points, cdf_points], **kwargs)

    def support_point(rv, size, x_points, pdf_points, cdf_points):
        """Estimates the expectation integral using the trapezoid rule; cdf_points are not used."""
        x_fx = pt.mul(x_points, pdf_points)  # x_i * f(x_i) for all xi's in x_points
        support_point = (
            pt.sum(pt.mul(pt.diff(x_points, axis=-1), x_fx[..., 1:] + x_fx[..., :-1])) / 2
        )

        if not rv_size_is_none(size):
            support_point = pt.full(size, support_point)

        return support_point

    def logp(value, x_points, pdf_points, cdf_points):
        # x_points and pdf_points are expected to be non-symbolic arrays wrapped
        # within a tensor.constant. We use the .data method to retrieve them
        interp = InterpolatedUnivariateSpline(x_points.data, pdf_points.data, k=1, ext="zeros")
        Z = interp.integral(x_points.data[..., 0], x_points.data[..., -1])

        # interp and Z are converted to symbolic variables here
        interp_op = SplineWrapper(interp)
        Z = pt.constant(Z)

        return pt.log(interp_op(value) / Z)


@_default_transform.register(Interpolated)
def interpolated_default_transform(op, rv):
    def transform_params(*params):
        _, _, x_points, _, _ = params
        return x_points[..., 0], x_points[..., -1]

    return transforms.Interval(bounds_fn=transform_params)


class MoyalRV(RandomVariable):
    name = "moyal"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("Moyal", "\\operatorname{Moyal}")

    @classmethod
    def rng_fn(cls, rng, mu, sigma, size=None) -> np.ndarray:
        return np.asarray(stats.moyal.rvs(mu, sigma, size=size, random_state=rng))


moyal = MoyalRV()


class Moyal(Continuous):
    r"""
    Moyal distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(z + e^{-z}\right)},

    where

    .. math::

       z = \frac{x-\mu}{\sigma}.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        import scipy.stats as st
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(-10, 20, 200)
        mus = [-1., 0., 4.]
        sigmas = [2., 2., 4.]
        for mu, sigma in zip(mus, sigmas):
            pdf = st.moyal.pdf(x, loc=mu, scale=sigma)
            plt.plot(x, pdf, label=r'$\mu$ = {}, $\sigma$ = {}'.format(mu, sigma))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  ==============================================================
    Support   :math:`x \in (-\infty, \infty)`
    Mean      :math:`\mu + \sigma\left(\gamma + \log 2\right)`, where :math:`\gamma` is the Euler-Mascheroni constant
    Variance  :math:`\frac{\pi^{2}}{2}\sigma^{2}`
    ========  ==============================================================

    Parameters
    ----------
    mu : tensor_like of float, default 0
        Location parameter.
    sigma : tensor_like of float, default 1
        Scale parameter (sigma > 0).
    """

    rv_op = moyal

    @classmethod
    def dist(cls, mu=0, sigma=1.0, *args, **kwargs):
        mu = pt.as_tensor_variable(mu)
        sigma = pt.as_tensor_variable(sigma)

        return super().dist([mu, sigma], *args, **kwargs)

    def support_point(rv, size, mu, sigma):
        mean = mu + sigma * (np.euler_gamma + pt.log(2))

        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, mu, sigma):
        scaled = (value - mu) / sigma
        res = -(1 / 2) * (scaled + pt.exp(-scaled)) - pt.log(sigma) - (1 / 2) * pt.log(2 * np.pi)
        return check_parameters(
            res,
            sigma > 0,
            msg="sigma > 0",
        )

    def logcdf(value, mu, sigma):
        scaled = (value - mu) / sigma
        res = pt.log(pt.erfc(pt.exp(-scaled / 2) * (2**-0.5)))
        return check_parameters(
            res,
            sigma > 0,
            msg="sigma > 0",
        )

    def icdf(value, mu, sigma):
        res = sigma * -pt.log(2.0 * pt.erfcinv(value) ** 2) + mu
        res = check_icdf_value(res, value)
        return check_icdf_parameters(
            res,
            sigma > 0,
            msg="sigma > 0",
        )


class PolyaGammaRV(RandomVariable):
    """Polya-Gamma random variable."""

    name = "polyagamma"
    signature = "(),()->()"
    dtype = "floatX"
    _print_name = ("PG", "\\operatorname{PG}")

    def __call__(self, h=1.0, z=0.0, size=None, **kwargs):
        return super().__call__(h, z, size=size, **kwargs)

    @classmethod
    def rng_fn(cls, rng, h, z, size=None) -> np.ndarray:
        """
        Generate a random sample from the distribution with the given parameters.

        Parameters
        ----------
        rng : Generator
        h : scalar or sequence
            The shape parameter of the distribution.
        z : scalar or sequence
            The exponential tilting parameter.
        size : int or tuple of ints, optional
            The number of elements to draw from the distribution. If size is
            ``None`` (default) then a single value is returned. If a tuple of
            integers is passed, the returned array will have the same shape.
            If the element(s) of size is not an integer type, it will be truncated
            to the largest integer smaller than its value (e.g (2.1, 1) -> (2, 1)).
            This parameter only applies if `h` and `z` are scalars.
        """
        # random_polyagamma needs explicit size to work correctly
        if size is None:
            size = np.broadcast_shapes(h.shape, z.shape)
        return np.asarray(
            random_polyagamma(h, z, size=size, random_state=rng).astype(pytensor.config.floatX)
        )


polyagamma = PolyaGammaRV()


class _PolyaGammaLogDistFunc(Op):
    __props__ = ("get_pdf",)

    def __init__(self, get_pdf=False):
        self.get_pdf = get_pdf

    def make_node(self, x, h, z):
        x = pt.as_tensor_variable(x)
        h = pt.as_tensor_variable(h)
        z = pt.as_tensor_variable(z)
        bshape = broadcast_shape(x, h, z)
        shape = [None] * len(bshape)
        return Apply(self, [x, h, z], [pt.TensorType(pytensor.config.floatX, shape)()])

    def perform(self, node, ins, outs):
        x, h, z = ins[0], ins[1], ins[2]
        outs[0][0] = (
            polyagamma_pdf(x, h, z, return_log=True)
            if self.get_pdf
            else polyagamma_cdf(x, h, z, return_log=True)
        ).astype(pytensor.config.floatX)


class PolyaGamma(PositiveContinuous):
    r"""
    The Polya-Gamma distribution.

    The distribution is parametrized by ``h`` (shape parameter) and ``z``
    (exponential tilting parameter). The pdf of this distribution is

    .. math::

       f(x \mid h, z) = cosh^h(\frac{z}{2})e^{-\frac{1}{2}xz^2}f(x \mid h, 0),

    where :math:`f(x \mid h, 0)` is the pdf of a :math:`PG(h, 0)` variable.
    Notice that the pdf of this distribution is expressed as an alternating-sign
    sum of inverse-Gaussian densities.

    .. math::

        X = \Sigma_{k=1}^{\infty}\frac{Ga(h, 1)}{d_k},

    where :math:`d_k = 2(k - 0.5)^2\pi^2 + z^2/2`, :math:`Ga(h, 1)` is a gamma
    random variable with shape  parameter ``h`` and scale parameter ``1``.

    .. plot::
        :context: close-figs

        import matplotlib.pyplot as plt
        import numpy as np
        from polyagamma import polyagamma_pdf
        import arviz as az
        plt.style.use('arviz-darkgrid')
        x = np.linspace(0.01, 5, 500);x.sort()
        hs = [1., 5., 10., 15.]
        zs = [0.] * 4
        for h, z in zip(hs, zs):
            pdf = polyagamma_pdf(x, h=h, z=z)
            plt.plot(x, pdf, label=r'$h$ = {}, $z$ = {}'.format(h, z))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\dfrac{h}{4}` if :math:`z=0`, :math:`\dfrac{tanh(z/2)h}{2z}` otherwise.
    Variance  :math:`0.041666688h` if :math:`z=0`, :math:`\dfrac{h(sinh(z) - z)(1 - tanh^2(z/2))}{4z^3}` otherwise.
    ========  =============================

    Parameters
    ----------
    h : tensor_like of float, default 1
        The shape parameter of the distribution (h > 0).
    z : tensor_like of float, default 0
        The exponential tilting parameter of the distribution.

    Examples
    --------
    .. code-block:: python

        rng = np.random.default_rng()
        with pm.Model():
            x = pm.PolyaGamma("x", h=1, z=5.5)
        with pm.Model():
            x = pm.PolyaGamma("x", h=25, z=-2.3, rng=rng, size=(100, 5))

    References
    ----------
    .. [1] Polson, Nicholas G., James G. Scott, and Jesse Windle.
           "Bayesian inference for logistic models using Pólya–Gamma latent
           variables." Journal of the American statistical Association
           108.504 (2013): 1339-1349.
    .. [2] Windle, Jesse, Nicholas G. Polson, and James G. Scott.
           "Sampling Polya-Gamma random variates: alternate and approximate
           techniques." arXiv preprint arXiv:1405.0506 (2014).
    .. [3] Luc Devroye. "On exact simulation algorithms for some distributions
           related to Jacobi theta functions." Statistics & Probability Letters,
           Volume 79, Issue 21, (2009): 2251-2259.
    .. [4] Windle, J. (2013). Forecasting high-dimensional, time-varying
           variance-covariance matrices with high-frequency data and sampling
           Pólya-Gamma random variates for posterior distributions derived
           from logistic likelihoods.(PhD thesis). Retrieved from
           http://hdl.handle.net/2152/21842
    """

    rv_op = polyagamma

    @classmethod
    def dist(cls, h=1.0, z=0.0, **kwargs):
        h = pt.as_tensor_variable(h)
        z = pt.as_tensor_variable(z)

        msg = f"The variable {h} specified for PolyaGamma has non-positive "
        msg += "values, making it unsuitable for this parameter."
        Assert(msg)(h, pt.all(pt.gt(h, 0.0)))

        return super().dist([h, z], **kwargs)

    def support_point(rv, size, h, z):
        mean = pt.switch(pt.eq(z, 0), h / 4, tanh(z / 2) * (h / (2 * z)))
        if not rv_size_is_none(size):
            mean = pt.full(size, mean)
        return mean

    def logp(value, h, z):
        res = pt.switch(
            pt.le(value, 0),
            -np.inf,
            _PolyaGammaLogDistFunc(get_pdf=True)(value, h, z),
        )
        return check_parameters(
            res,
            h > 0,
            msg="h > 0",
        )

    def logcdf(value, h, z):
        res = pt.switch(
            pt.le(value, 0),
            -np.inf,
            _PolyaGammaLogDistFunc(get_pdf=False)(value, h, z),
        )

        return check_parameters(
            res,
            h > 0,
            msg="h > 0",
        )
