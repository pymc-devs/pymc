#   Copyright 2024 The PyMC Developers
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

"""
Created on Mar 7, 2011.

@author: johnsalvatier
"""

import warnings

from collections.abc import Iterable
from functools import partial

import numpy as np
import pytensor
import pytensor.tensor as pt
import scipy.linalg
import scipy.stats

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.op import Op
from pytensor.scalar import UnaryScalarOp, upgrade_to_float_no_complex
from pytensor.tensor import gammaln
from pytensor.tensor.elemwise import Elemwise

from pymc.distributions.shape_utils import to_tuple
from pymc.logprob.utils import CheckParameterValue
from pymc.pytensorf import floatX

f = floatX
c = -0.5 * np.log(2.0 * np.pi)
_beta_clip_values = {
    dtype: (np.nextafter(0, 1, dtype=dtype), np.nextafter(1, 0, dtype=dtype))
    for dtype in ["float16", "float32", "float64"]
}


def check_parameters(
    expr: Variable,
    *conditions: Iterable[Variable],
    msg: str = "",
    can_be_replaced_by_ninf: bool = True,
):
    """Wrap an expression in a CheckParameterValue that asserts several conditions are met.

    When conditions are not met a ParameterValueError assertion is raised,
    with an optional custom message defined by `msg`.

    When the flag `can_be_replaced_by_ninf` is True (default), PyMC is allowed to replace the
    assertion by a switch(condition, expr, -inf). This is used for logp graphs!

    Note that check_parameter should not be used to enforce the logic of the
    expression under the normal parameter support as it can be disabled by the user via
    check_bounds = False in pm.Model()
    """
    # pt.all does not accept True/False, but accepts np.array(True)/np.array(False)
    conditions_ = [
        cond if (cond is not True and cond is not False) else np.array(cond) for cond in conditions
    ]
    all_true_scalar = pt.all([pt.all(cond) for cond in conditions_])

    return CheckParameterValue(msg, can_be_replaced_by_ninf)(expr, all_true_scalar)


check_icdf_parameters = partial(check_parameters, can_be_replaced_by_ninf=False)


def check_icdf_value(expr: Variable, value: Variable) -> Variable:
    """Wrap icdf expression in nan switch for value."""
    value = pt.as_tensor_variable(value)
    expr = pt.switch(
        pt.and_(value >= 0, value <= 1),
        expr,
        np.nan,
    )
    expr.name = "0 <= value <= 1"
    return expr


def logpow(x, m):
    """Calculate log(x**m) since m*log(x) will fail when m, x = 0."""
    # return m * log(x)
    return pt.switch(pt.eq(x, 0), pt.switch(pt.eq(m, 0), 0.0, -np.inf), m * pt.log(x))


def factln(n):
    return gammaln(n + 1)


def binomln(n, k):
    return factln(n) - factln(k) - factln(n - k)


def betaln(x, y):
    return gammaln(x) + gammaln(y) - gammaln(x + y)


def std_cdf(x):
    """Calculate the standard normal cumulative distribution function."""
    return 0.5 + 0.5 * pt.erf(x / pt.sqrt(2.0))


def normal_lcdf(mu, sigma, x):
    """Compute the log of the cumulative density function of the normal."""
    z = (x - mu) / sigma
    return pt.switch(
        pt.lt(z, -1.0),
        pt.log(pt.erfcx(-z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
        pt.log1p(-pt.erfc(z / pt.sqrt(2.0)) / 2.0),
    )


def normal_lccdf(mu, sigma, x):
    z = (x - mu) / sigma
    return pt.switch(
        pt.gt(z, 1.0),
        pt.log(pt.erfcx(z / pt.sqrt(2.0)) / 2.0) - pt.sqr(z) / 2.0,
        pt.log1p(-pt.erfc(-z / pt.sqrt(2.0)) / 2.0),
    )


def log_diff_normal_cdf(mu, sigma, x, y):
    r"""
    Compute :math:`\\log(\\Phi(\frac{x - \\mu}{\\sigma}) - \\Phi(\frac{y - \\mu}{\\sigma}))` safely in log space.

    Parameters
    ----------
    mu: float
        mean
    sigma: float
        std

    x: float

    y: float
        must be strictly less than x.

    Returns
    -------
    log (\\Phi(x) - \\Phi(y))

    """
    x = (x - mu) / sigma / pt.sqrt(2.0)
    y = (y - mu) / sigma / pt.sqrt(2.0)

    # To stabilize the computation, consider these three regions:
    # 1) x > y > 0 => Use erf(x) = 1 - e^{-x^2} erfcx(x) and erf(y) =1 - e^{-y^2} erfcx(y)
    # 2) 0 > x > y => Use erf(x) = e^{-x^2} erfcx(-x) and erf(y) = e^{-y^2} erfcx(-y)
    # 3) x > 0 > y => Naive formula log( (erf(x) - erf(y)) / 2 ) works fine.
    return pt.log(0.5) + pt.switch(
        pt.gt(y, 0),
        -pt.square(y) + pt.log(pt.erfcx(y) - pt.exp(pt.square(y) - pt.square(x)) * pt.erfcx(x)),
        pt.switch(
            pt.lt(x, 0),  # 0 > x > y
            -pt.square(x)
            + pt.log(pt.erfcx(-x) - pt.exp(pt.square(x) - pt.square(y)) * pt.erfcx(-y)),
            pt.log(pt.erf(x) - pt.erf(y)),  # x >0 > y
        ),
    )


def sigma2rho(sigma):
    """Convert `sigma` into `rho` with PyTensor.

    :math:`mu + sigma*e = mu + log(1+exp(rho))*e`.
    """
    return pt.log(pt.exp(pt.abs(sigma)) - 1.0)


def rho2sigma(rho):
    """Convert `rho` to `sigma` with PyTensor.

    :math:`mu + sigma*e = mu + log(1+exp(rho))*e`.
    """
    return pt.softplus(rho)


rho2sd = rho2sigma
sd2rho = sigma2rho


def log_normal(x, mean, **kwargs):
    """
    Calculate logarithm of normal distribution at point `x` with given `mean` and `std`.

    Parameters
    ----------
    x: Tensor
        point of evaluation
    mean: Tensor
        mean of normal distribution
    kwargs: one of parameters `{sigma, tau, w, rho}`

    Notes
    -----
    There are four variants for density parametrization.
    They are:
        1) standard deviation - `std`
        2) `w`, logarithm of `std` :math:`w = log(std)`
        3) `rho` that follows this equation :math:`rho = log(exp(std) - 1)`
        4) `tau` that follows this equation :math:`tau = std^{-1}`
    ----
    """
    sigma = kwargs.get("sigma")
    w = kwargs.get("w")
    rho = kwargs.get("rho")
    tau = kwargs.get("tau")
    eps = kwargs.get("eps", 0.0)
    check = sum(a is not None for a in [sigma, w, rho, tau])
    if check > 1:
        raise ValueError("more than one required kwarg is passed")
    if check == 0:
        raise ValueError("none of required kwarg is passed")
    if sigma is not None:
        std = sigma
    elif w is not None:
        std = pt.exp(w)
    elif rho is not None:
        std = rho2sigma(rho)
    else:
        std = tau ** (-1)
    std += f(eps)
    return f(c) - pt.log(pt.abs(std)) - (x - mean) ** 2 / (2.0 * std**2)


class SplineWrapper(Op):
    """Creates a PyTensor operation from scipy.interpolate.UnivariateSpline."""

    __props__ = ("spline",)

    def __init__(self, spline):
        self.spline = spline

    def make_node(self, x):
        x = pt.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    @property
    def grad_op(self):
        if not hasattr(self, "_grad_op"):
            try:
                self._grad_op = SplineWrapper(self.spline.derivative())
            except ValueError:
                self._grad_op = None

        if self._grad_op is None:
            raise NotImplementedError("Spline of order 0 is not differentiable")
        return self._grad_op

    def perform(self, node, inputs, output_storage):
        (x,) = inputs
        output_storage[0][0] = np.asarray(self.spline(x), dtype=x.dtype)

    def grad(self, inputs, grads):
        (x,) = inputs
        (x_grad,) = grads

        return [x_grad * self.grad_op(x)]


class I1e(UnaryScalarOp):
    """Modified Bessel function of the first kind of order 1, exponentially scaled."""

    nfunc_spec = ("scipy.special.i1e", 1, 1)

    def impl(self, x):
        return scipy.special.i1e(x)


i1e_scalar = I1e(upgrade_to_float_no_complex, name="i1e")
i1e = Elemwise(i1e_scalar, name="Elemwise{i1e,no_inplace}")


class I0e(UnaryScalarOp):
    """Modified Bessel function of the first kind of order 0, exponentially scaled."""

    nfunc_spec = ("scipy.special.i0e", 1, 1)

    def impl(self, x):
        return scipy.special.i0e(x)

    def grad(self, inp, grads):
        (x,) = inp
        (gz,) = grads
        return (gz * (i1e_scalar(x) - pytensor.scalar.sign(x) * i0e_scalar(x)),)


i0e_scalar = I0e(upgrade_to_float_no_complex, name="i0e")
i0e = Elemwise(i0e_scalar, name="Elemwise{i0e,no_inplace}")


def random_choice(p, size):
    """Return draws from categorical probability functions.

    Parameters
    ----------
    p : array
        Probability of each class. If p.ndim > 1, the last axis is
        interpreted as the probability of each class, and numpy.random.choice
        is iterated for every other axis element.
    size : int or tuple
        Shape of the desired output array. If p is multidimensional, size
        should broadcast with p.shape[:-1].

    Returns
    -------
    random_sample : array

    """
    k = p.shape[-1]

    if p.ndim > 1:
        # If p is an nd-array, the last axis is interpreted as the class
        # probability. We must iterate over the elements of all the other
        # dimensions.
        # We first ensure that p is broadcasted to the output's shape
        size = (*to_tuple(size), 1)
        p = np.broadcast_arrays(p, np.empty(size))[0]
        out_shape = p.shape[:-1]
        # np.random.choice accepts 1D p arrays, so we semiflatten p to
        # iterate calls using the last axis as the category probabilities
        p = np.reshape(p, (-1, p.shape[-1]))
        samples = np.array([np.random.choice(k, p=p_) for p_ in p])
        # We reshape to the desired output shape
        samples = np.reshape(samples, out_shape)
    else:
        samples = np.random.choice(k, p=p, size=size)
    return samples


def zvalue(value, sigma, mu):
    """Calculate the z-value for a normal distribution."""
    return (value - mu) / sigma


def clipped_beta_rvs(a, b, size=None, random_state=None, dtype="float64"):
    """Draw beta distributed random samples in the open :math:`(0, 1)` interval.

    The samples are generated with ``scipy.stats.beta.rvs``, but any value that
    is equal to 0 or 1 will be shifted towards the next floating point in the
    interval :math:`[0, 1]`, depending on the floating point precision that is
    given by ``dtype``.

    Parameters
    ----------
    a : float or array_like of floats
        Alpha, strictly positive (>0).
    b : float or array_like of floats
        Beta, strictly positive (>0).
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  If size is ``None`` (default),
        a single value is returned if ``a`` and ``b`` are both scalars.
        Otherwise, ``np.broadcast(a, b).size`` samples are drawn.
    dtype : str or dtype instance
        The floating point precision that the samples should have. This also
        determines the value that will be used to shift any samples returned
        by the numpy random number generator that are zero or one.

    Returns
    -------
    out : ndarray or scalar
        Drawn samples from the parameterized beta distribution. The scipy
        implementation can yield values that are equal to zero or one. We
        assume the support of the Beta distribution to be in the open interval
        :math:`(0, 1)`, so we shift any sample that is equal to 0 to
        ``np.nextafter(0, 1, dtype=dtype)`` and any sample that is equal to 1
        is shifted to ``np.nextafter(1, 0, dtype=dtype)``.

    """
    out = scipy.stats.beta.rvs(a, b, size=size, random_state=random_state).astype(dtype)
    lower, upper = _beta_clip_values[dtype]
    return np.maximum(np.minimum(out, upper), lower)


def multigammaln(a, p):
    """Multivariate Log Gamma.

    Parameters
    ----------
    a: tensor like
    p: int
       degrees of freedom. p > 0
    """
    i = pt.arange(1, p + 1)
    return p * (p - 1) * pt.log(np.pi) / 4.0 + pt.sum(gammaln(a + (1.0 - i) / 2.0), axis=0)


def log_i0(x):
    """Calculate the logarithm of the 0 order modified Bessel function of the first kind."""
    return pt.switch(
        pt.lt(x, 5),
        pt.log1p(
            x**2.0 / 4.0
            + x**4.0 / 64.0
            + x**6.0 / 2304.0
            + x**8.0 / 147456.0
            + x**10.0 / 14745600.0
            + x**12.0 / 2123366400.0
        ),
        x
        - 0.5 * pt.log(2.0 * np.pi * x)
        + pt.log1p(
            1.0 / (8.0 * x)
            + 9.0 / (128.0 * x**2.0)
            + 225.0 / (3072.0 * x**3.0)
            + 11025.0 / (98304.0 * x**4.0)
        ),
    )


def incomplete_beta(a, b, value):
    warnings.warn(
        "incomplete_beta has been deprecated. Use pytensor.tensor.betainc instead.",
        FutureWarning,
        stacklevel=2,
    )
    return pt.betainc(a, b, value)
