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
import abc
import warnings

from functools import singledispatch
from typing import Callable, Optional, Tuple, Union

import numpy as np

# ignore mypy error because it somehow considers that
# "numpy.core.numeric has no attribute normalize_axis_tuple"
from numpy.core.numeric import normalize_axis_tuple  # type: ignore
from pytensor import scan
from pytensor import tensor as pt
from pytensor.gradient import jacobian
from pytensor.graph import Op, Variable
from pytensor.tensor import TensorVariable

__all__ = [
    "Transform",
    "simplex",
    "logodds",
    "Interval",
    "log_exp_m1",
    "ordered",
    "log",
    "sum_to_1",
    "circular",
    "CholeskyCovPacked",
    "Chain",
    "ZeroSumTransform",
]

from pymc.pytensorf import floatX


def __getattr__(name):
    if name in ("univariate_ordered", "multivariate_ordered"):
        warnings.warn(f"{name} has been deprecated, use ordered instead.", FutureWarning)
        return ordered

    if name in ("univariate_sum_to_1, multivariate_sum_to_1"):
        warnings.warn(f"{name} has been deprecated, use sum_to_1 instead.", FutureWarning)
        return sum_to_1

    if name == "RVTransform":
        warnings.warn("RVTransform has been renamed to Transform", FutureWarning)
        return Transform

    raise AttributeError(f"module {__name__} has no attribute {name}")


@singledispatch
def _default_transform(op: Op, rv: TensorVariable):
    """Return default transform for a given Distribution `Op`"""
    return None


class Transform(abc.ABC):
    ndim_supp: Optional[int] = None

    @abc.abstractmethod
    def forward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Apply the transformation."""

    @abc.abstractmethod
    def backward(
        self, value: TensorVariable, *inputs: Variable
    ) -> Union[TensorVariable, Tuple[TensorVariable, ...]]:
        """Invert the transformation. Multiple values may be returned when the
        transformation is not 1-to-1"""

    def log_jac_det(self, value: TensorVariable, *inputs) -> TensorVariable:
        """Construct the log of the absolute value of the Jacobian determinant."""
        if self.ndim_supp not in (0, 1):
            raise NotImplementedError(
                f"RVTransform default log_jac_det only implemented for ndim_supp in (0, 1), got {self.ndim_supp=}"
            )
        if self.ndim_supp == 0:
            jac = pt.reshape(pt.grad(pt.sum(self.backward(value, *inputs)), [value]), value.shape)
            return pt.log(pt.abs(jac))
        else:
            phi_inv = self.backward(value, *inputs)
            return pt.log(pt.abs(pt.nlinalg.det(pt.atleast_2d(jacobian(phi_inv, [value])[0]))))

    def __str__(self):
        return f"{self.__class__.__name__}"


class LocTransform(Transform):
    name = "loc"

    def __init__(self, transform_args_fn):
        self.transform_args_fn = transform_args_fn

    def forward(self, value, *inputs):
        loc = self.transform_args_fn(*inputs)
        return value + loc

    def backward(self, value, *inputs):
        loc = self.transform_args_fn(*inputs)
        return value - loc

    def log_jac_det(self, value, *inputs):
        return pt.zeros_like(value)


class ScaleTransform(Transform):
    name = "scale"

    def __init__(self, transform_args_fn):
        self.transform_args_fn = transform_args_fn

    def forward(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return value * scale

    def backward(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return value / scale

    def log_jac_det(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return -pt.log(pt.abs(pt.broadcast_to(scale, value.shape)))


class IntervalTransform(Transform):
    name = "interval"

    def __init__(self, args_fn: Callable[..., Tuple[Optional[Variable], Optional[Variable]]]):
        """

        Parameters
        ----------
        args_fn
            Function that expects inputs of RandomVariable and returns the lower
            and upper bounds for the interval transformation. If one of these is
            None, the RV is considered to be unbounded on the respective edge.
        """
        self.args_fn = args_fn

    def forward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            return pt.log(value - a) - pt.log(b - value)
        elif a is not None:
            return pt.log(value - a)
        elif b is not None:
            return pt.log(b - value)
        else:
            raise ValueError("Both edges of IntervalTransform cannot be None")

    def backward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            sigmoid_x = pt.sigmoid(value)
            return sigmoid_x * b + (1 - sigmoid_x) * a
        elif a is not None:
            return pt.exp(value) + a
        elif b is not None:
            return b - pt.exp(value)
        else:
            raise ValueError("Both edges of IntervalTransform cannot be None")

    def log_jac_det(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            s = pt.softplus(-value)
            return pt.log(b - a) - 2 * s - value
        elif a is None and b is None:
            raise ValueError("Both edges of IntervalTransform cannot be None")
        else:
            return value


class Interval(IntervalTransform):
    """Wrapper around  :class:`IntervalTransform` for use in the
    ``transform`` argument of a random variable.

    Parameters
    ----------
    lower : int or float, optional
        Lower bound of the interval transform. Must be a constant finite value.
        By default (``lower=None``), the interval is not bounded below.
    upper : int or float, optional
        Upper bound of the interval transform. Must be a constant finite value.
        By default (``upper=None``), the interval is not bounded above.
    bounds_fn : callable, optional
        Alternative to lower and upper. Must return a tuple of lower and upper bounds
        as a symbolic function of the respective distribution inputs. If one of lower or
        upper is ``None``, the interval is unbounded on that edge.

        .. warning:: Expressions returned by `bounds_fn` should depend only on the
            distribution inputs or other constants. Expressions that depend on nonlocal
            variables, such as other distributions defined in the model context will
            likely break sampling.


    Examples
    --------
    Create an interval transform between -1 and +1

    .. code-block:: python

        with pm.Model():
            interval = pm.distributions.transforms.Interval(lower=-1, upper=1)
            x = pm.Normal("x", transform=interval)

    Create a lower-bounded interval transform at 0, using a callable

    .. code-block:: python

        def get_bounds(rng, size, dtype, mu, sigma):
            return 0, None

        with pm.Model():
            interval = pm.distributions.transforms.Interval(bounds_fn=get_bounds)
            x = pm.Normal("x", transform=interval)

    Create a lower-bounded interval transform that depends on a distribution parameter

    .. code-block:: python

        def get_bounds(rng, size, dtype, mu, sigma):
            return mu - 1, None

        interval = pm.distributions.transforms.Interval(bounds_fn=get_bounds)

        with pm.Model():
            mu = pm.Normal("mu")
            x = pm.Normal("x", mu=mu, sigma=2, transform=interval)
    """

    def __init__(self, lower=None, upper=None, *, bounds_fn=None):
        if bounds_fn is None:
            try:
                bounds = tuple(
                    None if bound is None else pt.constant(bound, ndim=0).data
                    for bound in (lower, upper)
                )
            except (ValueError, TypeError):
                raise ValueError(
                    "Interval bounds must be constant values. If you need expressions that "
                    "depend on symbolic variables use `args_fn`"
                )

            lower, upper = (
                None if (bound is None or np.isinf(bound)) else bound for bound in bounds
            )

            if lower is None and upper is None:
                raise ValueError("Lower and upper interval bounds cannot both be None")

            def bounds_fn(*rv_inputs):
                return lower, upper

        super().__init__(args_fn=bounds_fn)


class LogOddsTransform(Transform):
    name = "logodds"

    def backward(self, value, *inputs):
        return pt.expit(value)

    def forward(self, value, *inputs):
        return pt.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = pt.sigmoid(value)
        return pt.log(sigmoid_value) + pt.log1p(-sigmoid_value)


logodds = LogOddsTransform()
logodds.__doc__ = """
Instantiation of :class:`pymc.logprob.transforms.LogOddsTransform`
for use in the ``transform`` argument of a random variable."""


class SimplexTransform(Transform):
    name = "simplex"

    def forward(self, value, *inputs):
        value = pt.as_tensor(value)
        log_value = pt.log(value)
        N = value.shape[-1].astype(value.dtype)
        shift = pt.sum(log_value, -1, keepdims=True) / N
        return log_value[..., :-1] - shift

    def backward(self, value, *inputs):
        value = pt.concatenate([value, -pt.sum(value, -1, keepdims=True)], axis=-1)
        exp_value_max = pt.exp(value - pt.max(value, -1, keepdims=True))
        return exp_value_max / pt.sum(exp_value_max, -1, keepdims=True)

    def log_jac_det(self, value, *inputs):
        value = pt.as_tensor(value)
        N = value.shape[-1] + 1
        N = N.astype(value.dtype)
        sum_value = pt.sum(value, -1, keepdims=True)
        value_sum_expanded = value + sum_value
        value_sum_expanded = pt.concatenate([value_sum_expanded, pt.zeros(sum_value.shape)], -1)
        logsumexp_value_expanded = pt.logsumexp(value_sum_expanded, -1, keepdims=True)
        res = pt.log(N) + (N * sum_value) - (N * logsumexp_value_expanded)
        return pt.sum(res, -1)


simplex = SimplexTransform()
simplex.__doc__ = """
Instantiation of :class:`pymc.logprob.transforms.SimplexTransform`
for use in the ``transform`` argument of a random variable."""


class LogTransform(Transform):
    name = "log"

    def forward(self, value, *inputs):
        return pt.log(value)

    def backward(self, value, *inputs):
        return pt.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


log = LogTransform()
log.__doc__ = """
Instantiation of :class:`pymc.logprob.transforms.LogTransform`
for use in the ``transform`` argument of a random variable."""


class ExpTransform(Transform):
    name = "exp"

    def forward(self, value, *inputs):
        return pt.exp(value)

    def backward(self, value, *inputs):
        return pt.log(value)

    def log_jac_det(self, value, *inputs):
        return -pt.log(value)


class AbsTransform(Transform):
    name = "abs"

    def forward(self, value, *inputs):
        return pt.abs(value)

    def backward(self, value, *inputs):
        value = pt.switch(value >= 0, value, np.nan)
        return -value, value

    def log_jac_det(self, value, *inputs):
        return pt.switch(value >= 0, 0, np.nan)


class PowerTransform(Transform):
    name = "power"

    def __init__(self, power=None):
        if not isinstance(power, (int, float)):
            raise TypeError(f"Power must be integer or float, got {type(power)}")
        if power == 0:
            raise ValueError("Power cannot be 0")
        self.power = power
        super().__init__()

    def forward(self, value, *inputs):
        return pt.power(value, self.power)

    def backward(self, value, *inputs):
        inv_power = 1 / self.power

        # Powers that don't admit negative values
        if (np.abs(self.power) < 1) or (self.power % 2 == 0):
            backward_value = pt.switch(value >= 0, pt.power(value, inv_power), np.nan)
        # Powers that admit negative values require special logic, because (-1)**(1/3) returns `nan` in PyTensor
        else:
            backward_value = pt.power(pt.abs(value), inv_power) * pt.switch(value >= 0, 1, -1)

        # In this case the transform is not 1-to-1
        if self.power % 2 == 0:
            return -backward_value, backward_value
        else:
            return backward_value

    def log_jac_det(self, value, *inputs):
        inv_power = 1 / self.power

        # Note: This fails for value==0
        res = np.log(np.abs(inv_power)) + (inv_power - 1) * pt.log(pt.abs(value))

        # Powers that don't admit negative values
        if (np.abs(self.power) < 1) or (self.power % 2 == 0):
            res = pt.switch(value >= 0, res, np.nan)

        return res


class ArcsinhTransform(Transform):
    name = "arcsinh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arcsinh(value)

    def backward(self, value, *inputs):
        return pt.sinh(value)


class ArccoshTransform(Transform):
    name = "arccosh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arccosh(value)

    def backward(self, value, *inputs):
        return pt.cosh(value)


class ArctanhTransform(Transform):
    name = "arctanh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arctanh(value)

    def backward(self, value, *inputs):
        return pt.tanh(value)


class SinhTransform(Transform):
    name = "sinh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.sinh(value)

    def backward(self, value, *inputs):
        return pt.arcsinh(value)


class CoshTransform(Transform):
    name = "cosh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.cosh(value)

    def backward(self, value, *inputs):
        back_value = pt.arccosh(value)
        return (-back_value, back_value)

    def log_jac_det(self, value, *inputs):
        return pt.switch(
            value < 1,
            np.nan,
            -pt.log(pt.sqrt(value**2 - 1)),
        )


class TanhTransform(Transform):
    name = "tanh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.tanh(value)

    def backward(self, value, *inputs):
        return pt.arctanh(value)


class ErfTransform(Transform):
    name = "erf"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erf(value)

    def backward(self, value, *inputs):
        return pt.erfinv(value)


class ErfcTransform(Transform):
    name = "erfc"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erfc(value)

    def backward(self, value, *inputs):
        return pt.erfcinv(value)


class ErfcxTransform(Transform):
    name = "erfcx"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erfcx(value)

    def backward(self, value, *inputs):
        # computes the inverse of erfcx, this was adapted from
        # https://tinyurl.com/4mxfd3cz
        x = pt.switch(value <= 1, 1.0 / (value * pt.sqrt(np.pi)), -pt.sqrt(pt.log(value)))

        def calc_delta_x(value, prior_result):
            return prior_result - (pt.erfcx(prior_result) - value) / (
                2 * prior_result * pt.erfcx(prior_result) - 2 / pt.sqrt(np.pi)
            )

        result, updates = scan(
            fn=calc_delta_x,
            outputs_info=pt.ones_like(x),
            non_sequences=value,
            n_steps=10,
        )
        return result[-1]


class ChainTransform(Transform):
    name = "chain"

    def __init__(self, transform_list):
        self.transform_list = transform_list

    def forward(self, value, *inputs):
        for transform in self.transform_list:
            value = transform.forward(value, *inputs)
        return value

    def backward(self, value, *inputs):
        for transform in reversed(self.transform_list):
            value = transform.backward(value, *inputs)
        return value

    def log_jac_det(self, value, *inputs):
        value = pt.as_tensor_variable(value)
        det_list = []
        ndim0 = value.ndim
        for transform in reversed(self.transform_list):
            det_ = transform.log_jac_det(value, *inputs)
            det_list.append(det_)
            ndim0 = min(ndim0, det_.ndim)
            value = transform.backward(value, *inputs)
        # match the shape of the smallest jacobian_det
        det = 0.0
        for det_ in det_list:
            if det_.ndim > ndim0:
                ndim_diff = det_.ndim - ndim0
                det += det_.sum(axis=tuple(range(-ndim_diff, 0)))
            else:
                det += det_
        return det


# For backwards compat
Chain = ChainTransform


class CircularTransform(Transform):
    name = "circular"

    def backward(self, value, *inputs):
        return pt.arctan2(pt.sin(value), pt.cos(value))

    def forward(self, value, *inputs):
        return pt.as_tensor_variable(value)

    def log_jac_det(self, value, *inputs):
        return pt.zeros(value.shape)


circular = CircularTransform()
circular.__doc__ = """
Instantiation of :class:`pymc.logprob.transforms.CircularTransform`
for use in the ``transform`` argument of a random variable."""


class OrderedTransform(Transform):
    name = "ordered"

    def __init__(self, ndim_supp=None):
        if ndim_supp is not None:
            warnings.warn("ndim_supp argument is deprecated and has no effect", FutureWarning)

    def backward(self, value, *inputs):
        x = pt.zeros(value.shape)
        x = pt.set_subtensor(x[..., 0], value[..., 0])
        x = pt.set_subtensor(x[..., 1:], pt.exp(value[..., 1:]))
        return pt.cumsum(x, axis=-1)

    def forward(self, value, *inputs):
        y = pt.zeros(value.shape)
        y = pt.set_subtensor(y[..., 0], value[..., 0])
        y = pt.set_subtensor(y[..., 1:], pt.log(value[..., 1:] - value[..., :-1]))
        return y

    def log_jac_det(self, value, *inputs):
        return pt.sum(value[..., 1:], axis=-1)


ordered = OrderedTransform()
ordered.__doc__ = """
Instantiation of :class:`pymc.distributions.transforms.Ordered`
for use in the ``transform`` argument of a random variable."""


class LogExpM1Transform(Transform):
    name = "log_exp_m1"

    def backward(self, value, *inputs):
        return pt.softplus(value)

    def forward(self, value, *inputs):
        """Inverse operation of softplus.

        y = Log(Exp(x) - 1)
          = Log(1 - Exp(-x)) + x
        """
        return pt.log(1.0 - pt.exp(-value)) + value

    def log_jac_det(self, value, *inputs):
        return -pt.softplus(-value)


log_exp_m1 = LogExpM1Transform()
log_exp_m1.__doc__ = """
Instantiation of :class:`pymc.distributions.transforms.LogExpM1`
for use in the ``transform`` argument of a random variable."""


class SumTo1Transform(Transform):
    """
    Transforms K - 1 dimensional simplex space (k values in [0,1] and that sum to 1) to a K - 1 vector of values in [0,1]
    This Transformation operates on the last dimension of the input tensor.
    """

    name = "sumto1"

    def __init__(self, ndim_supp=None):
        if ndim_supp is not None:
            warnings.warn("ndim_supp argument is deprecated and has no effect", FutureWarning)

    def backward(self, value, *inputs):
        remaining = 1 - pt.sum(value[..., :], axis=-1, keepdims=True)
        return pt.concatenate([value[..., :], remaining], axis=-1)

    def forward(self, value, *inputs):
        return value[..., :-1]

    def log_jac_det(self, value, *inputs):
        y = pt.zeros(value.shape)
        return pt.sum(y, axis=-1)


sum_to_1 = SumTo1Transform()
sum_to_1.__doc__ = """
Instantiation of :class:`pymc.distributions.transforms.SumTo1`
for use in the ``transform`` argument of a random variable."""


class ZeroSumTransform(Transform):
    """
    Constrains any random samples to sum to zero along the user-provided ``zerosum_axes``.

    Parameters
    ----------
    zerosum_axes : list of ints
        Must be a list of integers (positive or negative).
    """

    name = "zerosum"

    __props__ = ("zerosum_axes",)

    def __init__(self, zerosum_axes):
        self.zerosum_axes = tuple(int(axis) for axis in zerosum_axes)

    @staticmethod
    def extend_axis_rev(array, axis):
        normalized_axis = normalize_axis_tuple(axis, array.ndim)[0]

        n = floatX(array.shape[normalized_axis])
        last = pt.take(array, [-1], axis=normalized_axis)

        sum_vals = -last * pt.sqrt(n)
        norm = sum_vals / (pt.sqrt(n) + n)
        slice_before = (slice(None, None),) * normalized_axis

        return array[slice_before + (slice(None, -1),)] + norm

    @staticmethod
    def extend_axis(array, axis):
        n = floatX(array.shape[axis] + 1)
        sum_vals = array.sum(axis, keepdims=True)
        norm = sum_vals / (pt.sqrt(n) + n)
        fill_val = norm - sum_vals / pt.sqrt(n)

        out = pt.concatenate([array, fill_val], axis=axis)
        return out - norm

    def forward(self, value, *rv_inputs):
        for axis in self.zerosum_axes:
            value = self.extend_axis_rev(value, axis=axis)
        return value

    def backward(self, value, *rv_inputs):
        for axis in self.zerosum_axes:
            value = self.extend_axis(value, axis=axis)
        return value

    def log_jac_det(self, value, *rv_inputs):
        return pt.constant(0.0)


class CholeskyCovPackedTransform(Transform):
    """
    Transforms the diagonal elements of the LKJCholeskyCov distribution to be on the
    log scale
    """

    name = "cholesky-cov-packed"

    def __init__(self, n):
        """

        Parameters
        ----------
        n: int
            Number of diagonal entries in the LKJCholeskyCov distribution
        """
        self.diag_idxs = pt.arange(1, n + 1).cumsum() - 1

    def backward(self, value, *inputs):
        return pt.set_subtensor(value[..., self.diag_idxs], pt.exp(value[..., self.diag_idxs]))

    def forward(self, value, *inputs):
        return pt.set_subtensor(value[..., self.diag_idxs], pt.log(value[..., self.diag_idxs]))

    def log_jac_det(self, value, *inputs):
        return pt.sum(value[..., self.diag_idxs], axis=-1)


CholeskyCovPacked = CholeskyCovPackedTransform
