#   Copyright 2020 The PyMC Developers

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from functools import singledispatch

import aesara.tensor as at
import numpy as np

from aeppl.transforms import (
    CircularTransform,
    IntervalTransform,
    LogOddsTransform,
    LogTransform,
    RVTransform,
    Simplex,
)
from aesara.graph import Op
from aesara.tensor import TensorVariable

__all__ = [
    "RVTransform",
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
]


@singledispatch
def _default_transform(op: Op, rv: TensorVariable):
    """Return default transform for a given Distribution `Op`"""
    return None


class LogExpM1(RVTransform):
    name = "log_exp_m1"

    def backward(self, value, *inputs):
        return at.softplus(value)

    def forward(self, value, *inputs):
        """Inverse operation of softplus.

        y = Log(Exp(x) - 1)
          = Log(1 - Exp(-x)) + x
        """
        return at.log(1.0 - at.exp(-value)) + value

    def log_jac_det(self, value, *inputs):
        return -at.softplus(-value)


class Ordered(RVTransform):
    name = "ordered"

    def backward(self, value, *inputs):
        x = at.zeros(value.shape)
        x = at.inc_subtensor(x[..., 0], value[..., 0])
        x = at.inc_subtensor(x[..., 1:], at.exp(value[..., 1:]))
        return at.cumsum(x, axis=-1)

    def forward(self, value, *inputs):
        y = at.zeros(value.shape)
        y = at.inc_subtensor(y[..., 0], value[..., 0])
        y = at.inc_subtensor(y[..., 1:], at.log(value[..., 1:] - value[..., :-1]))
        return y

    def log_jac_det(self, value, *inputs):
        return at.sum(value[..., 1:], axis=-1)


class SumTo1(RVTransform):
    """
    Transforms K - 1 dimensional simplex space (k values in [0,1] and that sum to 1) to a K - 1 vector of values in [0,1]
    This Transformation operates on the last dimension of the input tensor.
    """

    name = "sumto1"

    def backward(self, value, *inputs):
        remaining = 1 - at.sum(value[..., :], axis=-1, keepdims=True)
        return at.concatenate([value[..., :], remaining], axis=-1)

    def forward(self, value, *inputs):
        return value[..., :-1]

    def log_jac_det(self, value, *inputs):
        y = at.zeros(value.shape)
        return at.sum(y, axis=-1)


class CholeskyCovPacked(RVTransform):
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
        self.diag_idxs = at.arange(1, n + 1).cumsum() - 1

    def backward(self, value, *inputs):
        return at.set_subtensor(value[..., self.diag_idxs], at.exp(value[..., self.diag_idxs]))

    def forward(self, value, *inputs):
        return at.set_subtensor(value[..., self.diag_idxs], at.log(value[..., self.diag_idxs]))

    def log_jac_det(self, value, *inputs):
        return at.sum(value[..., self.diag_idxs], axis=-1)


class Chain(RVTransform):

    __slots__ = ("param_extract_fn", "transform_list", "name")

    def __init__(self, transform_list):
        self.transform_list = transform_list
        self.name = "+".join([transf.name for transf in self.transform_list])

    def forward(self, value, *inputs):
        y = value
        for transf in self.transform_list:
            # TODO:Needs proper discussion as to what should be
            # passed as inputs here
            y = transf.forward(y, *inputs)
        return y

    def backward(self, value, *inputs):
        x = value
        for transf in reversed(self.transform_list):
            x = transf.backward(x, *inputs)
        return x

    def log_jac_det(self, value, *inputs):
        y = at.as_tensor_variable(value)
        det_list = []
        ndim0 = y.ndim
        for transf in reversed(self.transform_list):
            det_ = transf.log_jac_det(y, *inputs)
            det_list.append(det_)
            y = transf.backward(y, *inputs)
            ndim0 = min(ndim0, det_.ndim)
        # match the shape of the smallest log_jac_det
        det = 0.0
        for det_ in det_list:
            if det_.ndim > ndim0:
                det += det_.sum(axis=-1)
            else:
                det += det_
        return det


simplex = Simplex()
simplex.__doc__ = """
Instantiation of :class:`aeppl.transforms.Simplex`
for use in the ``transform`` argument of a random variable."""

logodds = LogOddsTransform()
logodds.__doc__ = """
Instantiation of :class:`aeppl.transforms.LogOddsTransform`
for use in the ``transform`` argument of a random variable."""


class Interval(IntervalTransform):
    """Wrapper around  :class:`aeppl.transforms.IntervalTransform` for use in the
    ``transform`` argument of a random variable.

    Parameters
    ----------
    lower : int or float, optional
        Lower bound of the interval transform. Must be a constant finite value.
        By default (``lower=None``), the interval is not bounded below.
    upper : int or float, optinoal
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
    .. code-block:: python

        # Create an interval transform between -1 and +1
        with pm.Model():
            interval = pm.distributions.transforms.Interval(lower=-1, upper=1)
            x = pm.Normal("x", transform=interval)

    .. code-block:: python

        # Create an interval transform between -1 and +1 using a callable
        def get_bounds(rng, size, dtype, loc, scale):
            return 0, None

        with pm.Model():
            interval = pm.distributions.transforms.Interval(bouns_fn=get_bounds)
            x = pm.Normal("x", transform=interval)

    .. code-block:: python

        # Create a lower bounded interval transform based on a distribution parameter
        def get_bounds(rng, size, dtype, loc, scale):
            return loc, None

        interval = pm.distributions.transforms.Interval(bounds_fn=get_bounds)

        with pm.Model():
            loc = pm.Normal("loc")
            x = pm.Normal("x", mu=loc, sigma=2, transform=interval)
    """

    def __init__(self, lower=None, upper=None, *, bounds_fn=None):
        if bounds_fn is None:
            try:
                bounds = tuple(
                    None if bound is None else at.constant(bound, ndim=0).data
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


log_exp_m1 = LogExpM1()
log_exp_m1.__doc__ = """
Instantiation of :class:`pymc.distributions.transforms.LogExpM1`
for use in the ``transform`` argument of a random variable."""

ordered = Ordered()
ordered.__doc__ = """
Instantiation of :class:`pymc.distributions.transforms.Ordered`
for use in the ``transform`` argument of a random variable."""

log = LogTransform()
log.__doc__ = """
Instantiation of :class:`aeppl.transforms.LogTransform`
for use in the ``transform`` argument of a random variable."""

sum_to_1 = SumTo1()
sum_to_1.__doc__ = """
Instantiation of :class:`pymc.distributions.transforms.SumTo1`
for use in the ``transform`` argument of a random variable."""

circular = CircularTransform()
circular.__doc__ = """
Instantiation of :class:`aeppl.transforms.CircularTransform`
for use in the ``transform`` argument of a random variable."""
