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
import warnings

from functools import singledispatch

import numpy as np
import pytensor.tensor as pt
import pytensor


# ignore mypy error because it somehow considers that
# "numpy.core.numeric has no attribute normalize_axis_tuple"
from numpy.core.numeric import normalize_axis_tuple  # type: ignore
from pytensor.graph import Op
from pytensor.tensor import TensorVariable

from pymc.logprob.transforms import (
    ChainedTransform,
    CircularTransform,
    IntervalTransform,
    LogOddsTransform,
    LogTransform,
    SimplexTransform,
    Transform,
)

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
    "CholeskyCorr",
    "CholeskyCovPacked",
    "Chain",
    "ZeroSumTransform",
]


def __getattr__(name):
    if name in ("univariate_ordered", "multivariate_ordered"):
        warnings.warn(f"{name} has been deprecated, use ordered instead.", FutureWarning)
        return ordered

    if name in ("univariate_sum_to_1", "multivariate_sum_to_1"):
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


class LogExpM1(Transform):
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


class Ordered(Transform):
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


class SumTo1(Transform):
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


class CholeskyCorr(Transform):
    """
    Transforms unconstrained real numbers to the off-diagonal elements of
    a Cholesky decomposition of a correlation matrix.

    This ensures that the resulting correlation matrix is positive definite.

    #### Mathematical Details

    [Include detailed mathematical explanations similar to the original TFP bijector.]

    #### Examples

    ```python
    transform = CholeskyCorr(n=3)
    x = pt.as_tensor_variable([0.0, 0.0, 0.0])
    y = transform.forward(x).eval()
    # y will be the off-diagonal elements of the Cholesky factor

    x_reconstructed = transform.backward(y).eval()
    # x_reconstructed should closely match the original x
    ```

    #### References
    - [Stan Manual. Section 24.2. Cholesky LKJ Correlation Distribution.](https://mc-stan.org/docs/2_18/functions-reference/cholesky-lkj-correlation-distribution.html)
    - Lewandowski, D., Kurowicka, D., & Joe, H. (2009). "Generating random correlation matrices based on vines and extended onion method." *Journal of Multivariate Analysis, 100*(5), 1989-2001.
    """

    name = "cholesky-corr"

    def __init__(self, n, validate_args=False):
        """
        Initialize the CholeskyCorr transform.

        Parameters
        ----------
        n : int
            Size of the correlation matrix.
        validate_args : bool, default False
            Whether to validate input arguments.
        """
        self.n = n
        self.m = int(n * (n - 1) / 2)  # Number of off-diagonal elements
        self.tril_r_idxs, self.tril_c_idxs = self._generate_tril_indices()
        self.triu_r_idxs, self.triu_c_idxs = self._generate_triu_indices()
        super().__init__(validate_args=validate_args)

    def _generate_tril_indices(self):
        row_indices, col_indices = np.tril_indices(self.n, -1)
        return (row_indices, col_indices)

    def _generate_triu_indices(self):
        row_indices, col_indices = np.triu_indices(self.n, 1)
        return (row_indices, col_indices)

    def forward(self, x, *inputs):
        """
        Forward transform: Unconstrained real numbers to Cholesky factors.

        Parameters
        ----------
        x : tensor
            Unconstrained real numbers.

        Returns
        -------
        tensor
            Transformed Cholesky factors.
        """
        # Initialize a zero matrix
        chol = pt.zeros((self.n, self.n), dtype=x.dtype)

        # Assign the unconstrained values to the lower triangular part
        chol = pt.set_subtensor(
            chol[self.tril_r_idxs, self.tril_c_idxs],
            x
        )

        # Normalize each row to have unit L2 norm
        row_norms = pt.sqrt(pt.sum(chol ** 2, axis=1, keepdims=True))
        chol = chol / row_norms

        return chol[self.tril_r_idxs, self.tril_c_idxs]

    def backward(self, y, *inputs):
        """
        Backward transform: Cholesky factors to unconstrained real numbers.

        Parameters
        ----------
        y : tensor
            Cholesky factors.

        Returns
        -------
        tensor
            Unconstrained real numbers.
        """
        # Reconstruct the full Cholesky matrix
        chol = pt.zeros((self.n, self.n), dtype=y.dtype)
        chol = pt.set_subtensor(
            chol[self.triu_r_idxs, self.triu_c_idxs],
            y
        )
        chol = chol + pt.transpose(chol) + pt.eye(self.n, dtype=y.dtype)

        # Perform Cholesky decomposition
        chol = pt.linalg.cholesky(chol)

        # Extract the unconstrained parameters by normalizing
        row_norms = pt.sqrt(pt.sum(chol ** 2, axis=1))
        unconstrained = chol / row_norms[:, None]

        return unconstrained[self.tril_r_idxs, self.tril_c_idxs]

    def log_jac_det(self, y, *inputs):
        """
        Compute the log determinant of the Jacobian.

        The Jacobian determinant for normalization is the product of row norms.
        """
        row_norms = pt.sqrt(pt.sum(y ** 2, axis=1))
        return -pt.sum(pt.log(row_norms), axis=-1)


class CholeskyCovPacked(Transform):
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


Chain = ChainedTransform

simplex = SimplexTransform()
simplex.__doc__ = """
Instantiation of :class:`pymc.logprob.transforms.SimplexTransform`
for use in the ``transform`` argument of a random variable."""

logodds = LogOddsTransform()
logodds.__doc__ = """
Instantiation of :class:`pymc.logprob.transforms.LogOddsTransform`
for use in the ``transform`` argument of a random variable."""


class Interval(IntervalTransform):
    """Wrapper around  :class:`pymc.logprob.transforms.IntervalTransform` for use in the
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
    def extend_axis(array, axis):
        n = (array.shape[axis] + 1).astype("floatX")
        sum_vals = array.sum(axis, keepdims=True)
        norm = sum_vals / (pt.sqrt(n) + n)
        fill_val = norm - sum_vals / pt.sqrt(n)

        out = pt.concatenate([array, fill_val], axis=axis)
        return out - norm

    @staticmethod
    def extend_axis_rev(array, axis):
        normalized_axis = normalize_axis_tuple(axis, array.ndim)[0]

        n = array.shape[normalized_axis].astype("floatX")
        last = pt.take(array, [-1], axis=normalized_axis)

        sum_vals = -last * pt.sqrt(n)
        norm = sum_vals / (pt.sqrt(n) + n)
        slice_before = (slice(None, None),) * normalized_axis

        return array[(*slice_before, slice(None, -1))] + norm

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
Instantiation of :class:`pymc.logprob.transforms.LogTransform`
for use in the ``transform`` argument of a random variable."""

sum_to_1 = SumTo1()
sum_to_1.__doc__ = """
Instantiation of :class:`pymc.distributions.transforms.SumTo1`
for use in the ``transform`` argument of a random variable."""

circular = CircularTransform()
circular.__doc__ = """
Instantiation of :class:`pymc.logprob.transforms.CircularTransform`
for use in the ``transform`` argument of a random variable."""
