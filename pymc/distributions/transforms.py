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

from functools import singledispatch

import numpy as np
import pytensor.tensor as pt

from numpy.lib.array_utils import normalize_axis_tuple
from pytensor.graph import Op
from pytensor.tensor import TensorLike, TensorVariable

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
    "Chain",
    "CholeskyCorrTransform",
    "CholeskyCovPacked",
    "CholeskyCovTransform",
    "Interval",
    "Transform",
    "ZeroSumTransform",
    "circular",
    "log",
    "log_exp_m1",
    "logodds",
    "ordered",
    "simplex",
    "sum_to_1",
]


@singledispatch
def _default_transform(op: Op, rv: TensorVariable):
    """Return default transform for a given Distribution `Op`."""
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
    """
    Transforms a vector of values into a vector of ordered values.

    Parameters
    ----------
    positive: If True, all values are positive. This has better geometry than just chaining with a log transform.
    ascending: If True, the values are in ascending order (default). If False, the values are in descending order.
    """

    name = "ordered"

    def __init__(self, positive=False, ascending=True):
        self.positive = positive
        self.ascending = ascending

    def backward(self, value, *inputs):
        if self.positive:  # Transform both initial value and deltas to be positive
            x = pt.exp(value)
        else:  # Transform only deltas to be positive
            x = pt.empty(value.shape)
            x = pt.set_subtensor(x[..., 0], value[..., 0])
            x = pt.set_subtensor(x[..., 1:], pt.exp(value[..., 1:]))
        x = pt.cumsum(x, axis=-1)  # Add deltas cumulatively to initial value
        if not self.ascending:
            x = x[..., ::-1]
        return x

    def forward(self, value, *inputs):
        if not self.ascending:
            value = value[..., ::-1]
        y = pt.empty(value.shape)
        y = pt.set_subtensor(y[..., 0], pt.log(value[..., 0]) if self.positive else value[..., 0])
        y = pt.set_subtensor(y[..., 1:], pt.log(value[..., 1:] - value[..., :-1]))
        return y

    def log_jac_det(self, value, *inputs):
        if self.positive:
            return pt.sum(value, axis=-1)
        else:
            return pt.sum(value[..., 1:], axis=-1)


class SumTo1(Transform):
    """
    Transforms K - 1 dimensional simplex space (K values in [0, 1] that sum to 1) to a K - 1 vector of values in [0, 1].

    This transformation operates on the last dimension of the input tensor.
    """

    name = "sumto1"

    def backward(self, value, *inputs):
        remaining = 1 - pt.sum(value[..., :], axis=-1, keepdims=True)
        return pt.concatenate([value[..., :], remaining], axis=-1)

    def forward(self, value, *inputs):
        return value[..., :-1]

    def log_jac_det(self, value, *inputs):
        y = pt.zeros(value.shape)
        return pt.sum(y, axis=-1)


class CholeskyCorrTransform(Transform):
    r"""Map an unconstrained real vector to the Cholesky factor of a correlation matrix.

    Constrained space: ``(n, n)`` lower-triangular Cholesky factor ``L`` of a
    correlation matrix, with unit-norm rows (so ``L @ L.T`` has ones on the diagonal).
    Unconstrained space: ``(n*(n-1)/2,)`` flat real vector packed in row-major
    strictly-lower-triangular order.

    The transform composes three steps:

    1. Scatter the flat vector into the strictly lower-triangular positions of an
       ``(n, n)`` matrix and set the diagonal to 1.
    2. Normalize each row to unit L2 norm, producing the Cholesky factor ``L``.

    The Jacobian of the composite map uses the diagonal of the normalized factor:

    .. math::

        \log |J| = \sum_{k=0}^{n-1} (k+2)\,\log L_{kk}
                 = -\tfrac12 \sum_{k=0}^{n-1} (k+2)\,\log\!\bigl(1 + \|r_k\|^2\bigr)

    where :math:`r_k` are the off-diagonal elements placed in row *k*.

    References
    ----------
    .. [1] Lewandowski, D., Kurowicka, D., & Joe, H. (2009).
       Generating random correlation matrices based on vines and extended onion method.
       Journal of Multivariate Analysis, 100(9), 1989–2001.
    .. [2] Stan Development Team. Stan Functions Reference. Section on LKJ / Cholesky correlation.
    """

    name = "cholesky_corr"

    def __init__(self, n, upper: bool = False):
        if upper:
            raise NotImplementedError("upper=True is not supported")
        self.n = n
        self.upper = upper
        self.tril_idxs = pt.tril_indices(n, -1)

        super().__init__()

    def forward(self, chol_corr_matrix: TensorLike, *inputs):
        chol_corr_matrix = pt.as_tensor_variable(chol_corr_matrix)

        # Divide each row by its diagonal element to undo the normalization.
        diag = pt.diagonal(chol_corr_matrix, axis1=-2, axis2=-1)[..., None]
        unconstrained = chol_corr_matrix / diag

        # The strictly lower-triangular elements (row-major) are the free parameters.
        return unconstrained[..., self.tril_idxs[0], self.tril_idxs[1]]

    def backward(self, unconstrained_vector: TensorLike, *inputs):
        unconstrained_vector = pt.as_tensor_variable(unconstrained_vector)
        n = self.n

        # Scatter into the strictly-lower-triangular positions of an (n, n) matrix.
        L = pt.zeros((*unconstrained_vector.shape[:-1], n, n), dtype=unconstrained_vector.dtype)
        L = L[..., self.tril_idxs[0], self.tril_idxs[1]].set(unconstrained_vector)

        # Set diagonal to 1 (before normalization).
        diag_idx = pt.arange(n)
        L = L[..., diag_idx, diag_idx].set(1)

        # Normalize each row to unit L2 norm.
        L /= pt.linalg.norm(L, axis=-1, ord=2)[..., None]
        return L

    def log_jac_det(self, unconstrained_vector: TensorLike, *inputs) -> TensorVariable:
        unconstrained_vector = pt.as_tensor_variable(unconstrained_vector)
        n = self.n
        dtype = unconstrained_vector.dtype

        # Compute per-row sum of squares of the off-diagonal elements directly,
        # without constructing the full normalized matrix.
        sq = unconstrained_vector**2
        row_sums_sq = pt.zeros((*unconstrained_vector.shape[:-1], n), dtype=dtype)
        row_sums_sq = pt.inc_subtensor(row_sums_sq[..., self.tril_idxs[0]], sq)

        # After setting diagonal to 1 and normalizing, diag_k = 1/sqrt(1 + row_sums_sq_k).
        log_diag = pt.cast(-0.5, dtype) * pt.log1p(row_sums_sq)
        coeffs = pt.arange(2, 2 + n, dtype=dtype)
        return pt.sum(coeffs * log_diag, axis=-1)


class CholeskyCovPacked(Transform):
    """Transforms the diagonal elements of the LKJCholeskyCov distribution to be on the log scale."""

    name = "cholesky-cov-packed"

    def __init__(self, n):
        """Create a CholeskyCovPack object.

        Parameters
        ----------
        n : int
            Number of diagonal entries in the LKJCholeskyCov distribution.
        """
        self.diag_idxs = pt.arange(1, n + 1).cumsum() - 1

    def backward(self, value, *inputs):
        return pt.set_subtensor(value[..., self.diag_idxs], pt.exp(value[..., self.diag_idxs]))

    def forward(self, value, *inputs):
        return pt.set_subtensor(value[..., self.diag_idxs], pt.log(value[..., self.diag_idxs]))

    def log_jac_det(self, value, *inputs):
        return pt.sum(value[..., self.diag_idxs], axis=-1)


class CholeskyCovTransform(Transform):
    r"""Map an unconstrained real vector to a symmetric positive-definite (SPD) matrix.

    Constrained space: ``(n, n)`` SPD matrix ``X``.
    Unconstrained space: ``(n*(n+1)/2,)`` flat real vector packed in row-major
    lower-triangular order.

    The transform composes two steps:

    1. Reshape the flat vector into a lower-triangular matrix ``L``, exponentiating
       the diagonal entries (so ``L_kk > 0``).
    2. Form ``X = L @ L.T``, the SPD matrix.

    The lower-triangular ``L`` produced inside ``backward`` is tagged with
    ``lower_triangular = True`` so PyTensor's ``cholesky_ldotlt`` rewrite eliminates
    any subsequent ``cholesky(L @ L.T)`` calls (e.g. inside a Cholesky-based
    ``Wishart.logp``), avoiding a redundant decomposition at runtime.

    The Jacobian of the composite map (free reals :math:`\to L \to L L^\top`) is

    .. math::

        |J| = \prod_k L_{kk} \cdot 2^n \cdot \prod_k L_{kk}^{n-k+1}
            = 2^n \cdot \prod_k L_{kk}^{n-k+2}.

    With :math:`y_{kk} = \log L_{kk}` the diagonal entries of the unconstrained
    vector, the log-Jacobian is

    .. math::

        \log |J| = n \log 2 + \sum_{k=1}^{n} (n - k + 2)\, y_{kk}.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pytensor.tensor as pt
        from pymc.distributions.transforms import CholeskyCovTransform

        tr = CholeskyCovTransform(n=3)
        unconstrained = pt.as_tensor(np.array([0.0, 0.5, 0.0, 0.2, -0.1, 0.0]))
        Sigma = tr.backward(unconstrained)
        # Sigma is a (3, 3) SPD matrix.
    """

    name = "cholesky-cov"

    def __init__(self, n):
        """Create a CholeskyCovTransform.

        Parameters
        ----------
        n : int or TensorLike
            Side length of the SPD matrix.
        """
        self.n = n
        self.diag_idxs = pt.arange(1, n + 1).cumsum() - 1
        self.tril_idxs = pt.tril_indices(n)

    def backward(self, value, *inputs):
        value = pt.as_tensor_variable(value)
        # Exponentiate the diagonal entries so L has positive diagonal.
        value_pos = value[..., self.diag_idxs].set(pt.exp(value[..., self.diag_idxs]))
        # Scatter into a (..., n, n) lower-triangular matrix L.
        n = self.n
        L_shape = (*value.shape[:-1], n, n)
        L = pt.zeros(L_shape, dtype=value.dtype)
        L = L[..., self.tril_idxs[0], self.tril_idxs[1]].set(value_pos)
        # Tag L as lower-triangular so cholesky(L @ L.mT) → L is rewritten away.
        L.tag.lower_triangular = True
        return L @ L.mT

    def forward(self, value, *inputs):
        value = pt.as_tensor_variable(value)
        L = pt.linalg.cholesky(value)
        flat = L[..., self.tril_idxs[0], self.tril_idxs[1]]
        # log the diagonal entries to make them unconstrained.
        return flat[..., self.diag_idxs].set(pt.log(flat[..., self.diag_idxs]))

    def log_jac_det(self, value, *inputs):
        value = pt.as_tensor_variable(value)
        n = self.n
        log_diag = value[..., self.diag_idxs]
        coeffs = pt.arange(n + 1, 1, -1).astype(value.dtype)
        return n * pt.log(2.0) + pt.sum(coeffs * log_diag, axis=-1)


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
    """Wrapper around  :class:`pymc.logprob.transforms.IntervalTransform` for use in the ``transform`` argument of a random variable.

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

        def get_bounds(rng, size, mu, sigma):
            return 0, None


        with pm.Model():
            interval = pm.distributions.transforms.Interval(bounds_fn=get_bounds)
            x = pm.Normal("x", transform=interval)

    Create a lower-bounded interval transform that depends on a distribution parameter

    .. code-block:: python

        def get_bounds(rng, size, mu, sigma):
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
        return value.sum(self.zerosum_axes).zeros_like()


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
