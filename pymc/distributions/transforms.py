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
    "Chain",
    "CholeskyCorrTransform",
    "CholeskyCovPacked",
    "CholeskyCovPacked",
    "Interval",
    "Transform",
    "ZeroSumTransform",
    "ZeroSumTransform",
    "circular",
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
    """
    Map an unconstrained real vector the Cholesky factor of a correlation matrix.

    For detailed description of the transform, [1]_ and [2]_.

    This is typically used with :class:`~pymc.distributions.LKJCholeskyCov` to place priors on correlation structures.
    For a related transform that additionally rescales diagonal elements (working on covariance factors), see
    :class:`~pymc.distributions.transforms.CholeskyCovPacked`.

    Adapted from the implementation in TensorFlow Probability [3]_:
    https://github.com/tensorflow/probability/blob/94f592af363e13391858b48f785eb4c250912904/tensorflow_probability/python/bijectors/correlation_cholesky.py#L31

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import pytensor.tensor as pt
        from pymc.distributions.transforms import CholeskyCorr

        unconstrained_vector = pt.as_tensor(np.array([2.0, 2.0, 1.0]))
        n = unconstrained_vector.shape[0]
        tr = CholeskyCorr(n)
        constrained_matrix = tr.forward(unconstrained_vector)
        y.eval()
        array(
            [[1.0, 0.0, 0.0], [0.70710678, 0.70710678, 0.0], [0.66666667, 0.66666667, 0.33333333]]
        )

    References
    ----------
    .. [1] Lewandowski, D., Kurowicka, D., & Joe, H. (2009).
       Generating random correlation matrices based on vines and extended onion method.
       Journal of Multivariate Analysis, 100(9), 1989–2001.
    .. [2] Stan Development Team. Stan Functions Reference. Section on LKJ / Cholesky correlation.
    .. [3] TensorFlow Probability. Correlation Cholesky bijector implementation.
       https://github.com/tensorflow/probability/
    """

    name = "cholesky_corr"

    def __init__(self, n, upper: bool = False):
        """
        Initialize the CholeskyCorr transform.

        Parameters
        ----------
        n : int
            Size of the correlation matrix.
        upper: bool, default False
            If True, transform to an upper triangular matrix. If False, transform to a lower triangular matrix.
        """
        self.n = n
        self.m = (n * (n + 1)) // 2  # Number of triangular elements
        self.upper = upper

        super().__init__()

    def _fill_triangular_spiral(
        self, x_raveled: TensorLike, unit_diag: bool = True
    ) -> TensorVariable:
        """
        Create a triangular matrix from a vector by filling it in a spiral order.

        This code is adapted from the `fill_triangular` function in TensorFlow Probability:
        https://github.com/tensorflow/probability/blob/a26f4cbe5ce1549767e13798d9bf5032dac4257b/tensorflow_probability/python/math/linalg.py#L925

        Parameters
        ----------
        x_raveled: TensorLike
            The input vector to be reshaped into a triangular matrix.
        unit_diag: bool, default False
            If True, the diagonal elements are assumed to be 1 and are not filled from the input vector. The input
            vector is expected to have length m = n * (n - 1) / 2 in this case, containing only the off-diagonal
            elements.

        Returns
        -------
        triangular_matrix: TensorVariable
            The resulting triangular matrix.

        Notes
        -----
        By "spiral order", it is meant that the matrix is filled by jumping between the top and bottom rows, flipping
        the fill order from left-to-right to right-to-left on each jump. For example, to fill a 4x4 matrix with
        `order=True`, the matrix is filled in the following order:

        - Row 0, left to right
        - Row 3, right to left
        - Row 1, left to right
        - Row 2, right to left

        When `upper` if False, everything is reversed:

        - Row 3, right to left
        - Row 0, left to right
        - Row 2, right to left
        - Row 1, left to right

        After filling, entries not part of the triangular matrix are set to zero.

        Examples
        --------

        .. code-block:: python

            import numpy as np
            from pymc.distributions.transforms import CholeskyCorr

            tr = CholeskyCorr(n=4)
            x_unconstrained = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            tr._fill_triangular_spiral(x_unconstrained, upper=False).eval()

            # Out:
            # array([[ 5,  0,  0,  0],
            #        [ 9, 10,  0,  0],
            #        [ 8,  7,  6,  0],
            #        [ 4,  3,  2,  1]])
        """
        x_raveled = pt.as_tensor(x_raveled)
        *batch_shape, _ = x_raveled.shape
        n, m = self.n, self.m
        upper = self.upper

        if unit_diag:
            n = n - 1

        tail = x_raveled[..., n:]

        if upper:
            xc = pt.concatenate([x_raveled, pt.flip(tail, -1)], axis=-1)
        else:
            xc = pt.concatenate([tail, pt.flip(x_raveled, -1)], axis=-1)

        y = pt.reshape(xc, (*batch_shape, n, n))
        return pt.triu(y) if upper else pt.tril(y)

    def _inverse_fill_triangular_spiral(
        self, x: TensorLike, unit_diag: bool = True
    ) -> TensorVariable:
        """
        Inverse operation of `_fill_triangular_spiral`.

        Extracts the elements of a triangular matrix in spiral order and returns them as a vector. For details about
        what is meant by "spiral order", see the docstring of `_fill_triangular_spiral`.

        Parameters
        ----------
        x: TensorVariable
            The input triangular matrix.
        unit_diag: bool
            If True, the diagonal elements are assumed to be 1 and are not included in the output vector.

        Returns
        -------
        x_raveled: TensorVariable
            The resulting vector containing the elements of the triangular matrix in spiral order.
        """
        x = pt.as_tensor(x)
        *batch_shape, _, _ = x.shape
        n, m = self.n, self.m

        if unit_diag:
            m = m - n
            n = n - 1

        upper = self.upper

        if upper:
            initial_elements = x[..., 0, :]
            triangular_portion = x[..., 1:, :]
        else:
            initial_elements = pt.flip(x[..., -1, :], axis=-1)
            triangular_portion = x[..., :-1, :]

        rotated_triangular_portion = pt.flip(triangular_portion, axis=(-1, -2))  # type: ignore[arg-type]
        consolidated_matrix = triangular_portion + rotated_triangular_portion
        end_sequence = pt.reshape(
            consolidated_matrix,
            (*batch_shape, pt.cast(n * (n - 1), "int64")),
        )
        y = pt.concatenate([initial_elements, end_sequence[..., : m - n]], axis=-1)

        return y

    def forward(self, chol_corr_matrix: TensorLike, *inputs):
        """
        Transform the Cholesky factor of a correlation matrix into a real-valued vector.

        Parameters
        ----------
        chol_corr_matrix : TensorVariable
            Cholesky factor of a correlation matrix R = L @ L.T of shape (n,n).
        inputs:
            Additional input values. Not used; included for signature compatibility with other transformations.

        Returns
        -------
        unconstrained_vector: TensorVariable
            Real-valued vector of length m = n * (n - 1) / 2.
        """
        chol_corr_matrix = pt.as_tensor(chol_corr_matrix)
        n = self.n

        # Extract the reciprocal of the row norms from the diagonal.
        diag = pt.diagonal(chol_corr_matrix, axis1=-2, axis2=-1)[..., None]

        # Set the diagonal to 0s.
        diag_idx = pt.arange(n)
        chol_corr_matrix = chol_corr_matrix[..., diag_idx, diag_idx].set(0)

        # Multiply with the norm (or divide by its reciprocal) to recover the
        # unconstrained reals in the (strictly) lower triangular part.
        unconstrained_matrix = chol_corr_matrix / diag

        # Remove the first row and last column before inverting the fill_triangular_spiral
        # transformation.
        return self._inverse_fill_triangular_spiral(
            unconstrained_matrix[..., 1:, :-1], unit_diag=True
        )

    def backward(self, unconstrained_vector: TensorLike, *inputs):
        """
        Transform a real-valued vector of length m = n * (n - 1) / 2 into the Cholesky factor of a correlation matrix.

        Parameters
        ----------
        unconstrained_vector : TensorLike
            Real-valued vector of length m = n * (n - 1) / 2.
        inputs:
            Additional input values. Not used; included for signature compatibility with other transformations.

        Returns
        -------
        unconstrained_vector: TensorVariable
            Unconstrained real numbers.
        """
        unconstrained_vector = pt.as_tensor(unconstrained_vector)
        chol_corr_matrix = self._fill_triangular_spiral(unconstrained_vector, unit_diag=True)

        # Pad zeros on the top row and right column.
        ndim = chol_corr_matrix.ndim
        paddings = [*([(0, 0)] * (ndim - 2)), [1, 0], [0, 1]]
        chol_corr_matrix = pt.pad(chol_corr_matrix, paddings)

        diag_idx = pt.arange(self.n)
        chol_corr_matrix = chol_corr_matrix[..., diag_idx, diag_idx].set(1)

        # Normalize each row to have Euclidean (L2) norm 1.
        chol_corr_matrix /= pt.linalg.norm(chol_corr_matrix, axis=-1, ord=2)[..., None]

        return chol_corr_matrix

    def log_jac_det(self, unconstrained_vector: TensorLike, *inputs) -> TensorVariable:
        """
        Compute the log determinant of the Jacobian.

        Parameters
        ----------
        unconstrained_vector : TensorLike
            Real-valued vector of length m = n * (n - 1) / 2.
        inputs:
            Additional input values. Not used; included for signature compatibility with other transformations.

        Returns
        -------
        log_jac_det: TensorVariable
            Log determinant of the Jacobian of the transformation.
        """
        unconstrained_vector = pt.as_tensor(unconstrained_vector)
        chol_corr_matrix = self.backward(unconstrained_vector, *inputs)
        n = self.n
        input_dtype = unconstrained_vector.dtype

        # TODO: tfp has a negative sign here; verify if it is needed
        return pt.sum(
            pt.arange(2, 2 + n, dtype=input_dtype)
            * pt.log(pt.diagonal(chol_corr_matrix, axis1=-2, axis2=-1)),
            axis=-1,
        )


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
