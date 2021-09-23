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

import aesara.tensor as at

from aesara.tensor.subtensor import advanced_set_subtensor1
from aesara.tensor.var import TensorVariable

from pymc3.aesaraf import floatX, gradient
from pymc3.math import invlogit, logit, logsumexp

__all__ = [
    "Transform",
    "stick_breaking",
    "logodds",
    "interval",
    "log_exp_m1",
    "ordered",
    "log",
    "sum_to_1",
    "circular",
    "CholeskyCovPacked",
    "Chain",
]


class Transform:
    """A transformation of a random variable from one space into another.

    Attributes
    ----------
    name: str
        The name of the transform.
    param_extract_fn: callable
        A callable that takes a `TensorVariable` representing a random
        variable, and returns the parameters required by the transform.
        By customizing this function, one can broaden the applicability of--or
        specialize--a `Transform` without the need to create a new `Transform`
        class or altering existing `Transform` classes.  For instance,
        new `RandomVariable`s can supply their own `param_extract_fn`
        implementations that account for their own unique parameterizations.
    """

    __slots__ = ("param_extract_fn",)
    name = ""

    def forward(self, rv_var: TensorVariable, rv_value: TensorVariable) -> TensorVariable:
        """Applies transformation forward to input variable `rv_value`.

        When a transform is applied to a value of some random variable
        `rv_var`, it will transform the random variable `rv_value` after
        sampling from `rv_var`.

        **Do not apply transforms to `rv_var`.**  `rv_var` is only provided
        as a means of describing the random variable associated with `rv_value`.
        `rv_value` is the variable that should be transformed, and the transform
        can use information from `rv_var`--within `param_extract_fn`--to do
        that (e.g. the random variable's parameters via `rv_var.owner.inputs`).

        Parameters
        ----------
        rv_var
            The random variable.
        rv_value
            The variable representing a value of `rv_var`.

        Returns
        --------
        tensor
            Transformed tensor.
        """
        raise NotImplementedError

    def backward(self, rv_var: TensorVariable, rv_value: TensorVariable) -> TensorVariable:
        """Applies inverse of transformation.

        Parameters
        ----------
        rv_var
            The random variable.
        rv_value
            The variable representing a value of `rv_var`.

        Returns
        -------
        tensor
            Inverse transformed tensor.
        """
        raise NotImplementedError

    def jacobian_det(self, rv_var: TensorVariable, rv_value: TensorVariable) -> TensorVariable:
        """Calculates logarithm of the absolute value of the Jacobian determinant
        of the backward transformation.

        Parameters
        ----------
        rv_var
            The random variable.
        rv_value
            The variable representing a value of `rv_var`.

        Returns
        -------
        tensor
            The log abs Jacobian determinant w.r.t. this transform.
        """
        raise NotImplementedError

    def __str__(self):
        return self.name + " transform"


class ElemwiseTransform(Transform):
    def jacobian_det(self, rv_var, rv_value):
        grad = at.reshape(
            gradient(at.sum(self.backward(rv_var, rv_value)), [rv_value]), rv_value.shape
        )
        return at.log(at.abs_(grad))


class Log(ElemwiseTransform):
    name = "log"

    def backward(self, rv_var, rv_value):
        return at.exp(rv_value)

    def forward(self, rv_var, rv_value):
        return at.log(rv_value)

    def jacobian_det(self, rv_var, rv_value):
        return rv_value


log = Log()


class LogExpM1(ElemwiseTransform):
    name = "log_exp_m1"

    def backward(self, rv_var, rv_value):
        return at.softplus(rv_value)

    def forward(self, rv_var, rv_value):
        """Inverse operation of softplus.

        y = Log(Exp(x) - 1)
          = Log(1 - Exp(-x)) + x
        """
        return at.log(1.0 - at.exp(-rv_value)) + rv_value

    def jacobian_det(self, rv_var, rv_value):
        return -at.softplus(-rv_value)


log_exp_m1 = LogExpM1()


class LogOdds(ElemwiseTransform):
    name = "logodds"

    def backward(self, rv_var, rv_value):
        return invlogit(rv_value)

    def forward(self, rv_var, rv_value):
        return logit(rv_value)


logodds = LogOdds()


class Interval(ElemwiseTransform):
    """Transform from real line interval [a,b] to whole real line."""

    name = "interval"

    def __init__(self, param_extract_fn):
        self.param_extract_fn = param_extract_fn

    def backward(self, rv_var, rv_value):
        a, b = self.param_extract_fn(rv_var)

        if a is not None and b is not None:
            sigmoid_x = at.sigmoid(rv_value)
            return sigmoid_x * b + (1 - sigmoid_x) * a
        elif a is not None:
            return at.exp(rv_value) + a
        elif b is not None:
            return b - at.exp(rv_value)
        else:
            return rv_value

    def forward(self, rv_var, rv_value):
        a, b = self.param_extract_fn(rv_var)
        if a is not None and b is not None:
            return at.log(rv_value - a) - at.log(b - rv_value)
        elif a is not None:
            return at.log(rv_value - a)
        elif b is not None:
            return at.log(b - rv_value)
        else:
            return rv_value

    def jacobian_det(self, rv_var, rv_value):
        a, b = self.param_extract_fn(rv_var)

        if a is not None and b is not None:
            s = at.softplus(-rv_value)
            return at.log(b - a) - 2 * s - rv_value
        else:
            return rv_value


interval = Interval


class Ordered(Transform):
    name = "ordered"

    def backward(self, rv_var, rv_value):
        x = at.zeros(rv_value.shape)
        x = at.inc_subtensor(x[..., 0], rv_value[..., 0])
        x = at.inc_subtensor(x[..., 1:], at.exp(rv_value[..., 1:]))
        return at.cumsum(x, axis=-1)

    def forward(self, rv_var, rv_value):
        y = at.zeros(rv_value.shape)
        y = at.inc_subtensor(y[..., 0], rv_value[..., 0])
        y = at.inc_subtensor(y[..., 1:], at.log(rv_value[..., 1:] - rv_value[..., :-1]))
        return y

    def jacobian_det(self, rv_var, rv_value):
        return at.sum(rv_value[..., 1:], axis=-1)


ordered = Ordered()
"""
Instantiation of ``Ordered`` (:class: Ordered) Transform (:class: Transform) class
for use in the ``transform`` argument of a random variable.
"""


class SumTo1(Transform):
    """
    Transforms K - 1 dimensional simplex space (k values in [0,1] and that sum to 1) to a K - 1 vector of values in [0,1]
    This Transformation operates on the last dimension of the input tensor.
    """

    name = "sumto1"

    def backward(self, rv_var, rv_value):
        remaining = 1 - at.sum(rv_value[..., :], axis=-1, keepdims=True)
        return at.concatenate([rv_value[..., :], remaining], axis=-1)

    def forward(self, rv_var, rv_value):
        return rv_value[..., :-1]

    def jacobian_det(self, rv_var, rv_value):
        y = at.zeros(rv_value.shape)
        return at.sum(y, axis=-1)


sum_to_1 = SumTo1()


class StickBreaking(Transform):
    """
    Transforms K - 1 dimensional simplex space (k values in [0,1] and that sum to 1) to a K - 1 vector of real values.
    This is a variant of the isometric logration transformation ::

        Egozcue, J.J., Pawlowsky-Glahn, V., Mateu-Figueras, G. et al.
        Isometric Logratio Transformations for Compositional Data Analysis.
        Mathematical Geology 35, 279–300 (2003). https://doi.org/10.1023/A:1023818214614
    """

    name = "stickbreaking"

    def forward(self, rv_var, rv_value):
        if rv_var.broadcastable[-1]:
            # If this variable is just a bunch of scalars/degenerate
            # Dirichlets, we can't transform it
            return rv_value

        x = rv_value.T
        n = x.shape[0]
        lx = at.log(x)
        shift = at.sum(lx, 0, keepdims=True) / n
        y = lx[:-1] - shift
        return floatX(y.T)

    def backward(self, rv_var, rv_value):
        if rv_var.broadcastable[-1]:
            # If this variable is just a bunch of scalars/degenerate
            # Dirichlets, we can't transform it
            return rv_value

        y = rv_value.T
        y = at.concatenate([y, -at.sum(y, 0, keepdims=True)])
        # "softmax" with vector support and no deprication warning:
        e_y = at.exp(y - at.max(y, 0, keepdims=True))
        x = e_y / at.sum(e_y, 0, keepdims=True)
        return floatX(x.T)

    def jacobian_det(self, rv_var, rv_value):
        if rv_var.broadcastable[-1]:
            # If this variable is just a bunch of scalars/degenerate
            # Dirichlets, we can't transform it
            return at.ones_like(rv_value)

        y = rv_value.T
        Km1 = y.shape[0] + 1
        sy = at.sum(y, 0, keepdims=True)
        r = at.concatenate([y + sy, at.zeros(sy.shape)])
        sr = logsumexp(r, 0, keepdims=True)
        d = at.log(Km1) + (Km1 * sy) - (Km1 * sr)
        return at.sum(d, 0).T


stick_breaking = StickBreaking()


class Circular(ElemwiseTransform):
    """Transforms a linear space into a circular one."""

    name = "circular"

    def backward(self, rv_var, rv_value):
        return at.arctan2(at.sin(rv_value), at.cos(rv_value))

    def forward(self, rv_var, rv_value):
        return at.as_tensor_variable(rv_value)

    def jacobian_det(self, rv_var, rv_value):
        return at.zeros(rv_value.shape)


circular = Circular()


class CholeskyCovPacked(Transform):
    name = "cholesky-cov-packed"

    def __init__(self, param_extract_fn):
        self.param_extract_fn = param_extract_fn

    def backward(self, rv_var, rv_value):
        diag_idxs = self.param_extract_fn(rv_var)
        return advanced_set_subtensor1(rv_value, at.exp(rv_value[diag_idxs]), diag_idxs)

    def forward(self, rv_var, rv_value):
        diag_idxs = self.param_extract_fn(rv_var)
        return advanced_set_subtensor1(rv_value, at.log(rv_value[diag_idxs]), diag_idxs)

    def jacobian_det(self, rv_var, rv_value):
        diag_idxs = self.param_extract_fn(rv_var)
        return at.sum(rv_value[diag_idxs])


class Chain(Transform):

    __slots__ = ("param_extract_fn", "transform_list", "name")

    def __init__(self, transform_list):
        self.transform_list = transform_list
        self.name = "+".join([transf.name for transf in self.transform_list])

    def forward(self, rv_var, rv_value):
        y = rv_value
        for transf in self.transform_list:
            y = transf.forward(rv_var, y)
        return y

    def backward(self, rv_var, rv_value):
        x = rv_value
        for transf in reversed(self.transform_list):
            x = transf.backward(rv_var, x)
        return x

    def jacobian_det(self, rv_var, rv_value):
        y = at.as_tensor_variable(rv_value)
        det_list = []
        ndim0 = y.ndim
        for transf in reversed(self.transform_list):
            det_ = transf.jacobian_det(rv_var, y)
            det_list.append(det_)
            y = transf.backward(rv_var, y)
            ndim0 = min(ndim0, det_.ndim)
        # match the shape of the smallest jacobian_det
        det = 0.0
        for det_ in det_list:
            if det_.ndim > ndim0:
                det += det_.sum(axis=-1)
            else:
                det += det_
        return det
