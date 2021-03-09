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

import warnings

import aesara.tensor as at
import numpy as np

from aesara.tensor.subtensor import advanced_set_subtensor1

from pymc3.aesaraf import floatX, gradient
from pymc3.math import invlogit, logit, logsumexp

__all__ = [
    "Transform",
    "stick_breaking",
    "logodds",
    "interval",
    "log_exp_m1",
    "lowerbound",
    "upperbound",
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
    """

    name = ""

    def forward(self, x):
        """Applies transformation forward to input variable `x`.
        When transform is used on some distribution `p`, it will transform the random variable `x` after sampling
        from `p`.

        Parameters
        ----------
        x: tensor
            Input tensor to be transformed.

        Returns
        --------
        tensor
            Transformed tensor.
        """
        raise NotImplementedError

    def backward(self, z):
        """Applies inverse of transformation to input variable `z`.
        When transform is used on some distribution `p`, which has observed values `z`, it is used to
        transform the values of `z` correctly to the support of `p`.

        Parameters
        ----------
        z: tensor
            Input tensor to be inverse transformed.

        Returns
        -------
        tensor
            Inverse transformed tensor.
        """
        raise NotImplementedError

    def jacobian_det(self, x):
        """Calculates logarithm of the absolute value of the Jacobian determinant
        of the backward transformation for input `x`.

        Parameters
        ----------
        x: tensor
            Input to calculate Jacobian determinant of.

        Returns
        -------
        tensor
            The log abs Jacobian determinant of `x` w.r.t. this transform.
        """
        raise NotImplementedError

    def __str__(self):
        return self.name + " transform"


class ElemwiseTransform(Transform):
    def jacobian_det(self, x):
        grad = at.reshape(gradient(at.sum(self.backward(x)), [x]), x.shape)
        return at.log(at.abs_(grad))


class Log(ElemwiseTransform):
    name = "log"

    def backward(self, x):
        return at.exp(x)

    def forward(self, x):
        return at.log(x)

    def jacobian_det(self, x):
        return x


log = Log()


class LogExpM1(ElemwiseTransform):
    name = "log_exp_m1"

    def backward(self, x):
        return at.nnet.softplus(x)

    def forward(self, x):
        """Inverse operation of softplus.

        y = Log(Exp(x) - 1)
          = Log(1 - Exp(-x)) + x
        """
        return at.log(1.0 - at.exp(-x)) + x

    def jacobian_det(self, x):
        return -at.nnet.softplus(-x)


log_exp_m1 = LogExpM1()


class LogOdds(ElemwiseTransform):
    name = "logodds"

    def backward(self, x):
        return invlogit(x, 0.0)

    def forward(self, x):
        return logit(x)


logodds = LogOdds()


class Interval(ElemwiseTransform):
    """Transform from real line interval [a,b] to whole real line."""

    name = "interval"

    def __init__(self, a, b):
        self.a = at.as_tensor_variable(a)
        self.b = at.as_tensor_variable(b)

    def backward(self, x):
        a, b = self.a, self.b
        sigmoid_x = at.nnet.sigmoid(x)
        r = sigmoid_x * b + (1 - sigmoid_x) * a
        return r

    def forward(self, x):
        a, b = self.a, self.b
        return at.log(x - a) - at.log(b - x)

    def jacobian_det(self, x):
        s = at.nnet.softplus(-x)
        return at.log(self.b - self.a) - 2 * s - x


interval = Interval


class LowerBound(ElemwiseTransform):
    """Transform from real line interval [a,inf] to whole real line."""

    name = "lowerbound"

    def __init__(self, a):
        self.a = at.as_tensor_variable(a)

    def backward(self, x):
        a = self.a
        r = at.exp(x) + a
        return r

    def forward(self, x):
        a = self.a
        return at.log(x - a)

    def jacobian_det(self, x):
        return x


lowerbound = LowerBound
"""
Alias for ``LowerBound`` (:class: LowerBound) Transform (:class: Transform) class
for use in the ``transform`` argument of a random variable.
"""


class UpperBound(ElemwiseTransform):
    """Transform from real line interval [-inf,b] to whole real line."""

    name = "upperbound"

    def __init__(self, b):
        self.b = at.as_tensor_variable(b)

    def backward(self, x):
        b = self.b
        r = b - at.exp(x)
        return r

    def forward(self, x):
        b = self.b
        return at.log(b - x)

    def jacobian_det(self, x):
        return x


upperbound = UpperBound
"""
Alias for ``UpperBound`` (:class: UpperBound) Transform (:class: Transform) class
for use in the ``transform`` argument of a random variable.
"""


class Ordered(Transform):
    name = "ordered"

    def backward(self, y):
        x = at.zeros(y.shape)
        x = at.inc_subtensor(x[..., 0], y[..., 0])
        x = at.inc_subtensor(x[..., 1:], at.exp(y[..., 1:]))
        return at.cumsum(x, axis=-1)

    def forward(self, x):
        y = at.zeros(x.shape)
        y = at.inc_subtensor(y[..., 0], x[..., 0])
        y = at.inc_subtensor(y[..., 1:], at.log(x[..., 1:] - x[..., :-1]))
        return y

    def jacobian_det(self, y):
        return at.sum(y[..., 1:], axis=-1)


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

    def backward(self, y):
        remaining = 1 - at.sum(y[..., :], axis=-1, keepdims=True)
        return at.concatenate([y[..., :], remaining], axis=-1)

    def forward(self, x):
        return x[..., :-1]

    def jacobian_det(self, x):
        y = at.zeros(x.shape)
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

    def __init__(self, eps=None):
        if eps is not None:
            warnings.warn(
                "The argument `eps` is deprecated and will not be used.", DeprecationWarning
            )

    def forward(self, x_):
        x = x_.T
        n = x.shape[0]
        lx = at.log(x)
        shift = at.sum(lx, 0, keepdims=True) / n
        y = lx[:-1] - shift
        return floatX(y.T)

    def backward(self, y_):
        y = y_.T
        y = at.concatenate([y, -at.sum(y, 0, keepdims=True)])
        # "softmax" with vector support and no deprication warning:
        e_y = at.exp(y - at.max(y, 0, keepdims=True))
        x = e_y / at.sum(e_y, 0, keepdims=True)
        return floatX(x.T)

    def jacobian_det(self, y_):
        y = y_.T
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

    def backward(self, y):
        return at.arctan2(at.sin(y), at.cos(y))

    def forward(self, x):
        return at.as_tensor_variable(x)

    def jacobian_det(self, x):
        return at.zeros(x.shape)


circular = Circular()


class CholeskyCovPacked(Transform):
    name = "cholesky-cov-packed"

    def __init__(self, n):
        self.diag_idxs = np.arange(1, n + 1).cumsum() - 1

    def backward(self, x):
        return advanced_set_subtensor1(x, at.exp(x[self.diag_idxs]), self.diag_idxs)

    def forward(self, y):
        return advanced_set_subtensor1(y, at.log(y[self.diag_idxs]), self.diag_idxs)

    def jacobian_det(self, y):
        return at.sum(y[self.diag_idxs])


class Chain(Transform):
    def __init__(self, transform_list):
        self.transform_list = transform_list
        self.name = "+".join([transf.name for transf in self.transform_list])

    def forward(self, x):
        y = x
        for transf in self.transform_list:
            y = transf.forward(y)
        return y

    def backward(self, y):
        x = y
        for transf in reversed(self.transform_list):
            x = transf.backward(x)
        return x

    def jacobian_det(self, y):
        y = at.as_tensor_variable(y)
        det_list = []
        ndim0 = y.ndim
        for transf in reversed(self.transform_list):
            det_ = transf.jacobian_det(y)
            det_list.append(det_)
            y = transf.backward(y)
            ndim0 = min(ndim0, det_.ndim)
        # match the shape of the smallest jacobian_det
        det = 0.0
        for det_ in det_list:
            if det_.ndim > ndim0:
                det += det_.sum(axis=-1)
            else:
                det += det_
        return det
