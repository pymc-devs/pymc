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

import aesara.tensor as at

from aeppl.transforms import (
    CircularTransform,
    IntervalTransform,
    LogOddsTransform,
    LogTransform,
    RVTransform,
    Simplex,
)
from aesara.tensor.subtensor import advanced_set_subtensor1

__all__ = [
    "RVTransform",
    "simplex",
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
    name = "cholesky-cov-packed"

    def __init__(self, param_extract_fn):
        self.param_extract_fn = param_extract_fn

    def backward(self, value, *inputs):
        diag_idxs = self.param_extract_fn(inputs)
        return advanced_set_subtensor1(value, at.exp(value[diag_idxs]), diag_idxs)

    def forward(self, value, *inputs):
        diag_idxs = self.param_extract_fn(inputs)
        return advanced_set_subtensor1(value, at.log(value[diag_idxs]), diag_idxs)

    def log_jac_det(self, value, *inputs):
        diag_idxs = self.param_extract_fn(inputs)
        return at.sum(value[diag_idxs])


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
logodds = LogOddsTransform()
interval = IntervalTransform
log_exp_m1 = LogExpM1()
ordered = Ordered()
log = LogTransform()
sum_to_1 = SumTo1()
circular = CircularTransform()
