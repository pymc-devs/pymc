#   Copyright 2025 - present The PyMC Developers
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
import pytensor.tensor as pt
import pytensor.xtensor as ptx

from pymc.logprob.transforms import Transform


class DimTransform(Transform):
    """Base class for transforms that are applied to dim distriubtions."""


class LogTransform(DimTransform):
    name = "log"

    def forward(self, value, *inputs):
        return ptx.math.log(value)

    def backward(self, value, *inputs):
        return ptx.math.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


log_transform = LogTransform()


class LogOddsTransform(DimTransform):
    name = "logodds"

    def backward(self, value, *inputs):
        return ptx.math.expit(value)

    def forward(self, value, *inputs):
        return ptx.math.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = ptx.math.sigmoid(value)
        return ptx.math.log(sigmoid_value) + ptx.math.log1p(-sigmoid_value)


log_odds_transform = LogOddsTransform()


class SimplexTransform(DimTransform):
    name = "simplex"

    def __init__(self, dim: str):
        self.core_dim = dim

    def forward(self, value, *inputs):
        log_value = ptx.math.log(value)
        N = value.sizes[self.core_dim].astype(value.dtype)
        shift = log_value.sum(self.core_dim) / N
        return log_value.isel({self.core_dim: slice(None, -1)}) - shift

    def backward(self, value, *inputs):
        value = ptx.concat([value, -value.sum(self.core_dim)], dim=self.core_dim)
        exp_value_max = ptx.math.exp(value - value.max(self.core_dim))
        return exp_value_max / exp_value_max.sum(self.core_dim)

    def log_jac_det(self, value, *inputs):
        N = value.sizes[self.core_dim] + 1
        N = N.astype(value.dtype)
        sum_value = value.sum(self.core_dim)
        value_sum_expanded = value + sum_value
        value_sum_expanded = ptx.concat([value_sum_expanded, 0], dim=self.core_dim)
        logsumexp_value_expanded = ptx.math.logsumexp(value_sum_expanded, dim=self.core_dim)
        res = ptx.math.log(N) + (N * sum_value) - (N * logsumexp_value_expanded)
        return res


class ZeroSumTransform(DimTransform):
    name = "zerosum"

    def __init__(self, dims: tuple[str, ...]):
        self.dims = dims

    @staticmethod
    def extend_dim(array, dim):
        n = (array.sizes[dim] + 1).astype("floatX")
        sum_vals = array.sum(dim)
        norm = sum_vals / (pt.sqrt(n) + n)
        fill_val = norm - sum_vals / pt.sqrt(n)

        out = ptx.concat([array, fill_val], dim=dim)
        return out - norm

    @staticmethod
    def reduce_dim(array, dim):
        n = array.sizes[dim].astype("floatX")
        last = array.isel({dim: -1})

        sum_vals = -last * pt.sqrt(n)
        norm = sum_vals / (pt.sqrt(n) + n)
        return array.isel({dim: slice(None, -1)}) + norm

    def forward(self, value, *rv_inputs):
        for dim in self.dims:
            value = self.reduce_dim(value, dim=dim)
        return value

    def backward(self, value, *rv_inputs):
        for dim in self.dims:
            value = self.extend_dim(value, dim=dim)
        return value

    def log_jac_det(self, value, *rv_inputs):
        # Use following once broadcast_like is implemented
        # as_xtensor(0).broadcast_like(value, exclude=self.dims)`
        return value.sum(self.dims) * 0
