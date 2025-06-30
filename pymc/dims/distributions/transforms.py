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
