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
from typing import List

import pytensor.tensor as pt

from pymc.distributions.continuous import Normal
from pymc.logprob.abstract import _kl_div


@_kl_div.register(Normal, Normal)
def _normal_normal_kl(
    q_dist: Normal,
    p_dist: Normal,
    q_inputs: List[pt.TensorVariable],
    p_inputs: List[pt.TensorVariable],
):
    _, _, _, q_mu, q_sigma = q_inputs
    _, _, _, p_mu, p_sigma = p_inputs
    diff_log_scale = pt.log(q_sigma) - pt.log(p_sigma)
    return (
        0.5 * (q_mu / p_sigma - p_mu / p_sigma) ** 2
        + 0.5 * pt.expm1(2.0 * diff_log_scale)
        - diff_log_scale
    )
