#   Copyright 2021 The PyMC Developers
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


from aesara.tensor.random.op import RandomVariable

from pymc3.aesaraf import walk_model
from pymc3.distributions.continuous import Normal, Uniform
from pymc3.distributions.logp import logpt
from pymc3.model import Model


def test_logpt_basic():
    """Make sure we can compute a log-likelihood for a hierarchical model with transforms."""

    with Model() as m:
        a = Uniform("a", 0.0, 1.0)
        c = Normal("c")
        b_l = c * a + 2.0
        b = Uniform("b", b_l, b_l + 1.0)

    a_value_var = m.rvs_to_values[a]
    assert a_value_var.tag.transform

    b_value_var = m.rvs_to_values[b]
    assert b_value_var.tag.transform

    c_value_var = m.rvs_to_values[c]

    b_logp = logpt(b, b_value_var)

    res_ancestors = list(walk_model((b_logp,), walk_past_rvs=True))
    res_rv_ancestors = [
        v for v in res_ancestors if v.owner and isinstance(v.owner.op, RandomVariable)
    ]

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert len(res_rv_ancestors) == 0
    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors
    assert a_value_var in res_ancestors
