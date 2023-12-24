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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.

import numpy as np
import pytensor.tensor as pt
import pytest
import scipy.stats.distributions as sp

from pytensor.graph import ancestors
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    IncSubtensor,
    Subtensor,
)

from pymc.distributions.transforms import logodds
from pymc.logprob.basic import conditional_logp
from pymc.logprob.rewriting import cleanup_ir, local_lift_DiracDelta
from pymc.logprob.transform_value import TransformedValue, TransformValuesRewrite
from pymc.logprob.utils import DiracDelta, dirac_delta


def test_local_lift_DiracDelta():
    c_at = pt.vector()
    dd_at = dirac_delta(c_at)

    Z_at = pt.cast(dd_at, "int64")

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Elemwise)

    Z_at = dd_at.dimshuffle("x", 0)

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, DimShuffle)

    Z_at = dd_at[0]

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Subtensor)

    # Don't lift multi-output `Op`s
    c_at = pt.matrix()
    dd_at = dirac_delta(c_at)
    Z_at = pt.nlinalg.svd(dd_at)[0]

    res = rewrite_graph(Z_at, custom_rewrite=in2out(local_lift_DiracDelta), clone=False)
    assert res is Z_at


def test_local_remove_DiracDelta():
    c_at = pt.vector()
    dd_at = dirac_delta(c_at) + dirac_delta(5)
    assert sum(isinstance(v.owner.op, DiracDelta) for v in ancestors([dd_at]) if v.owner) == 2

    cleanup_ir([dd_at])
    assert not any(isinstance(v.owner.op, DiracDelta) for v in ancestors([dd_at]) if v.owner)


def test_local_remove_TransformedVariable():
    p_rv = pt.random.beta(1, 1, name="p")
    p_vv = p_rv.clone()

    tr = TransformValuesRewrite({p_vv: logodds})
    [p_logp] = conditional_logp({p_rv: p_vv}, extra_rewrites=tr).values()

    assert not any(isinstance(v.owner.op, TransformedValue) for v in ancestors([p_logp]) if v.owner)


@pytest.mark.parametrize(
    "indices, size",
    [
        (slice(0, 2), 5),
        (np.r_[True, True, False, False, True], 5),
        (np.r_[0, 1, 4], 5),
        ((np.array([0, 1, 4]), np.array([0, 1, 4])), (5, 5)),
    ],
)
def test_joint_logprob_incsubtensor(indices, size):
    """Make sure we can compute a joint log-probability for ``Y[idx] = data`` where ``Y`` is univariate."""

    rng = np.random.RandomState(232)
    mu = np.power(10, np.arange(np.prod(size))).reshape(size)
    sigma = 0.001
    data = rng.normal(mu[indices], 1.0)
    y_val = rng.normal(mu, sigma, size=size)

    Y_base_rv = pt.random.normal(mu, sigma, size=size)
    Y_rv = pt.set_subtensor(Y_base_rv[indices], data)
    Y_rv.name = "Y"
    y_value_var = Y_rv.clone()
    y_value_var.name = "y"

    assert isinstance(Y_rv.owner.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1))

    Y_rv_logp = conditional_logp({Y_rv: y_value_var})
    Y_rv_logp_combined = pt.add(*Y_rv_logp.values())

    obs_logps = Y_rv_logp_combined.eval({y_value_var: y_val})

    y_val_idx = y_val.copy()
    y_val_idx[indices] = data
    exp_obs_logps = sp.norm.logpdf(y_val_idx, mu, sigma)

    np.testing.assert_almost_equal(obs_logps, exp_obs_logps)
