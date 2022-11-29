#   Copyright 2022- The PyMC Developers
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

import aesara
import aesara.tensor as at
import numpy as np
import scipy.stats as st

from pymc.logprob import joint_logprob
from pymc.logprob.censoring import MeasurableClip
from pymc.logprob.rewriting import construct_ir_fgraph
from pymc.tests.helpers import assert_no_rvs


def test_scalar_clipped_mixture():
    x = at.clip(at.random.normal(loc=1), 0.5, 1.5)
    x.name = "x"
    y = at.random.beta(1, 2, name="y")

    comps = at.stack([x, y])
    comps.name = "comps"
    idxs = at.random.bernoulli(0.4, name="idxs")
    mix = comps[idxs]
    mix.name = "mix"

    mix_vv = mix.clone()
    mix_vv.name = "mix_val"
    idxs_vv = idxs.clone()
    idxs_vv.name = "idxs_val"

    logp = joint_logprob({idxs: idxs_vv, mix: mix_vv})

    logp_fn = aesara.function([idxs_vv, mix_vv], logp)
    assert logp_fn(0, 0.4) == -np.inf
    assert np.isclose(logp_fn(0, 0.5), st.norm.logcdf(0.5, 1) + np.log(0.6))
    assert np.isclose(logp_fn(0, 1.3), st.norm.logpdf(1.3, 1) + np.log(0.6))
    assert np.isclose(logp_fn(1, 0.4), st.beta.logpdf(0.4, 1, 2) + np.log(0.4))


def test_nested_scalar_mixtures():
    x = at.random.normal(loc=-50, name="x")
    y = at.random.normal(loc=50, name="y")
    comps1 = at.stack([x, y])
    comps1.name = "comps1"
    idxs1 = at.random.bernoulli(0.5, name="idxs1")
    mix1 = comps1[idxs1]
    mix1.name = "mix1"

    w = at.random.normal(loc=-100, name="w")
    z = at.random.normal(loc=100, name="z")
    comps2 = at.stack([w, z])
    comps2.name = "comps2"
    idxs2 = at.random.bernoulli(0.5, name="idxs2")
    mix2 = comps2[idxs2]
    mix2.name = "mix2"

    comps12 = at.stack([mix1, mix2])
    comps12.name = "comps12"
    idxs12 = at.random.bernoulli(0.5, name="idxs12")
    mix12 = comps12[idxs12]
    mix12.name = "mix12"

    idxs1_vv = idxs1.clone()
    idxs2_vv = idxs2.clone()
    idxs12_vv = idxs12.clone()
    mix12_vv = mix12.clone()

    logp = joint_logprob({idxs1: idxs1_vv, idxs2: idxs2_vv, idxs12: idxs12_vv, mix12: mix12_vv})
    logp_fn = aesara.function([idxs1_vv, idxs2_vv, idxs12_vv, mix12_vv], logp)

    expected_mu_logpdf = st.norm.logpdf(0) + np.log(0.5) * 3
    assert np.isclose(logp_fn(0, 0, 0, -50), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 1, 0, -50), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 0, 0, 50), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 1, 0, 50), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 0, 1, -100), expected_mu_logpdf)
    assert np.isclose(logp_fn(0, 1, 1, 100), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 0, 1, -100), expected_mu_logpdf)
    assert np.isclose(logp_fn(1, 1, 1, 100), expected_mu_logpdf)

    assert np.isclose(logp_fn(0, 0, 0, 50), st.norm.logpdf(100) + np.log(0.5) * 3)
    assert np.isclose(logp_fn(0, 0, 1, 50), st.norm.logpdf(150) + np.log(0.5) * 3)


def test_unvalued_ir_reversion():
    """Make sure that un-valued IR rewrites are reverted."""
    x_rv = at.random.normal()
    y_rv = at.clip(x_rv, 0, 1)
    z_rv = at.random.normal(y_rv, 1, name="z")
    z_vv = z_rv.clone()

    # Only the `z_rv` is "valued", so `y_rv` doesn't need to be converted into
    # measurable IR.
    rv_values = {z_rv: z_vv}

    z_fgraph, _, memo = construct_ir_fgraph(rv_values)

    assert memo[y_rv] in z_fgraph.preserve_rv_mappings.measurable_conversions

    measurable_y_rv = z_fgraph.preserve_rv_mappings.measurable_conversions[memo[y_rv]]
    assert isinstance(measurable_y_rv.owner.op, MeasurableClip)

    # `construct_ir_fgraph` should've reverted the un-valued measurable IR
    # change
    assert measurable_y_rv not in z_fgraph


def test_shifted_cumsum():
    x = at.random.normal(size=(5,), name="x")
    y = 5 + at.cumsum(x)
    y.name = "y"

    y_vv = y.clone()
    logp = joint_logprob({y: y_vv})
    assert np.isclose(
        logp.eval({y_vv: np.arange(5) + 1 + 5}),
        st.norm.logpdf(1) * 5,
    )


def test_double_log_transform_rv():
    base_rv = at.random.normal(0, 1)
    y_rv = at.log(at.log(base_rv))
    y_rv.name = "y"

    y_vv = y_rv.clone()
    logp = joint_logprob({y_rv: y_vv}, sum=False)
    logp_fn = aesara.function([y_vv], logp)

    log_log_y_val = np.asarray(0.5)
    log_y_val = np.exp(log_log_y_val)
    y_val = np.exp(log_y_val)
    np.testing.assert_allclose(
        logp_fn(log_log_y_val),
        st.norm().logpdf(y_val) + log_y_val + log_log_y_val,
    )


def test_affine_transform_rv():
    loc = at.scalar("loc")
    scale = at.vector("scale")
    rv_size = 3

    y_rv = loc + at.random.normal(0, 1, size=rv_size, name="base_rv") * scale
    y_rv.name = "y"
    y_vv = y_rv.clone()

    logp = joint_logprob({y_rv: y_vv}, sum=False)
    assert_no_rvs(logp)
    logp_fn = aesara.function([loc, scale, y_vv], logp)

    loc_test_val = 4.0
    scale_test_val = np.full(rv_size, 0.5)
    y_test_val = np.full(rv_size, 1.0)

    np.testing.assert_allclose(
        logp_fn(loc_test_val, scale_test_val, y_test_val),
        st.norm(loc_test_val, scale_test_val).logpdf(y_test_val),
    )


def test_affine_log_transform_rv():
    a, b = at.scalars("a", "b")
    base_rv = at.random.lognormal(0, 1, name="base_rv", size=(1, 2))
    y_rv = a + at.log(base_rv) * b
    y_rv.name = "y"

    y_vv = y_rv.clone()

    logp = joint_logprob({y_rv: y_vv}, sum=False)
    logp_fn = aesara.function([a, b, y_vv], logp)

    a_val = -1.5
    b_val = 3.0
    y_val = [[0.1, 0.1]]

    assert np.allclose(
        logp_fn(a_val, b_val, y_val),
        st.norm(a_val, b_val).logpdf(y_val),
    )
