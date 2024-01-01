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
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats as st

from pymc import draw, logp
from pymc.logprob.abstract import MeasurableVariable
from pymc.logprob.basic import conditional_logp
from pymc.logprob.censoring import MeasurableClip
from pymc.logprob.rewriting import construct_ir_fgraph
from pymc.testing import assert_no_rvs


def test_scalar_clipped_mixture():
    x = pt.clip(pt.random.normal(loc=1), 0.5, 1.5)
    x.name = "x"
    y = pt.random.beta(1, 2, name="y")

    comps = pt.stack([x, y])
    comps.name = "comps"
    idxs = pt.random.bernoulli(0.4, name="idxs")
    mix = comps[idxs]
    mix.name = "mix"

    mix_vv = mix.clone()
    mix_vv.name = "mix_val"
    idxs_vv = idxs.clone()
    idxs_vv.name = "idxs_val"

    logp = conditional_logp({idxs: idxs_vv, mix: mix_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    logp_fn = pytensor.function([idxs_vv, mix_vv], logp_combined)
    assert logp_fn(0, 0.4) == -np.inf
    assert np.isclose(logp_fn(0, 0.5), st.norm.logcdf(0.5, 1) + np.log(0.6))
    assert np.isclose(logp_fn(0, 1.3), st.norm.logpdf(1.3, 1) + np.log(0.6))
    assert np.isclose(logp_fn(1, 0.4), st.beta.logpdf(0.4, 1, 2) + np.log(0.4))


def test_nested_scalar_mixtures():
    x = pt.random.normal(loc=-50, name="x")
    y = pt.random.normal(loc=50, name="y")
    comps1 = pt.stack([x, y])
    comps1.name = "comps1"
    idxs1 = pt.random.bernoulli(0.5, name="idxs1")
    mix1 = comps1[idxs1]
    mix1.name = "mix1"

    w = pt.random.normal(loc=-100, name="w")
    z = pt.random.normal(loc=100, name="z")
    comps2 = pt.stack([w, z])
    comps2.name = "comps2"
    idxs2 = pt.random.bernoulli(0.5, name="idxs2")
    mix2 = comps2[idxs2]
    mix2.name = "mix2"

    comps12 = pt.stack([mix1, mix2])
    comps12.name = "comps12"
    idxs12 = pt.random.bernoulli(0.5, name="idxs12")
    mix12 = comps12[idxs12]
    mix12.name = "mix12"

    idxs1_vv = idxs1.clone()
    idxs2_vv = idxs2.clone()
    idxs12_vv = idxs12.clone()
    mix12_vv = mix12.clone()

    logp = conditional_logp({idxs1: idxs1_vv, idxs2: idxs2_vv, idxs12: idxs12_vv, mix12: mix12_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])

    logp_fn = pytensor.function([idxs1_vv, idxs2_vv, idxs12_vv, mix12_vv], logp_combined)

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


@pytest.mark.parametrize("nested", (False, True))
def test_unvalued_ir_reversion(nested):
    """Make sure that un-valued IR rewrites are reverted."""
    x_rv = pt.random.normal()
    y_rv = pt.clip(x_rv, 0, 1)
    if nested:
        y_rv = y_rv + 5
    z_rv = pt.random.normal(y_rv, 1, name="z")
    z_vv = z_rv.clone()

    # Only the `z_rv` is "valued", so `y_rv` doesn't need to be converted into
    # measurable IR.
    rv_values = {z_rv: z_vv}

    z_fgraph, _, memo = construct_ir_fgraph(rv_values)

    # assert len(z_fgraph.preserve_rv_mappings.measurable_conversions) == 1
    assert (
        sum(isinstance(node.op, MeasurableVariable) for node in z_fgraph.apply_nodes) == 2
    )  # Just the 2 rvs


def test_shifted_cumsum():
    x = pt.random.normal(size=(5,), name="x")
    y = 5 + pt.cumsum(x)
    y.name = "y"

    y_vv = y.clone()
    logprob = logp(y, y_vv)
    assert np.isclose(
        logprob.eval({y_vv: np.arange(5) + 1 + 5}).sum(),
        st.norm.logpdf(1) * 5,
    )


def test_double_log_transform_rv():
    base_rv = pt.random.normal(0, 1)
    y_rv = pt.log(pt.log(base_rv))
    y_rv.name = "y"

    y_vv = y_rv.clone()
    logprob = logp(y_rv, y_vv)
    logp_fn = pytensor.function([y_vv], logprob)

    log_log_y_val = np.asarray(0.5)
    log_y_val = np.exp(log_log_y_val)
    y_val = np.exp(log_y_val)
    np.testing.assert_allclose(
        logp_fn(log_log_y_val),
        st.norm().logpdf(y_val) + log_y_val + log_log_y_val,
    )


def test_affine_transform_rv():
    loc = pt.scalar("loc")
    scale = pt.vector("scale")
    rv_size = 3

    y_rv = loc + pt.random.normal(0, 1, size=rv_size, name="base_rv") * scale
    y_rv.name = "y"
    y_vv = y_rv.clone()

    logprob = logp(y_rv, y_vv)
    assert_no_rvs(logprob)
    logp_fn = pytensor.function([loc, scale, y_vv], logprob)

    loc_test_val = 4.0
    scale_test_val = np.full(rv_size, 0.5)
    y_test_val = np.full(rv_size, 1.0)

    np.testing.assert_allclose(
        logp_fn(loc_test_val, scale_test_val, y_test_val),
        st.norm(loc_test_val, scale_test_val).logpdf(y_test_val),
    )


def test_affine_log_transform_rv():
    a, b = pt.scalars("a", "b")
    base_rv = pt.random.lognormal(0, 1, name="base_rv", size=(1, 2))
    y_rv = a + pt.log(base_rv) * b
    y_rv.name = "y"

    y_vv = y_rv.clone()

    logprob = logp(y_rv, y_vv)
    logp_fn = pytensor.function([a, b, y_vv], logprob)

    a_val = -1.5
    b_val = 3.0
    y_val = [[0.1, 0.1]]

    assert np.allclose(
        logp_fn(a_val, b_val, y_val),
        st.norm(a_val, b_val).logpdf(y_val),
    )


@pytest.mark.parametrize("reverse", (False, True))
def test_affine_join_interdependent(reverse):
    x = pt.random.normal(name="x")
    y_rvs = []
    prev_rv = x
    for i in range(3):
        next_rv = pt.exp(prev_rv + pt.random.beta(3, 1, name=f"y{i}", size=(1, 2)))
        y_rvs.append(next_rv)
        prev_rv = next_rv

    if reverse:
        y_rvs = y_rvs[::-1]

    ys = pt.concatenate(y_rvs, axis=0)
    ys.name = "ys"

    x_vv = x.clone()
    ys_vv = ys.clone()

    logp = conditional_logp({x: x_vv, ys: ys_vv})
    logp_combined = pt.sum([pt.sum(factor) for factor in logp.values()])
    assert_no_rvs(logp_combined)

    y0_vv = y_rvs[0].clone()
    y1_vv = y_rvs[1].clone()
    y2_vv = y_rvs[2].clone()

    ref_logp = conditional_logp({x: x_vv, y_rvs[0]: y0_vv, y_rvs[1]: y1_vv, y_rvs[2]: y2_vv})
    ref_logp_combined = pt.sum([pt.sum(factor) for factor in ref_logp.values()])

    rng = np.random.default_rng()
    x_vv_test, ys_vv_test = draw([x, ys], random_seed=rng)
    ys_vv_test = rng.normal(size=(3, 2))
    np.testing.assert_allclose(
        logp_combined.eval({x_vv: x_vv_test, ys_vv: ys_vv_test}),
        ref_logp_combined.eval(
            {
                x_vv: x_vv_test,
                y0_vv: ys_vv_test[0:1],
                y1_vv: ys_vv_test[1:2],
                y2_vv: ys_vv_test[2:3],
            }
        ),
    )
