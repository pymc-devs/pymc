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
import scipy.stats.distributions as sp

from pytensor import function
from pytensor.graph.basic import Variable, equal_computations
from pytensor.ifelse import ifelse
from pytensor.tensor.random.basic import CategoricalRV
from pytensor.tensor.shape import shape_tuple
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    AdvancedSubtensor1,
    Subtensor,
    as_index_constant,
)

from pymc.logprob.abstract import MeasurableVariable
from pymc.logprob.basic import conditional_logp, logp
from pymc.logprob.mixture import MeasurableSwitchMixture, MixtureRV, expand_indices
from pymc.logprob.rewriting import construct_ir_fgraph
from pymc.logprob.utils import dirac_delta
from pymc.testing import assert_no_rvs
from tests.logprob.utils import scipy_logprob


def test_mixture_basics():
    def create_mix_model(size, axis):
        X_rv = pt.random.normal(0, 1, size=size, name="X")
        Y_rv = pt.random.gamma(0.5, scale=2.0, size=size, name="Y")

        p_at = pt.scalar("p")
        p_at.tag.test_value = 0.5

        I_rv = pt.random.bernoulli(p_at, size=size, name="I")
        i_vv = I_rv.clone()
        i_vv.name = "i"

        if isinstance(axis, Variable):
            M_rv = pt.join(axis, X_rv, Y_rv)[I_rv]
        else:
            M_rv = pt.stack([X_rv, Y_rv], axis=axis)[I_rv]

        M_rv.name = "M"
        m_vv = M_rv.clone()
        m_vv.name = "m"

        return locals()

    env = create_mix_model(None, 0)
    X_rv = env["X_rv"]
    I_rv = env["I_rv"]
    i_vv = env["i_vv"]
    M_rv = env["M_rv"]
    m_vv = env["m_vv"]

    x_vv = X_rv.clone()
    x_vv.name = "x"

    with pytest.raises(RuntimeError, match="could not be derived: {m}"):
        conditional_logp({M_rv: m_vv, I_rv: i_vv, X_rv: x_vv})

    with pytest.raises(RuntimeError, match="could not be derived: {m}"):
        axis_at = pt.lscalar("axis")
        axis_at.tag.test_value = 0
        env = create_mix_model((2,), axis_at)
        I_rv = env["I_rv"]
        i_vv = env["i_vv"]
        M_rv = env["M_rv"]
        m_vv = env["m_vv"]
        conditional_logp({M_rv: m_vv, I_rv: i_vv})


@pytensor.config.change_flags(compute_test_value="warn")
@pytest.mark.parametrize(
    "op_constructor",
    [
        lambda _I, _X, _Y: pt.stack([_X, _Y])[_I],
        lambda _I, _X, _Y: pt.switch(_I, _X, _Y),
    ],
)
def test_compute_test_value(op_constructor):
    X_rv = pt.random.normal(0, 1, name="X")
    Y_rv = pt.random.gamma(0.5, scale=2.0, name="Y")

    p_at = pt.scalar("p")
    p_at.tag.test_value = 0.3

    I_rv = pt.random.bernoulli(p_at, name="I")

    i_vv = I_rv.clone()
    i_vv.name = "i"

    M_rv = op_constructor(I_rv, X_rv, Y_rv)
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    del M_rv.tag.test_value

    M_logp = conditional_logp({M_rv: m_vv, I_rv: i_vv})
    M_logp_combined = pt.add(*M_logp.values())

    assert isinstance(M_logp_combined.tag.test_value, np.ndarray)


@pytest.mark.parametrize(
    "p_val, size, supported",
    [
        (np.array(0.0, dtype=pytensor.config.floatX), (), True),
        (np.array(1.0, dtype=pytensor.config.floatX), (), True),
        (np.array([0.1, 0.9], dtype=pytensor.config.floatX), (), True),
        # The cases belowe are not supported because they may pick repeated values via AdvancedIndexing
        (np.array(0.0, dtype=pytensor.config.floatX), (2,), False),
        (np.array(1.0, dtype=pytensor.config.floatX), (2, 1), False),
        (np.array(1.0, dtype=pytensor.config.floatX), (2, 3), False),
        (np.array([0.1, 0.9], dtype=pytensor.config.floatX), (2, 3), False),
    ],
)
def test_hetero_mixture_binomial(p_val, size, supported):
    X_rv = pt.random.normal(0, 1, size=size, name="X")
    Y_rv = pt.random.gamma(0.5, scale=2.0, size=size, name="Y")

    if np.ndim(p_val) == 0:
        p_at = pt.scalar("p")
        p_at.tag.test_value = p_val
        I_rv = pt.random.bernoulli(p_at, size=size, name="I")
        p_val_1 = p_val
    else:
        p_at = pt.vector("p")
        p_at.tag.test_value = np.array(p_val, dtype=pytensor.config.floatX)
        I_rv = pt.random.categorical(p_at, size=size, name="I")
        p_val_1 = p_val[1]

    i_vv = I_rv.clone()
    i_vv.name = "i"

    M_rv = pt.stack([X_rv, Y_rv])[I_rv]
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    if supported:
        M_logp = conditional_logp({M_rv: m_vv, I_rv: i_vv})
        M_logp_combined = pt.add(*M_logp.values())
    else:
        with pytest.raises(RuntimeError, match="could not be derived: {m}"):
            conditional_logp({M_rv: m_vv, I_rv: i_vv})
        return

    M_logp_fn = pytensor.function([p_at, m_vv, i_vv], M_logp_combined)

    assert_no_rvs(M_logp_fn.maker.fgraph.outputs[0])

    decimals = 6 if pytensor.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    bern_sp = sp.bernoulli(p_val_1)
    norm_sp = sp.norm(loc=0, scale=1)
    gamma_sp = sp.gamma(0.5, scale=2.0)

    for i in range(10):
        i_val = bern_sp.rvs(size=size, random_state=test_val_rng)
        x_val = norm_sp.rvs(size=size, random_state=test_val_rng)
        y_val = gamma_sp.rvs(size=size, random_state=test_val_rng)

        component_logps = np.stack([norm_sp.logpdf(x_val), gamma_sp.logpdf(y_val)])[i_val]
        exp_obs_logps = component_logps + bern_sp.logpmf(i_val)

        m_val = np.stack([x_val, y_val])[i_val]
        logp_vals = M_logp_fn(p_val, m_val, i_val)

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)


@pytest.mark.parametrize(
    "X_args, Y_args, Z_args, p_val, comp_size, idx_size, extra_indices, join_axis, supported",
    [
        # Scalar components, scalar index
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (),
            (),
            (),
            0,
            True,
        ),
        # Degenerate vector mixture components, scalar index along join axis
        (
            (
                np.array([0], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5], dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array([100], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            None,
            (),
            (),
            0,
            True,
        ),
        # Degenerate vector mixture components, scalar index along join axis (axis=1)
        (
            (
                np.array([0], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5], dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array([100], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            None,
            (),
            (slice(None),),
            1,
            True,
        ),
        # Vector mixture components, scalar index along the join axis
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (4,),
            (),
            (),
            0,
            True,
        ),
        # Vector mixture components, scalar index along the join axis (axis=1)
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (4,),
            (),
            (slice(None),),
            1,
            True,
        ),
        # Vector mixture components, scalar index that mixes across components
        pytest.param(
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.1, 0.3], dtype=pytensor.config.floatX),
            (4,),
            (),
            (),
            1,
            True,
            marks=pytest.mark.xfail(
                AssertionError,
                match="Arrays are not almost equal to 6 decimals",  # This is ignored, but that's where it should fail!
                reason="IfElse Mixture logprob fails when indexing mixes across components",
            ),
        ),
        # Matrix components, scalar index along first axis
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (2, 3),
            (),
            (),
            0,
            True,
        ),
        # All the tests below rely on AdvancedIndexing, which is not supported at the moment
        # See https://github.com/pymc-devs/pymc/issues/6398
        # Scalar mixture components, vector index along first axis
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (),
            (6,),
            (),
            0,
            False,
        ),
        # Vector mixture components, vector index along first axis
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (2,),
            (2,),
            (slice(None),),
            0,
            False,
        ),
        # Vector mixture components, vector index along last axis
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (2,),
            (4,),
            (slice(None),),
            1,
            False,
        ),
        # Vector mixture components (with degenerate vector parameters), vector index along first axis
        (
            (
                np.array([0], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5], dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array([100], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (2,),
            (2,),
            (),
            0,
            False,
        ),
        # Vector mixture components (with vector parameters), vector index along first axis
        (
            (
                np.array([0, -100], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5, 1], dtype=pytensor.config.floatX),
                np.array([2.0, 1], dtype=pytensor.config.floatX),
            ),
            (
                np.array([100, 1000], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([[0.1, 0.5, 0.4], [0.4, 0.1, 0.5]], dtype=pytensor.config.floatX),
            (2,),
            (2,),
            (),
            0,
            False,
        ),
        # Vector mixture components (with vector parameters), vector index along first axis, implicit sizes
        (
            (
                np.array([0, -100], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5, 1], dtype=pytensor.config.floatX),
                np.array([2.0, 1], dtype=pytensor.config.floatX),
            ),
            (
                np.array([100, 1000], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([[0.1, 0.5, 0.4], [0.4, 0.1, 0.5]], dtype=pytensor.config.floatX),
            None,
            None,
            (),
            0,
            False,
        ),
        # Matrix mixture components, matrix index
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (2, 3),
            (2, 3),
            (),
            0,
            False,
        ),
        # Vector components, matrix indexing (constant along first dimension, then random)
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (5,),
            (5,),
            (np.arange(5),),
            0,
            False,
        ),
        # Vector mixture components, tensor3 indexing (constant along first dimension, then degenerate, then random)
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(2.0, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (5,),
            (5,),
            (np.arange(5), None),
            0,
            False,
        ),
    ],
)
def test_hetero_mixture_categorical(
    X_args, Y_args, Z_args, p_val, comp_size, idx_size, extra_indices, join_axis, supported
):
    X_rv = pt.random.normal(*X_args, size=comp_size, name="X")
    Y_rv = pt.random.gamma(Y_args[0], scale=Y_args[1], size=comp_size, name="Y")
    Z_rv = pt.random.normal(*Z_args, size=comp_size, name="Z")

    p_at = pt.as_tensor(p_val).type()
    p_at.name = "p"
    p_at.tag.test_value = np.array(p_val, dtype=pytensor.config.floatX)
    I_rv = pt.random.categorical(p_at, size=idx_size, name="I")

    i_vv = I_rv.clone()
    i_vv.name = "i"

    indices_at = list(extra_indices)
    indices_at.insert(join_axis, I_rv)
    indices_at = tuple(indices_at)

    M_rv = pt.stack([X_rv, Y_rv, Z_rv], axis=join_axis)[indices_at]
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    if supported:
        logp_parts = conditional_logp({M_rv: m_vv, I_rv: i_vv}, sum=False)
    else:
        with pytest.raises(RuntimeError, match="could not be derived: {m}"):
            conditional_logp({M_rv: m_vv, I_rv: i_vv}, sum=False)
        return

    I_logp_fn = pytensor.function([p_at, i_vv], logp_parts[i_vv])
    M_logp_fn = pytensor.function([m_vv, i_vv], logp_parts[m_vv])

    assert_no_rvs(I_logp_fn.maker.fgraph.outputs[0])
    assert_no_rvs(M_logp_fn.maker.fgraph.outputs[0])

    decimals = 6 if pytensor.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    norm_1_sp = sp.norm(loc=X_args[0], scale=X_args[1])
    gamma_sp = sp.gamma(Y_args[0], scale=Y_args[1])
    norm_2_sp = sp.norm(loc=Z_args[0], scale=Z_args[1])

    # Handle scipy annoying squeeze of random draws
    real_comp_size = tuple(X_rv.shape.eval())

    for i in range(10):
        i_val = CategoricalRV.rng_fn(test_val_rng, p_val, idx_size)

        indices_val = list(extra_indices)
        indices_val.insert(join_axis, i_val)
        indices_val = tuple(indices_val)

        x_val = np.broadcast_to(
            norm_1_sp.rvs(size=comp_size, random_state=test_val_rng), real_comp_size
        )
        y_val = np.broadcast_to(
            gamma_sp.rvs(size=comp_size, random_state=test_val_rng), real_comp_size
        )
        z_val = np.broadcast_to(
            norm_2_sp.rvs(size=comp_size, random_state=test_val_rng), real_comp_size
        )

        component_logps = np.stack(
            [norm_1_sp.logpdf(x_val), gamma_sp.logpdf(y_val), norm_2_sp.logpdf(z_val)],
            axis=join_axis,
        )[indices_val]
        index_logps = scipy_logprob(i_val, p_val)
        exp_obs_logps = component_logps + index_logps[(Ellipsis,) + (None,) * join_axis]

        m_val = np.stack([x_val, y_val, z_val], axis=join_axis)[indices_val]

        I_logp_vals = I_logp_fn(p_val, i_val)
        M_logp_vals = M_logp_fn(m_val, i_val)

        logp_vals = M_logp_vals + I_logp_vals[(Ellipsis,) + (None,) * join_axis]

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)


@pytest.mark.parametrize(
    "A_parts, indices",
    [
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.array([[0, 1], [2, 2]]), slice(2, 3)),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), np.array([[0, 1], [2, 2]])),
        ),
        (
            (
                np.random.normal(size=(5, 4, 3)),
                np.random.normal(size=(5, 4, 3)),
                np.random.normal(size=(5, 4, 3)),
            ),
            (
                np.array([[0], [2], [1]]),
                slice(None),
                np.array([2, 1]),
                slice(2, 3),
            ),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), np.array([0, 1, 2])),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.array([[0, 1], [2, 2]]), np.array([[0, 1], [2, 2]])),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (
                np.array([[0, 1], [2, 2]]),
                np.array([[0, 1], [2, 2]]),
                np.array([[0, 1], [2, 2]]),
            ),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.array([[0, 1], [2, 2]]), np.array([[0, 1], [2, 2]]), 1),
        ),
        (
            (
                np.random.normal(size=(5, 4, 3)),
                np.random.normal(size=(5, 4, 3)),
            ),
            (slice(0, 2),),
        ),
        (
            (
                np.random.normal(size=(5, 4, 3)),
                np.random.normal(size=(5, 4, 3)),
            ),
            (slice(0, 2), np.random.randint(3, size=(2, 3))),
        ),
    ],
)
def test_expand_indices_basic(A_parts, indices):
    A = pt.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


@pytest.mark.parametrize(
    "A_parts, indices",
    [
        (
            (
                np.random.normal(size=(6, 5, 4, 3)),
                np.random.normal(size=(6, 5, 4, 3)),
                np.random.normal(size=(6, 5, 4, 3)),
            ),
            (
                slice(None),
                np.array([[0], [2], [1]]),
                slice(None),
                np.array([2, 1]),
                slice(2, 3),
            ),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.array([[0, 1], [2, 2]]), slice(None), np.array([[0, 1], [2, 2]])),
        ),
    ],
)
def test_expand_indices_moved_subspaces(A_parts, indices):
    A = pt.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


@pytest.mark.parametrize(
    "A_parts, indices",
    [
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), np.array([0, 1, 2]), 1),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), 1, np.array([0, 1, 2])),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (1, slice(2, 3), np.array([0, 1, 2])),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (np.random.randint(2, size=(4, 3)), 1, 0),
        ),
    ],
)
def test_expand_indices_single_indices(A_parts, indices):
    A = pt.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


@pytest.mark.parametrize(
    "A_parts, indices",
    [
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (None,),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (None, None, None),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (None, 1, None, 0, None),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), None, 1, None, 0, None),
        ),
        (
            (
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
                np.random.normal(size=(4, 3)),
            ),
            (slice(2, 3), None, 1, 0, None),
        ),
    ],
)
def test_expand_indices_newaxis(A_parts, indices):
    A = pt.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


def test_mixture_with_DiracDelta():
    srng = pt.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, name="X")
    Y_rv = dirac_delta(0.0)
    Y_rv.name = "Y"

    I_rv = srng.categorical([0.5, 0.5], size=1)

    i_vv = I_rv.clone()
    i_vv.name = "i"

    M_rv = pt.stack([X_rv, Y_rv])[I_rv]
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    logp_res = conditional_logp({M_rv: m_vv, I_rv: i_vv})

    assert m_vv in logp_res


def test_scalar_switch_mixture():
    srng = pt.random.RandomStream(29833)

    X_rv = srng.normal(-10.0, 0.1, name="X")
    Y_rv = srng.normal(10.0, 0.1, name="Y")

    I_rv = srng.bernoulli(0.5, name="I")
    i_vv = I_rv.clone()
    i_vv.name = "i"

    # When I_rv == True, X_rv flows through otherwise Y_rv does
    Z1_rv = pt.switch(I_rv, X_rv, Y_rv)
    Z1_rv.name = "Z1"

    assert Z1_rv.eval({I_rv: 0}) > 5
    assert Z1_rv.eval({I_rv: 1}) < -5

    z_vv = Z1_rv.clone()
    z_vv.name = "z1"

    fgraph, _, _ = construct_ir_fgraph({Z1_rv: z_vv, I_rv: i_vv})
    assert isinstance(fgraph.outputs[0].owner.op, MeasurableSwitchMixture)

    # building the identical graph but with a stack to check that mixture logps are identical
    Z2_rv = pt.stack((Y_rv, X_rv))[I_rv]

    assert Z2_rv.eval({I_rv: 0}) > 5
    assert Z2_rv.eval({I_rv: 1}) < -5

    z1_logp = conditional_logp({Z1_rv: z_vv, I_rv: i_vv})
    z2_logp = conditional_logp({Z2_rv: z_vv, I_rv: i_vv})
    z1_logp_combined = pt.sum([pt.sum(factor) for factor in z1_logp.values()])
    z2_logp_combined = pt.sum([pt.sum(factor) for factor in z2_logp.values()])
    np.testing.assert_almost_equal(0.69049938, z1_logp_combined.eval({z_vv: -10, i_vv: 1}))
    np.testing.assert_almost_equal(0.69049938, z2_logp_combined.eval({z_vv: -10, i_vv: 1}))


@pytest.mark.parametrize("switch_cond_scalar", (True, False))
def test_switch_mixture_vector(switch_cond_scalar):
    if switch_cond_scalar:
        switch_cond = pt.scalar("switch_cond", dtype=bool)
    else:
        switch_cond = pt.vector("switch_cond", dtype=bool)
    true_branch = pt.exp(pt.random.normal(size=(4,)))
    false_branch = pt.abs(pt.random.normal(size=(4,)))

    switch = pt.switch(switch_cond, true_branch, false_branch)
    switch.name = "switch_mix"
    switch_value = switch.clone()
    switch_logp = logp(switch, switch_value)

    if switch_cond_scalar:
        test_switch_cond = np.array(0, dtype=bool)
    else:
        test_switch_cond = np.array([0, 1, 0, 1], dtype=bool)
    test_switch_value = np.linspace(0.1, 2.5, 4)
    np.testing.assert_allclose(
        switch_logp.eval({switch_cond: test_switch_cond, switch_value: test_switch_value}),
        np.where(
            test_switch_cond,
            logp(true_branch, test_switch_value).eval(),
            logp(false_branch, test_switch_value).eval(),
        ),
    )


def test_switch_mixture_measurable_cond_fails():
    """Test that logprob inference fails when the switch condition is an unvalued measurable variable.

    Otherwise, the logp function would have to marginalize over this variable.

    NOTE: This could be supported in the future, in which case this test can be removed/adapted
    """
    cond_var = 1 - pt.random.bernoulli(p=0.5)
    true_branch = pt.random.normal()
    false_branch = pt.random.normal()

    switch = pt.switch(cond_var, true_branch, false_branch)
    with pytest.raises(NotImplementedError, match="Logprob method not implemented for"):
        logp(switch, switch.type())


def test_switch_mixture_invalid_bcast():
    """Test that we don't mark switches where components are broadcasted as measurable"""
    valid_switch_cond = pt.vector("switch_cond", dtype=bool)
    invalid_switch_cond = pt.matrix("switch_cond", dtype=bool)

    valid_true_branch = pt.exp(pt.random.normal(size=(4,)))
    valid_false_branch = pt.abs(pt.random.normal(size=(4,)))
    invalid_false_branch = pt.abs(pt.random.normal(size=()))

    valid_mix = pt.switch(valid_switch_cond, valid_true_branch, valid_false_branch)
    fgraph, _, _ = construct_ir_fgraph({valid_mix: valid_mix.type()})
    assert isinstance(fgraph.outputs[0].owner.op, MeasurableVariable)
    assert isinstance(fgraph.outputs[0].owner.op, MeasurableSwitchMixture)

    invalid_mix = pt.switch(invalid_switch_cond, valid_true_branch, valid_false_branch)
    fgraph, _, _ = construct_ir_fgraph({invalid_mix: invalid_mix.type()})
    assert not isinstance(fgraph.outputs[0].owner.op, MeasurableVariable)

    invalid_mix = pt.switch(valid_switch_cond, valid_true_branch, invalid_false_branch)
    fgraph, _, _ = construct_ir_fgraph({invalid_mix: invalid_mix.type()})
    assert not isinstance(fgraph.outputs[0].owner.op, MeasurableVariable)


def test_ifelse_mixture_one_component():
    if_rv = pt.random.bernoulli(0.5, name="if")
    scale_rv = pt.random.halfnormal(name="scale")
    comp_then = pt.random.normal(0, scale_rv, size=(2,), name="comp_then")
    comp_else = pt.random.halfnormal(0, scale_rv, size=(4,), name="comp_else")
    mix_rv = ifelse(if_rv, comp_then, comp_else, name="mix")

    if_vv = if_rv.clone()
    scale_vv = scale_rv.clone()
    mix_vv = mix_rv.clone()
    mix_logp = conditional_logp({if_rv: if_vv, scale_rv: scale_vv, mix_rv: mix_vv})[mix_vv]
    assert_no_rvs(mix_logp)

    fn = function([if_vv, scale_vv, mix_vv], mix_logp)
    scale_vv_test = 0.75
    mix_vv_test = np.r_[1.0, 2.5]
    np.testing.assert_array_almost_equal(
        fn(1, scale_vv_test, mix_vv_test),
        sp.norm(0, scale_vv_test).logpdf(mix_vv_test),
    )
    mix_vv_test = np.r_[1.0, 2.5, 3.5, 4.0]
    np.testing.assert_array_almost_equal(
        fn(0, scale_vv_test, mix_vv_test), sp.halfnorm(0, scale_vv_test).logpdf(mix_vv_test)
    )


def test_ifelse_mixture_multiple_components():
    rng = np.random.default_rng(968)

    if_var = pt.scalar("if_var", dtype="bool")
    comp_then1 = pt.random.normal(size=(2,), name="comp_true1")
    comp_then2 = comp_then1 + pt.random.normal(size=(2, 2), name="comp_then2")
    comp_else1 = pt.random.halfnormal(size=(4,), name="comp_else1")
    comp_else2 = pt.random.halfnormal(size=(4, 4), name="comp_else2")

    mix_rv1, mix_rv2 = ifelse(
        if_var, [comp_then1, comp_then2], [comp_else1, comp_else2], name="mix"
    )
    mix_vv1 = mix_rv1.clone()
    mix_vv2 = mix_rv2.clone()
    mix_logp1, mix_logp2 = conditional_logp({mix_rv1: mix_vv1, mix_rv2: mix_vv2}).values()
    assert_no_rvs(mix_logp1)
    assert_no_rvs(mix_logp2)

    fn = function([if_var, mix_vv1, mix_vv2], mix_logp1.sum() + mix_logp2.sum())
    mix_vv1_test = np.abs(rng.normal(size=(2,)))
    mix_vv2_test = np.abs(rng.normal(size=(2, 2)))
    np.testing.assert_almost_equal(
        fn(True, mix_vv1_test, mix_vv2_test),
        sp.norm(0, 1).logpdf(mix_vv1_test).sum()
        + sp.norm(mix_vv1_test, 1).logpdf(mix_vv2_test).sum(),
    )
    mix_vv1_test = np.abs(rng.normal(size=(4,)))
    mix_vv2_test = np.abs(rng.normal(size=(4, 4)))
    np.testing.assert_almost_equal(
        fn(False, mix_vv1_test, mix_vv2_test),
        sp.halfnorm(0, 1).logpdf(mix_vv1_test).sum() + sp.halfnorm(0, 1).logpdf(mix_vv2_test).sum(),
    )


def test_ifelse_mixture_shared_component():
    rng = np.random.default_rng(1009)

    if_var = pt.scalar("if_var", dtype="bool")
    outer_rv = pt.random.normal(name="outer")
    # comp_shared need not be an output of ifelse at all,
    # but since we allow arbitrary graphs we test it works as expected.
    comp_shared = pt.random.normal(size=(2,), name="comp_shared")
    comp_then = outer_rv + pt.random.normal(comp_shared, 1, size=(4, 2), name="comp_then")
    comp_else = outer_rv + pt.random.normal(comp_shared, 10, size=(8, 2), name="comp_else")
    shared_rv, mix_rv = ifelse(
        if_var, [comp_shared, comp_then], [comp_shared, comp_else], name="mix"
    )

    outer_vv = outer_rv.clone()
    shared_vv = shared_rv.clone()
    mix_vv = mix_rv.clone()
    outer_logp, mix_logp1, mix_logp2 = conditional_logp(
        {outer_rv: outer_vv, shared_rv: shared_vv, mix_rv: mix_vv}
    ).values()
    assert_no_rvs(outer_logp)
    assert_no_rvs(mix_logp1)
    assert_no_rvs(mix_logp2)

    fn = function([if_var, outer_vv, shared_vv, mix_vv], mix_logp1.sum() + mix_logp2.sum())
    outer_vv_test = rng.normal()
    shared_vv_test = rng.normal(size=(2,))
    mix_vv_test = rng.normal(size=(4, 2))
    np.testing.assert_almost_equal(
        fn(True, outer_vv_test, shared_vv_test, mix_vv_test),
        (
            sp.norm(0, 1).logpdf(shared_vv_test).sum()
            + sp.norm(outer_vv_test + shared_vv_test, 1).logpdf(mix_vv_test).sum()
        ),
    )
    mix_vv_test = rng.normal(size=(8, 2))
    np.testing.assert_almost_equal(
        fn(False, outer_vv_test, shared_vv_test, mix_vv_test),
        (
            sp.norm(0, 1).logpdf(shared_vv_test).sum()
            + sp.norm(outer_vv_test + shared_vv_test, 10).logpdf(mix_vv_test).sum()
        ),
        decimal=6,
    )


@pytest.mark.xfail(reason="Relied on rewrite-case that is no longer supported by PyTensor")
def test_joint_logprob_subtensor():
    """Make sure we can compute a joint log-probability for ``Y[I]`` where ``Y`` and ``I`` are random variables."""

    size = 5

    mu_base = np.power(10, np.arange(np.prod(size))).reshape(size)
    mu = np.stack([mu_base, -mu_base])
    sigma = 0.001
    rng = pytensor.shared(np.random.RandomState(232), borrow=True)

    A_rv = pt.random.normal(mu, sigma, rng=rng)
    A_rv.name = "A"

    p = 0.5

    I_rv = pt.random.bernoulli(p, size=size, rng=rng)
    I_rv.name = "I"

    # The rewrite for lifting subtensored RVs refuses to work with advanced
    # indexing as it could lead to repeated draws.
    # TODO: Re-enable rewrite for cases where this is not a concern
    #  (e.g., at least one of the advanced indexes has non-repeating values)
    A_idx = A_rv[I_rv, pt.ogrid[A_rv.shape[-1] :]]

    assert isinstance(A_idx.owner.op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1))

    A_idx_value_var = A_idx.type()
    A_idx_value_var.name = "A_idx_value"

    I_value_var = I_rv.type()
    I_value_var.name = "I_value"

    A_idx_logp = conditional_logp({A_idx: A_idx_value_var, I_rv: I_value_var})
    A_idx_logp_comb = pt.add(*A_idx_logp.values())

    logp_vals_fn = pytensor.function([A_idx_value_var, I_value_var], A_idx_logp_comb)

    # The compiled graph should not contain any `RandomVariables`
    assert_no_rvs(logp_vals_fn.maker.fgraph.outputs[0])

    decimals = 6 if pytensor.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    for i in range(10):
        bern_sp = sp.bernoulli(p)
        I_value = bern_sp.rvs(size=size, random_state=test_val_rng).astype(I_rv.dtype)

        norm_sp = sp.norm(mu[I_value, np.ogrid[mu.shape[1] :]], sigma)
        A_idx_value = norm_sp.rvs(random_state=test_val_rng).astype(A_idx.dtype)

        exp_obs_logps = norm_sp.logpdf(A_idx_value)
        exp_obs_logps += bern_sp.logpmf(I_value)

        logp_vals = logp_vals_fn(A_idx_value, I_value)

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)


def test_nested_ifelse():
    idx = pt.scalar("idx", dtype=int)

    dist0 = pt.random.normal(-5, 1)
    dist1 = pt.random.normal(0, 1)
    dist2 = pt.random.normal(5, 1)
    mix = ifelse(pt.eq(idx, 0), dist0, ifelse(pt.eq(idx, 1), dist1, dist2))
    mix.name = "mix"

    value = mix.clone()
    mix_logp = logp(mix, value)
    assert mix_logp.name == "mix_logprob"
    mix_logp_fn = pytensor.function([idx, value], mix_logp)

    test_value = 0.25
    np.testing.assert_almost_equal(mix_logp_fn(0, test_value), sp.norm.logpdf(test_value, -5, 1))
    np.testing.assert_almost_equal(mix_logp_fn(1, test_value), sp.norm.logpdf(test_value, 0, 1))
    np.testing.assert_almost_equal(mix_logp_fn(2, test_value), sp.norm.logpdf(test_value, 5, 1))
