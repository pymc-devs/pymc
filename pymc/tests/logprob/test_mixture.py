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

import numpy as np
import pytensor
import pytensor.tensor as at
import pytest
import scipy.stats.distributions as sp

from pytensor.graph.basic import Variable, equal_computations
from pytensor.tensor.random.basic import CategoricalRV
from pytensor.tensor.shape import shape_tuple
from pytensor.tensor.subtensor import as_index_constant

from pymc.logprob.joint_logprob import factorized_joint_logprob, joint_logprob
from pymc.logprob.mixture import MixtureRV, expand_indices
from pymc.logprob.rewriting import construct_ir_fgraph
from pymc.logprob.utils import dirac_delta
from pymc.tests.helpers import assert_no_rvs
from pymc.tests.logprob.utils import scipy_logprob


def test_mixture_basics():
    srng = at.random.RandomStream(29833)

    def create_mix_model(size, axis):
        X_rv = srng.normal(0, 1, size=size, name="X")
        Y_rv = srng.gamma(0.5, 0.5, size=size, name="Y")

        p_at = at.scalar("p")
        p_at.tag.test_value = 0.5

        I_rv = srng.bernoulli(p_at, size=size, name="I")
        i_vv = I_rv.clone()
        i_vv.name = "i"

        if isinstance(axis, Variable):
            M_rv = at.join(axis, X_rv, Y_rv)[I_rv]
        else:
            M_rv = at.stack([X_rv, Y_rv], axis=axis)[I_rv]

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
        factorized_joint_logprob({M_rv: m_vv, I_rv: i_vv, X_rv: x_vv})

    with pytest.raises(NotImplementedError):
        axis_at = at.lscalar("axis")
        axis_at.tag.test_value = 0
        env = create_mix_model((2,), axis_at)
        I_rv = env["I_rv"]
        i_vv = env["i_vv"]
        M_rv = env["M_rv"]
        m_vv = env["m_vv"]
        joint_logprob({M_rv: m_vv, I_rv: i_vv})


@pytensor.config.change_flags(compute_test_value="warn")
@pytest.mark.parametrize(
    "op_constructor",
    [
        lambda _I, _X, _Y: at.stack([_X, _Y])[_I],
        lambda _I, _X, _Y: at.switch(_I, _X, _Y),
    ],
)
def test_compute_test_value(op_constructor):

    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, name="X")
    Y_rv = srng.gamma(0.5, 0.5, name="Y")

    p_at = at.scalar("p")
    p_at.tag.test_value = 0.3

    I_rv = srng.bernoulli(p_at, name="I")

    i_vv = I_rv.clone()
    i_vv.name = "i"

    M_rv = op_constructor(I_rv, X_rv, Y_rv)
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    del M_rv.tag.test_value

    M_logp = joint_logprob({M_rv: m_vv, I_rv: i_vv}, sum=False)

    assert isinstance(M_logp.tag.test_value, np.ndarray)


@pytest.mark.parametrize(
    "p_val, size",
    [
        (np.array(0.0, dtype=pytensor.config.floatX), ()),
        (np.array(1.0, dtype=pytensor.config.floatX), ()),
        (np.array(0.0, dtype=pytensor.config.floatX), (2,)),
        (np.array(1.0, dtype=pytensor.config.floatX), (2, 1)),
        (np.array(1.0, dtype=pytensor.config.floatX), (2, 3)),
        (np.array([0.1, 0.9], dtype=pytensor.config.floatX), (2, 3)),
    ],
)
def test_hetero_mixture_binomial(p_val, size):
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, size=size, name="X")
    Y_rv = srng.gamma(0.5, 0.5, size=size, name="Y")

    if np.ndim(p_val) == 0:
        p_at = at.scalar("p")
        p_at.tag.test_value = p_val
        I_rv = srng.bernoulli(p_at, size=size, name="I")
        p_val_1 = p_val
    else:
        p_at = at.vector("p")
        p_at.tag.test_value = np.array(p_val, dtype=pytensor.config.floatX)
        I_rv = srng.categorical(p_at, size=size, name="I")
        p_val_1 = p_val[1]

    i_vv = I_rv.clone()
    i_vv.name = "i"

    M_rv = at.stack([X_rv, Y_rv])[I_rv]
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    M_logp = joint_logprob({M_rv: m_vv, I_rv: i_vv}, sum=False)

    M_logp_fn = pytensor.function([p_at, m_vv, i_vv], M_logp)

    assert_no_rvs(M_logp_fn.maker.fgraph.outputs[0])

    decimals = 6 if pytensor.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    bern_sp = sp.bernoulli(p_val_1)
    norm_sp = sp.norm(loc=0, scale=1)
    gamma_sp = sp.gamma(0.5, scale=1.0 / 0.5)

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
    "X_args, Y_args, Z_args, p_val, comp_size, idx_size, extra_indices, join_axis",
    [
        # Scalar mixture components, scalar index
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
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
        ),
        # Degenerate vector mixture components, scalar index
        (
            (
                np.array([0], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5], dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
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
        ),
        # Scalar mixture components, vector index
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
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
        ),
        (
            (
                np.array([0, -100], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5, 1], dtype=pytensor.config.floatX),
                np.array([0.5, 1], dtype=pytensor.config.floatX),
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
        ),
        (
            (
                np.array([0, -100], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5, 1], dtype=pytensor.config.floatX),
                np.array([0.5, 1], dtype=pytensor.config.floatX),
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
        ),
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (2,),
            (2,),
            (),
            0,
        ),
        # Same as before but with degenerate vector parameters
        (
            (
                np.array([0], dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array([0.5], dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
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
        ),
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
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
        ),
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
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
        ),
        pytest.param(
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
            ),
            (
                np.array(100, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            np.array([0.1, 0.5, 0.4], dtype=pytensor.config.floatX),
            (3,),
            (3,),
            (slice(None),),
            1,
            marks=pytest.mark.xfail(IndexError, reason="Bug in AdvancedIndex Mixture logprob"),
        ),
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
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
        ),
        (
            (
                np.array(0, dtype=pytensor.config.floatX),
                np.array(1, dtype=pytensor.config.floatX),
            ),
            (
                np.array(0.5, dtype=pytensor.config.floatX),
                np.array(0.5, dtype=pytensor.config.floatX),
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
        ),
    ],
)
def test_hetero_mixture_categorical(
    X_args, Y_args, Z_args, p_val, comp_size, idx_size, extra_indices, join_axis
):
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(*X_args, size=comp_size, name="X")
    Y_rv = srng.gamma(*Y_args, size=comp_size, name="Y")
    Z_rv = srng.normal(*Z_args, size=comp_size, name="Z")

    p_at = at.as_tensor(p_val).type()
    p_at.name = "p"
    p_at.tag.test_value = np.array(p_val, dtype=pytensor.config.floatX)
    I_rv = srng.categorical(p_at, size=idx_size, name="I")

    i_vv = I_rv.clone()
    i_vv.name = "i"

    indices_at = list(extra_indices)
    indices_at.insert(join_axis, I_rv)
    indices_at = tuple(indices_at)

    M_rv = at.stack([X_rv, Y_rv, Z_rv], axis=join_axis)[indices_at]
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    logp_parts = factorized_joint_logprob({M_rv: m_vv, I_rv: i_vv}, sum=False)

    I_logp_fn = pytensor.function([p_at, i_vv], logp_parts[i_vv])
    M_logp_fn = pytensor.function([m_vv, i_vv], logp_parts[m_vv])

    assert_no_rvs(I_logp_fn.maker.fgraph.outputs[0])
    assert_no_rvs(M_logp_fn.maker.fgraph.outputs[0])

    decimals = 6 if pytensor.config.floatX == "float64" else 4

    test_val_rng = np.random.RandomState(3238)

    norm_1_sp = sp.norm(loc=X_args[0], scale=X_args[1])
    gamma_sp = sp.gamma(Y_args[0], scale=1 / Y_args[1])
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
    A = at.stack(A_parts)
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
    A = at.stack(A_parts)
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
    A = at.stack(A_parts)
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
    A = at.stack(A_parts)
    at_indices = [as_index_constant(idx) for idx in indices]
    full_indices = expand_indices(at_indices, shape_tuple(A))
    assert len(full_indices) == A.ndim
    exp_res = A[indices].eval()
    res = A[full_indices].eval()
    assert np.array_equal(res, exp_res)


def test_mixture_with_DiracDelta():
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(0, 1, name="X")
    Y_rv = dirac_delta(0.0)
    Y_rv.name = "Y"

    I_rv = srng.categorical([0.5, 0.5], size=4)

    i_vv = I_rv.clone()
    i_vv.name = "i"

    M_rv = at.stack([X_rv, Y_rv])[I_rv]
    M_rv.name = "M"

    m_vv = M_rv.clone()
    m_vv.name = "m"

    logp_res = factorized_joint_logprob({M_rv: m_vv, I_rv: i_vv})

    assert m_vv in logp_res


def test_switch_mixture():
    srng = at.random.RandomStream(29833)

    X_rv = srng.normal(-10.0, 0.1, name="X")
    Y_rv = srng.normal(10.0, 0.1, name="Y")

    I_rv = srng.bernoulli(0.5, name="I")
    i_vv = I_rv.clone()
    i_vv.name = "i"

    Z1_rv = at.switch(I_rv, X_rv, Y_rv)
    z_vv = Z1_rv.clone()
    z_vv.name = "z1"

    fgraph, _, _ = construct_ir_fgraph({Z1_rv: z_vv, I_rv: i_vv})

    assert isinstance(fgraph.outputs[0].owner.op, MixtureRV)
    assert not hasattr(
        fgraph.outputs[0].tag, "test_value"
    )  # pytensor.config.compute_test_value == "off"
    assert fgraph.outputs[0].name is None

    Z1_rv.name = "Z1"

    fgraph, _, _ = construct_ir_fgraph({Z1_rv: z_vv, I_rv: i_vv})

    assert fgraph.outputs[0].name == "Z1-mixture"

    # building the identical graph but with a stack to check that mixture computations are identical

    Z2_rv = at.stack((X_rv, Y_rv))[I_rv]

    fgraph2, _, _ = construct_ir_fgraph({Z2_rv: z_vv, I_rv: i_vv})

    assert equal_computations(fgraph.outputs, fgraph2.outputs)

    z1_logp = joint_logprob({Z1_rv: z_vv, I_rv: i_vv})
    z2_logp = joint_logprob({Z2_rv: z_vv, I_rv: i_vv})

    # below should follow immediately from the equal_computations assertion above
    assert equal_computations([z1_logp], [z2_logp])

    np.testing.assert_almost_equal(0.69049938, z1_logp.eval({z_vv: -10, i_vv: 0}))
    np.testing.assert_almost_equal(0.69049938, z2_logp.eval({z_vv: -10, i_vv: 0}))
