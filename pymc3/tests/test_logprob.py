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
import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats.distributions as sp

from aesara.gradient import DisconnectedGrad
from aesara.graph.basic import Constant, ancestors, graph_inputs
from aesara.graph.fg import FunctionGraph
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)

from pymc3.aesaraf import floatX, walk_model
from pymc3.distributions.continuous import Normal, Uniform
from pymc3.distributions.discrete import Bernoulli
from pymc3.distributions.logprob import logcdf, logp, logpt
from pymc3.model import Model
from pymc3.tests.helpers import select_by_precision


def assert_no_rvs(var):
    assert not any(isinstance(v.owner.op, RandomVariable) for v in ancestors([var]) if v.owner)
    return var


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


@pytest.mark.parametrize(
    "indices, size",
    [
        (slice(0, 2), 5),
        (np.r_[True, True, False, False, True], 5),
        (np.r_[0, 1, 4], 5),
        ((np.array([0, 1, 4]), np.array([0, 1, 4])), (5, 5)),
    ],
)
def test_logpt_incsubtensor(indices, size):
    """Make sure we can compute a log-likelihood for ``Y[idx] = data`` where ``Y`` is univariate."""

    mu = floatX(np.power(10, np.arange(np.prod(size)))).reshape(size)
    data = mu[indices]
    sigma = 0.001
    rng = aesara.shared(np.random.RandomState(232), borrow=True)

    a = Normal.dist(mu, sigma, size=size, rng=rng)
    a.name = "a"

    a_idx = at.set_subtensor(a[indices], data)

    assert isinstance(a_idx.owner.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1))

    a_idx_value_var = a_idx.type()
    a_idx_value_var.name = "a_idx_value"

    a_idx_logp = logpt(a_idx, a_idx_value_var)

    logp_vals = a_idx_logp.eval()

    # The indices that were set should all have the same log-likelihood values,
    # because the values they were set to correspond to the unique means along
    # that dimension.  This helps us confirm that the log-likelihood is
    # associating the assigned values with their correct parameters.
    exp_obs_logps = sp.norm.logpdf(mu, mu, sigma)[indices]
    np.testing.assert_almost_equal(logp_vals[indices], exp_obs_logps)

    # Next, we need to confirm that the unset indices are being sampled
    # from the original random variable in the correct locations.
    # rng.get_value(borrow=True).seed(232)

    res_ancestors = list(walk_model((a_idx_logp,), walk_past_rvs=True))
    res_rv_ancestors = tuple(
        v for v in res_ancestors if v.owner and isinstance(v.owner.op, RandomVariable)
    )

    # The imputed missing values are drawn from the original distribution
    (a_new,) = res_rv_ancestors
    assert a_new is not a
    assert a_new.owner.op == a.owner.op

    fg = FunctionGraph(
        [v for v in graph_inputs((a_idx_logp,)) if not isinstance(v, Constant)],
        [a_idx_logp],
        clone=False,
    )

    ((a_client, _),) = fg.clients[a_new]
    # The imputed values should be treated as constants when gradients are
    # taken
    assert isinstance(a_client.op, DisconnectedGrad)

    ((a_client, _),) = fg.clients[a_client.outputs[0]]
    assert isinstance(a_client.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1))
    indices = tuple(i.eval() for i in a_client.inputs[2:])
    np.testing.assert_almost_equal(indices, indices)


def test_logpt_subtensor():
    """Make sure we can compute a log-likelihood for ``Y[I]`` where ``Y`` and ``I`` are random variables."""

    size = 5

    mu_base = floatX(np.power(10, np.arange(np.prod(size)))).reshape(size)
    mu = np.stack([mu_base, -mu_base])
    sigma = 0.001
    rng = aesara.shared(np.random.RandomState(232), borrow=True)

    A_rv = Normal.dist(mu, sigma, rng=rng)
    A_rv.name = "A"

    p = 0.5

    I_rv = Bernoulli.dist(p, size=size, rng=rng)
    I_rv.name = "I"

    A_idx = A_rv[I_rv, at.ogrid[A_rv.shape[-1] :]]

    assert isinstance(A_idx.owner.op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1))

    A_idx_value_var = A_idx.type()
    A_idx_value_var.name = "A_idx_value"

    I_value_var = I_rv.type()
    I_value_var.name = "I_value"

    A_idx_logp = logpt(A_idx, {A_idx: A_idx_value_var, I_rv: I_value_var})

    logp_vals_fn = aesara.function([A_idx_value_var, I_value_var], A_idx_logp)

    # The compiled graph should not contain any `RandomVariables`
    assert_no_rvs(logp_vals_fn.maker.fgraph.outputs[0])

    decimals = select_by_precision(float64=6, float32=4)

    for i in range(10):
        bern_sp = sp.bernoulli(p)
        I_value = bern_sp.rvs(size=size).astype(I_rv.dtype)

        norm_sp = sp.norm(mu[I_value, np.ogrid[mu.shape[1] :]], sigma)
        A_idx_value = norm_sp.rvs().astype(A_idx.dtype)

        exp_obs_logps = norm_sp.logpdf(A_idx_value)
        exp_obs_logps += bern_sp.logpmf(I_value)

        logp_vals = logp_vals_fn(A_idx_value, I_value)

        np.testing.assert_almost_equal(logp_vals, exp_obs_logps, decimal=decimals)


def test_logp_helper():
    value = at.vector("value")
    x = Normal.dist(0, 1, size=2)

    x_logp = logp(x, value)
    np.testing.assert_almost_equal(x_logp.eval({value: [0, 1]}), sp.norm(0, 1).logpdf([0, 1]))

    x_logp = logp(x, [0, 1])
    np.testing.assert_almost_equal(x_logp.eval(), sp.norm(0, 1).logpdf([0, 1]))


def test_logcdf_helper():
    value = at.vector("value")
    x = Normal.dist(0, 1, size=2)

    x_logp = logcdf(x, value)
    np.testing.assert_almost_equal(x_logp.eval({value: [0, 1]}), sp.norm(0, 1).logcdf([0, 1]))

    x_logp = logcdf(x, [0, 1])
    np.testing.assert_almost_equal(x_logp.eval(), sp.norm(0, 1).logcdf([0, 1]))
