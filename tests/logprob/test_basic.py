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

import warnings

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.stats.distributions as sp

from pytensor.graph.basic import ancestors, equal_computations
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)

import pymc as pm

from pymc.logprob.basic import factorized_joint_logprob, icdf, joint_logp, logcdf, logp
from pymc.logprob.transforms import LogTransform
from pymc.logprob.utils import rvs_to_value_vars, walk_model
from pymc.pytensorf import replace_rvs_by_values
from pymc.testing import assert_no_rvs
from tests.logprob.utils import joint_logprob


def test_joint_logprob_basic():
    # A simple check for when `joint_logprob` is the same as `logprob`
    a = pt.random.uniform(0.0, 1.0)
    a.name = "a"
    a_value_var = a.clone()

    a_logp = joint_logprob({a: a_value_var}, sum=False)
    a_logp_exp = logp(a, a_value_var)

    assert equal_computations([a_logp], [a_logp_exp])

    # Let's try a hierarchical model
    sigma = pt.random.invgamma(0.5, 0.5)
    Y = pt.random.normal(0.0, sigma)

    sigma_value_var = sigma.clone()
    y_value_var = Y.clone()

    total_ll = joint_logprob({Y: y_value_var, sigma: sigma_value_var}, sum=False)

    # We need to replace the reference to `sigma` in `Y` with its value
    # variable
    ll_Y = logp(Y, y_value_var)
    (ll_Y,), _ = rvs_to_value_vars(
        [ll_Y],
        initial_replacements={sigma: sigma_value_var},
    )
    total_ll_exp = logp(sigma, sigma_value_var) + ll_Y

    assert equal_computations([total_ll], [total_ll_exp])

    # Now, make sure we can compute a joint log-probability for a hierarchical
    # model with some non-`RandomVariable` nodes
    c = pt.random.normal()
    c.name = "c"
    b_l = c * a + 2.0
    b = pt.random.uniform(b_l, b_l + 1.0)
    b.name = "b"

    b_value_var = b.clone()
    c_value_var = c.clone()

    b_logp = joint_logprob({a: a_value_var, b: b_value_var, c: c_value_var})

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert_no_rvs(b_logp)

    res_ancestors = list(walk_model((b_logp,), walk_past_rvs=True))
    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors
    assert a_value_var in res_ancestors


def test_joint_logprob_multi_obs():
    a = pt.random.uniform(0.0, 1.0)
    b = pt.random.normal(0.0, 1.0)

    a_val = a.clone()
    b_val = b.clone()

    logp_res = joint_logprob({a: a_val, b: b_val}, sum=False)
    logp_exp = logp(a, a_val) + logp(b, b_val)

    assert equal_computations([logp_res], [logp_exp])

    x = pt.random.normal(0, 1)
    y = pt.random.normal(x, 1)

    x_val = x.clone()
    y_val = y.clone()

    logp_res = joint_logprob({x: x_val, y: y_val})
    exp_logp = joint_logprob({x: x_val, y: y_val})

    assert equal_computations([logp_res], [exp_logp])


def test_joint_logprob_diff_dims():
    M = pt.matrix("M")
    x = pt.random.normal(0, 1, size=M.shape[1], name="X")
    y = pt.random.normal(M.dot(x), 1, name="Y")

    x_vv = x.clone()
    x_vv.name = "x"
    y_vv = y.clone()
    y_vv.name = "y"

    logp = joint_logprob({x: x_vv, y: y_vv})

    M_val = np.random.normal(size=(10, 3))
    x_val = np.random.normal(size=(3,))
    y_val = np.random.normal(size=(10,))

    point = {M: M_val, x_vv: x_val, y_vv: y_val}
    logp_val = logp.eval(point)

    exp_logp_val = (
        sp.norm.logpdf(x_val, 0, 1).sum() + sp.norm.logpdf(y_val, M_val.dot(x_val), 1).sum()
    )
    assert exp_logp_val == pytest.approx(logp_val)


def test_incsubtensor_original_values_output_dict():
    """
    Test that the original un-incsubtensor value variable appears an the key of
    the logprob factor
    """

    base_rv = pt.random.normal(0, 1, size=2)
    rv = pt.set_subtensor(base_rv[0], 5)
    vv = rv.clone()

    logp_dict = factorized_joint_logprob({rv: vv})
    assert vv in logp_dict


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

    A_idx = A_rv[I_rv, pt.ogrid[A_rv.shape[-1] :]]

    assert isinstance(A_idx.owner.op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1))

    A_idx_value_var = A_idx.type()
    A_idx_value_var.name = "A_idx_value"

    I_value_var = I_rv.type()
    I_value_var.name = "I_value"

    A_idx_logp = joint_logprob({A_idx: A_idx_value_var, I_rv: I_value_var}, sum=False)

    logp_vals_fn = pytensor.function([A_idx_value_var, I_value_var], A_idx_logp)

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


def test_persist_inputs():
    """Make sure we don't unnecessarily clone variables."""
    x = pt.scalar("x")
    beta_rv = pt.random.normal(0, 1, name="beta")
    Y_rv = pt.random.normal(beta_rv * x, 1, name="y")

    beta_vv = beta_rv.type()
    y_vv = Y_rv.clone()

    logp = joint_logprob({beta_rv: beta_vv, Y_rv: y_vv})

    assert x in ancestors([logp])

    # Make sure we don't clone value variables when they're graphs.
    y_vv_2 = y_vv * 2
    logp_2 = joint_logprob({beta_rv: beta_vv, Y_rv: y_vv_2})

    assert y_vv in ancestors([logp_2])
    assert y_vv_2 in ancestors([logp_2])

    # Even when they are random
    y_vv = pt.random.normal(name="y_vv2")
    y_vv_2 = y_vv * 2
    logp_2 = joint_logprob({beta_rv: beta_vv, Y_rv: y_vv_2})

    assert y_vv in ancestors([logp_2])
    assert y_vv_2 in ancestors([logp_2])


def test_warn_random_found_factorized_joint_logprob():
    x_rv = pt.random.normal(name="x")
    y_rv = pt.random.normal(x_rv, 1, name="y")

    y_vv = y_rv.clone()

    with pytest.warns(UserWarning, match="Found a random variable that was neither among"):
        factorized_joint_logprob({y_rv: y_vv})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factorized_joint_logprob({y_rv: y_vv}, warn_missing_rvs=False)


def test_multiple_rvs_to_same_value_raises():
    x_rv1 = pt.random.normal(name="x1")
    x_rv2 = pt.random.normal(name="x2")
    x = x_rv1.type()
    x.name = "x"

    msg = "More than one logprob factor was assigned to the value var x"
    with pytest.raises(ValueError, match=msg):
        joint_logprob({x_rv1: x, x_rv2: x})


def test_joint_logp_basic():
    """Make sure we can compute a log-likelihood for a hierarchical model with transforms."""

    with pm.Model() as m:
        a = pm.Uniform("a", 0.0, 1.0)
        c = pm.Normal("c")
        b_l = c * a + 2.0
        b = pm.Uniform("b", b_l, b_l + 1.0)

    a_value_var = m.rvs_to_values[a]
    assert m.rvs_to_transforms[a]

    b_value_var = m.rvs_to_values[b]
    assert m.rvs_to_transforms[b]

    c_value_var = m.rvs_to_values[c]

    (b_logp,) = joint_logp(
        (b,),
        rvs_to_values=m.rvs_to_values,
        rvs_to_transforms=m.rvs_to_transforms,
    )

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert_no_rvs(b_logp)

    res_ancestors = list(walk_model((b_logp,)))
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
def test_joint_logp_incsubtensor(indices, size):
    """Make sure we can compute a log-likelihood for ``Y[idx] = data`` where ``Y`` is univariate."""

    mu = pm.floatX(np.power(10, np.arange(np.prod(size)))).reshape(size)
    data = mu[indices]
    sigma = 0.001
    rng = np.random.RandomState(232)
    a_val = rng.normal(mu, sigma, size=size).astype(pytensor.config.floatX)

    rng = pytensor.shared(rng, borrow=False)
    a = pm.Normal.dist(mu, sigma, size=size, rng=rng)
    a_value_var = a.type()
    a.name = "a"

    a_idx = pt.set_subtensor(a[indices], data)

    assert isinstance(a_idx.owner.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1))

    a_idx_value_var = a_idx.type()
    a_idx_value_var.name = "a_idx_value"

    a_idx_logp = joint_logp(
        (a_idx,),
        rvs_to_values={a_idx: a_value_var},
        rvs_to_transforms={},
    )

    logp_vals = a_idx_logp[0].eval({a_value_var: a_val})

    # The indices that were set should all have the same log-likelihood values,
    # because the values they were set to correspond to the unique means along
    # that dimension.  This helps us confirm that the log-likelihood is
    # associating the assigned values with their correct parameters.
    a_val_idx = a_val.copy()
    a_val_idx[indices] = data
    exp_obs_logps = sp.norm.logpdf(a_val_idx, mu, sigma)
    np.testing.assert_almost_equal(logp_vals, exp_obs_logps)


def test_model_unchanged_logprob_access():
    # Issue #5007
    with pm.Model() as model:
        a = pm.Normal("a")
        c = pm.Uniform("c", lower=a - 1, upper=1)

    original_inputs = set(pytensor.graph.graph_inputs([c]))
    # Extract model.logp
    model.logp()
    new_inputs = set(pytensor.graph.graph_inputs([c]))
    assert original_inputs == new_inputs


def test_unexpected_rvs():
    with pm.Model() as model:
        x = pm.Normal("x")
        y = pm.CustomDist("y", logp=lambda *args: x)

    with pytest.raises(ValueError, match="^Random variables detected in the logp graph"):
        model.logp()


def test_hierarchical_logp():
    """Make sure there are no random variables in a model's log-likelihood graph."""
    with pm.Model() as m:
        x = pm.Uniform("x", lower=0, upper=1)
        y = pm.Uniform("y", lower=0, upper=x)

    logp_ancestors = list(ancestors([m.logp()]))
    ops = {a.owner.op for a in logp_ancestors if a.owner}
    assert len(ops) > 0
    assert not any(isinstance(o, RandomVariable) for o in ops)
    assert m.rvs_to_values[x] in logp_ancestors
    assert m.rvs_to_values[y] in logp_ancestors


def test_hierarchical_obs_logp():
    obs = np.array([0.5, 0.4, 5, 2])

    with pm.Model() as model:
        x = pm.Uniform("x", 0, 1, observed=obs)
        pm.Uniform("y", x, 2, observed=obs)

    logp_ancestors = list(ancestors([model.logp()]))
    ops = {a.owner.op for a in logp_ancestors if a.owner}
    assert len(ops) > 0
    assert not any(isinstance(o, RandomVariable) for o in ops)


@pytest.mark.parametrize(
    "func, scipy_func",
    [
        (logp, "logpdf"),
        (logcdf, "logcdf"),
        (icdf, "ppf"),
    ],
)
def test_probability_direct_dispatch(func, scipy_func):
    value = pt.vector("value")
    x = pm.Normal.dist(0, 1)

    np.testing.assert_almost_equal(
        func(x, value).eval({value: [0, 1]}),
        getattr(sp.norm(0, 1), scipy_func)([0, 1]),
    )

    np.testing.assert_almost_equal(
        func(x, [0, 1]).eval(),
        getattr(sp.norm(0, 1), scipy_func)([0, 1]),
    )


@pytest.mark.parametrize(
    "func, scipy_func, test_value",
    [
        (logp, "logpdf", 5.0),
        (logcdf, "logcdf", 5.0),
        (icdf, "ppf", 0.7),
    ],
)
def test_probability_inference(func, scipy_func, test_value):
    assert np.isclose(
        func(pt.exp(pm.Normal.dist()), test_value).eval(),
        getattr(sp.lognorm(s=1), scipy_func)(test_value),
    )


@pytest.mark.parametrize(
    "func, func_name",
    [
        (logp, "Logprob"),
        (logcdf, "LogCDF"),
        (icdf, "Inverse CDF"),
    ],
)
def test_probability_inference_fails(func, func_name):
    with pytest.raises(
        NotImplementedError,
        match=f"{func_name} method not implemented for Elemwise{{cos,no_inplace}}",
    ):
        func(pt.cos(pm.Normal.dist()), 1)


@pytest.mark.parametrize(
    "func, scipy_func, test_value",
    [
        (logp, "logpdf", 5.0),
        (logcdf, "logcdf", 5.0),
        (icdf, "ppf", 0.7),
    ],
)
def test_warn_random_found_probability_inference(func, scipy_func, test_value):
    # Fail if unexpected warning is issued
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        input_rv = pm.Normal.dist(0, name="input")
        # Note: This graph could correspond to a convolution of two normals
        # In which case the inference should either return that or fail explicitly
        # For now, the lopgrob submodule treats the input as a stochastic value.
        rv = pt.exp(pm.Normal.dist(input_rv))
        with pytest.warns(UserWarning, match="RandomVariables were found in the derived graph"):
            assert func(rv, 0.0)

        res = func(rv, 0.0, warn_missing_rvs=False)
        # This is the problem we are warning about, as now we can no longer identify the original rv in the graph
        # or replace it by the respective value
        assert rv not in ancestors([res])

        # Test that the prescribed solution does not raise a warning and works as expected
        input_vv = input_rv.clone()
        [new_rv] = replace_rvs_by_values(
            [rv],
            rvs_to_values={input_rv: input_vv},
            rvs_to_transforms={input_rv: LogTransform()},
        )
        input_vv_test = 1.3
        np.testing.assert_almost_equal(
            func(new_rv, test_value).eval({input_vv: input_vv_test}),
            getattr(sp.lognorm(s=1, loc=0, scale=np.exp(np.exp(input_vv_test))), scipy_func)(
                test_value
            ),
        )
