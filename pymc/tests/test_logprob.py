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
import warnings

import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats.distributions as sp

from aeppl.abstract import get_measurable_outputs
from aesara.graph.basic import ancestors
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)

from pymc import DensityDist
from pymc.aesaraf import floatX, walk_model
from pymc.distributions.continuous import (
    HalfFlat,
    LogNormal,
    Normal,
    TruncatedNormal,
    Uniform,
)
from pymc.distributions.discrete import Bernoulli
from pymc.distributions.logprob import (
    _get_scaling,
    ignore_logprob,
    joint_logp,
    joint_logpt,
    logcdf,
    logp,
)
from pymc.model import Model, Potential
from pymc.tests.helpers import select_by_precision


def assert_no_rvs(var):
    assert not any(isinstance(v.owner.op, RandomVariable) for v in ancestors([var]) if v.owner)
    return var


def test_get_scaling():

    assert _get_scaling(None, (2, 3), 2).eval() == 1
    # ndim >=1 & ndim<1
    assert _get_scaling(45, (2, 3), 1).eval() == 22.5
    assert _get_scaling(45, (2, 3), 0).eval() == 45

    # list or tuple tests
    # total_size contains other than Ellipsis, None and Int
    with pytest.raises(TypeError, match="Unrecognized `total_size` type"):
        _get_scaling([2, 4, 5, 9, 11.5], (2, 3), 2)
    # check with Ellipsis
    with pytest.raises(ValueError, match="Double Ellipsis in `total_size` is restricted"):
        _get_scaling([1, 2, 5, Ellipsis, Ellipsis], (2, 3), 2)
    with pytest.raises(
        ValueError,
        match="Length of `total_size` is too big, number of scalings is bigger that ndim",
    ):
        _get_scaling([1, 2, 5, Ellipsis], (2, 3), 2)

    assert _get_scaling([Ellipsis], (2, 3), 2).eval() == 1

    assert _get_scaling([4, 5, 9, Ellipsis, 32, 12], (2, 3, 2), 5).eval() == 960
    assert _get_scaling([4, 5, 9, Ellipsis], (2, 3, 2), 5).eval() == 15
    # total_size with no Ellipsis (end = [ ])
    with pytest.raises(
        ValueError,
        match="Length of `total_size` is too big, number of scalings is bigger that ndim",
    ):
        _get_scaling([1, 2, 5], (2, 3), 2)

    assert _get_scaling([], (2, 3), 2).eval() == 1
    assert _get_scaling((), (2, 3), 2).eval() == 1
    # total_size invalid type
    with pytest.raises(
        TypeError,
        match="Unrecognized `total_size` type, expected int or list of ints, got {1, 2, 5}",
    ):
        _get_scaling({1, 2, 5}, (2, 3), 2)

    # test with rvar from model graph
    with Model() as m2:
        rv_var = Uniform("a", 0.0, 1.0)
    total_size = []
    assert _get_scaling(total_size, shape=rv_var.shape, ndim=rv_var.ndim).eval() == 1.0


def test_joint_logp_basic():
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

    b_logp = joint_logp(b, b_value_var, sum=False)

    with pytest.warns(FutureWarning):
        b_logpt = joint_logpt(b, b_value_var, sum=False)

    res_ancestors = list(walk_model(b_logp, walk_past_rvs=True))
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
def test_joint_logp_incsubtensor(indices, size):
    """Make sure we can compute a log-likelihood for ``Y[idx] = data`` where ``Y`` is univariate."""

    mu = floatX(np.power(10, np.arange(np.prod(size)))).reshape(size)
    data = mu[indices]
    sigma = 0.001
    rng = np.random.RandomState(232)
    a_val = rng.normal(mu, sigma, size=size).astype(aesara.config.floatX)

    rng = aesara.shared(rng, borrow=False)
    a = Normal.dist(mu, sigma, size=size, rng=rng)
    a_value_var = a.type()
    a.name = "a"

    a_idx = at.set_subtensor(a[indices], data)

    assert isinstance(a_idx.owner.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1))

    a_idx_value_var = a_idx.type()
    a_idx_value_var.name = "a_idx_value"

    a_idx_logp = joint_logp(a_idx, {a_idx: a_value_var}, sum=False)

    logp_vals = a_idx_logp[0].eval({a_value_var: a_val})

    # The indices that were set should all have the same log-likelihood values,
    # because the values they were set to correspond to the unique means along
    # that dimension.  This helps us confirm that the log-likelihood is
    # associating the assigned values with their correct parameters.
    a_val_idx = a_val.copy()
    a_val_idx[indices] = data
    exp_obs_logps = sp.norm.logpdf(a_val_idx, mu, sigma)
    np.testing.assert_almost_equal(logp_vals, exp_obs_logps)


def test_joint_logp_subtensor():
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

    A_idx_logps = joint_logp(A_idx, {A_idx: A_idx_value_var, I_rv: I_value_var}, sum=False)
    A_idx_logp = at.add(*A_idx_logps)

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
    x = Normal.dist(0, 1)

    x_logp = logp(x, value)
    np.testing.assert_almost_equal(x_logp.eval({value: [0, 1]}), sp.norm(0, 1).logpdf([0, 1]))

    x_logp = logp(x, [0, 1])
    np.testing.assert_almost_equal(x_logp.eval(), sp.norm(0, 1).logpdf([0, 1]))


def test_logp_helper_derived_rv():
    assert np.isclose(
        logp(at.exp(Normal.dist()), 5).eval(),
        logp(LogNormal.dist(), 5).eval(),
    )


def test_logp_helper_exceptions():
    with pytest.raises(TypeError, match="When RV is not a pure distribution"):
        logp(at.exp(Normal.dist()), [1, 2])

    with pytest.raises(NotImplementedError, match="PyMC could not infer logp of input variable"):
        logp(at.cos(Normal.dist()), 1)


def test_logcdf_helper():
    value = at.vector("value")
    x = Normal.dist(0, 1)

    x_logcdf = logcdf(x, value)
    np.testing.assert_almost_equal(x_logcdf.eval({value: [0, 1]}), sp.norm(0, 1).logcdf([0, 1]))

    x_logcdf = logcdf(x, [0, 1])
    np.testing.assert_almost_equal(x_logcdf.eval(), sp.norm(0, 1).logcdf([0, 1]))


def test_logcdf_transformed_argument():
    with Model() as m:
        sigma = HalfFlat("sigma")
        x = Normal("x", 0, sigma)
        Potential("norm_term", -logcdf(x, 1.0))

    sigma_value_log = -1.0
    sigma_value = np.exp(sigma_value_log)
    x_value = 0.5

    observed = m.compile_logp(jacobian=False)({"sigma_log__": sigma_value_log, "x": x_value})
    expected = logp(TruncatedNormal.dist(0, sigma_value, lower=None, upper=1.0), x_value).eval()
    assert np.isclose(observed, expected)


def test_model_unchanged_logprob_access():
    # Issue #5007
    with Model() as model:
        a = Normal("a")
        c = Uniform("c", lower=a - 1, upper=1)

    original_inputs = set(aesara.graph.graph_inputs([c]))
    # Extract model.logp
    model.logp()
    new_inputs = set(aesara.graph.graph_inputs([c]))
    assert original_inputs == new_inputs


def test_unexpected_rvs():
    with Model() as model:
        x = Normal("x")
        y = DensityDist("y", logp=lambda *args: x)

    with pytest.raises(ValueError, match="^Random variables detected in the logp graph"):
        model.logp()


def test_ignore_logprob_basic():
    x = Normal.dist()
    (measurable_x_out,) = get_measurable_outputs(x.owner.op, x.owner)
    assert measurable_x_out is x.owner.outputs[1]

    new_x = ignore_logprob(x)
    assert new_x is not x
    assert isinstance(new_x.owner.op, Normal)
    assert type(new_x.owner.op).__name__ == "UnmeasurableNormalRV"
    # Confirm that it does not have measurable output
    assert get_measurable_outputs(new_x.owner.op, new_x.owner) is None

    # Test that it will not clone a variable that is already unmeasurable
    new_new_x = ignore_logprob(new_x)
    assert new_new_x is new_x


def test_ignore_logprob_model():
    # logp that does not depend on input
    def logp(value, x):
        return value

    with Model() as m:
        x = Normal.dist()
        y = DensityDist("y", x, logp=logp)
    with pytest.warns(
        UserWarning,
        match="Found a random variable that was neither among the observations "
        "nor the conditioned variables",
    ):
        assert joint_logp([y], {y: y.type()})

    # The above warning should go away with ignore_logprob.
    with Model() as m:
        x = ignore_logprob(Normal.dist())
        y = DensityDist("y", x, logp=logp)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert joint_logp([y], {y: y.type()})
