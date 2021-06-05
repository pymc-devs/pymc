import aesara
import aesara.tensor as at
import numpy as np
import pytest
import scipy.stats.distributions as sp
from aesara.gradient import DisconnectedGrad
from aesara.graph.basic import Constant, equal_computations, graph_inputs
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

from aeppl.joint_logprob import joint_logprob
from aeppl.logprob import logprob
from aeppl.utils import rvs_to_value_vars, walk_model
from tests.utils import assert_no_rvs


def test_joint_logprob_basic():

    # A simple check for when `joint_logprob` is the same as `logprob`
    a = at.random.uniform(0.0, 1.0)
    a.name = "a"
    a_value_var = a.clone()

    a_logp = joint_logprob(a, {a: a_value_var})
    a_logp_exp = logprob(a, a_value_var)

    assert equal_computations([a_logp], [a_logp_exp])

    # Let's try a hierarchical model
    sigma = at.random.invgamma(0.5, 0.5)
    Y = at.random.normal(0.0, sigma)

    sigma_value_var = sigma.clone()
    y_value_var = Y.clone()

    total_ll = joint_logprob(Y, {Y: y_value_var, sigma: sigma_value_var})

    # We need to replace the reference to `sigma` in `Y` with its value
    # variable
    ll_Y = logprob(Y, y_value_var)
    (ll_Y,), _ = rvs_to_value_vars(
        [ll_Y],
        initial_replacements={sigma: sigma_value_var},
    )
    total_ll_exp = logprob(sigma, sigma_value_var) + ll_Y

    assert equal_computations([total_ll], [total_ll_exp])

    # Now, make sure we can compute a joint log-probability for a hierarchical
    # model with some non-`RandomVariable` nodes
    c = at.random.normal()
    c.name = "c"
    b_l = c * a + 2.0
    b = at.random.uniform(b_l, b_l + 1.0)
    b.name = "b"

    b_value_var = b.clone()
    c_value_var = c.clone()

    b_logp = joint_logprob(b, {a: a_value_var, b: b_value_var, c: c_value_var})

    # There shouldn't be any `RandomVariable`s in the resulting graph
    assert_no_rvs(b_logp)

    res_ancestors = list(walk_model((b_logp,), walk_past_rvs=True))
    assert b_value_var in res_ancestors
    assert c_value_var in res_ancestors
    assert a_value_var in res_ancestors


@pytest.mark.xfail(
    reason="This transform currently introduces duplicate log-probability terms"
)
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

    mu = np.power(10, np.arange(np.prod(size))).reshape(size)
    data = at.as_tensor(mu[indices], name="data")

    sigma = 0.001
    rng = aesara.shared(np.random.RandomState(232), borrow=True)

    Y_rv = at.random.normal(mu, sigma, size=size, rng=rng)
    Y_rv.name = "Y"
    y_value_var = Y_rv.clone()
    y_value_var.name = "y"

    idx = at.set_subtensor(Y_rv[indices], data)

    assert isinstance(
        idx.owner.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)
    )

    a_idx_value_var = idx.type()
    a_idx_value_var.name = "a_idx_value"

    a_idx_logp = joint_logprob(idx, {idx: a_idx_value_var, Y_rv: y_value_var})

    logp_vals = a_idx_logp.eval()

    # The indices that were set should all have the same joint log-probability
    # values, because the values they were set to correspond to the unique
    # means along that dimension.  This helps us confirm that the joint
    # log-probability is associating the assigned values with their correct
    # parameters.
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
    assert a_new is not Y_rv
    assert a_new.owner.op == Y_rv.owner.op

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
    assert isinstance(
        a_client.op, (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)
    )
    indices = tuple(i.eval() for i in a_client.inputs[2:])
    np.testing.assert_almost_equal(indices, indices)


def test_joint_logprob_subtensor():
    """Make sure we can compute a joint log-probability for ``Y[I]`` where ``Y`` and ``I`` are random variables."""

    size = 5

    mu_base = np.power(10, np.arange(np.prod(size))).reshape(size)
    mu = np.stack([mu_base, -mu_base])
    sigma = 0.001
    rng = aesara.shared(np.random.RandomState(232), borrow=True)

    A_rv = at.random.normal(mu, sigma, rng=rng)
    A_rv.name = "A"

    p = 0.5

    I_rv = at.random.bernoulli(p, size=size, rng=rng)
    I_rv.name = "I"

    A_idx = A_rv[I_rv, at.ogrid[A_rv.shape[-1] :]]

    assert isinstance(
        A_idx.owner.op, (Subtensor, AdvancedSubtensor, AdvancedSubtensor1)
    )

    A_idx_value_var = A_idx.type()
    A_idx_value_var.name = "A_idx_value"

    I_value_var = I_rv.type()
    I_value_var.name = "I_value"

    A_idx_logp = joint_logprob(A_idx, {A_idx: A_idx_value_var, I_rv: I_value_var})

    logp_vals_fn = aesara.function([A_idx_value_var, I_value_var], A_idx_logp)

    # The compiled graph should not contain any `RandomVariables`
    assert_no_rvs(logp_vals_fn.maker.fgraph.outputs[0])

    decimals = 6 if aesara.config.floatX == "float64" else 4

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
