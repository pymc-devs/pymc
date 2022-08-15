import aesara
import numpy as np
import pytest
from aesara import tensor as at
from aesara.graph import optimize_graph
from aesara.graph.opt import in2out
from aesara.tensor.extra_ops import BroadcastTo
from scipy import stats as st

from aeppl import factorized_joint_logprob, joint_logprob
from aeppl.tensor import naive_bcast_rv_lift


def test_naive_bcast_rv_lift():
    r"""Make sure `naive_bcast_rv_lift` can handle useless scalar `BroadcastTo`\s."""
    X_rv = at.random.normal()
    Z_at = BroadcastTo()(X_rv, ())

    # Make sure we're testing what we intend to test
    assert isinstance(Z_at.owner.op, BroadcastTo)

    res = optimize_graph(Z_at, custom_opt=in2out(naive_bcast_rv_lift), clone=False)
    assert res is X_rv


def test_naive_bcast_rv_lift_valued_var():
    r"""Check that `naive_bcast_rv_lift` won't touch valued variables"""

    x_rv = at.random.normal(name="x")
    broadcasted_x_rv = at.broadcast_to(x_rv, (2,))

    y_rv = at.random.normal(broadcasted_x_rv, name="y")

    x_vv = x_rv.clone()
    y_vv = y_rv.clone()
    logp_map = factorized_joint_logprob({x_rv: x_vv, y_rv: y_vv})
    assert x_vv in logp_map
    assert y_vv in logp_map
    assert len(logp_map) == 2
    assert np.allclose(logp_map[x_vv].eval({x_vv: 0}), st.norm(0).logpdf(0))
    assert np.allclose(
        logp_map[y_vv].eval({x_vv: 0, y_vv: [0, 0]}), st.norm(0).logpdf([0, 0])
    )


def test_measurable_make_vector():
    base1_rv = at.random.normal(name="base1")
    base2_rv = at.random.halfnormal(name="base2")
    base3_rv = at.random.exponential(name="base3")
    y_rv = at.stack((base1_rv, base2_rv, base3_rv))
    y_rv.name = "y"

    base1_vv = base1_rv.clone()
    base2_vv = base2_rv.clone()
    base3_vv = base3_rv.clone()
    y_vv = y_rv.clone()

    ref_logp = joint_logprob(
        {base1_rv: base1_vv, base2_rv: base2_vv, base3_rv: base3_vv}
    )
    make_vector_logp = joint_logprob({y_rv: y_vv}, sum=False)

    base1_testval = base1_rv.eval()
    base2_testval = base2_rv.eval()
    base3_testval = base3_rv.eval()
    y_testval = np.stack((base1_testval, base2_testval, base3_testval))

    ref_logp_eval_eval = ref_logp.eval(
        {base1_vv: base1_testval, base2_vv: base2_testval, base3_vv: base3_testval}
    )
    make_vector_logp_eval = make_vector_logp.eval({y_vv: y_testval})

    assert make_vector_logp_eval.shape == y_testval.shape
    assert np.isclose(make_vector_logp_eval.sum(), ref_logp_eval_eval)


@pytest.mark.parametrize(
    "size1, size2, axis, concatenate",
    [
        ((5,), (3,), 0, True),
        ((5,), (3,), -1, True),
        ((5, 2), (3, 2), 0, True),
        ((2, 5), (2, 3), 1, True),
        ((2, 5), (2, 5), 0, False),
        ((2, 5), (2, 5), 1, False),
        ((2, 5), (2, 5), 2, False),
    ],
)
def test_measurable_join_univariate(size1, size2, axis, concatenate):
    base1_rv = at.random.normal(size=size1, name="base1")
    base2_rv = at.random.exponential(size=size2, name="base2")
    if concatenate:
        y_rv = at.concatenate((base1_rv, base2_rv), axis=axis)
    else:
        y_rv = at.stack((base1_rv, base2_rv), axis=axis)
    y_rv.name = "y"

    base1_vv = base1_rv.clone()
    base2_vv = base2_rv.clone()
    y_vv = y_rv.clone()

    base_logps = list(
        factorized_joint_logprob({base1_rv: base1_vv, base2_rv: base2_vv}).values()
    )
    if concatenate:
        base_logps = at.concatenate(base_logps, axis=axis)
    else:
        base_logps = at.stack(base_logps, axis=axis)
    y_logp = joint_logprob({y_rv: y_vv}, sum=False)

    base1_testval = base1_rv.eval()
    base2_testval = base2_rv.eval()
    if concatenate:
        y_testval = np.concatenate((base1_testval, base2_testval), axis=axis)
    else:
        y_testval = np.stack((base1_testval, base2_testval), axis=axis)
    np.testing.assert_allclose(
        base_logps.eval({base1_vv: base1_testval, base2_vv: base2_testval}),
        y_logp.eval({y_vv: y_testval}),
    )


@pytest.mark.parametrize(
    "size1, supp_size1, size2, supp_size2, axis, concatenate",
    [
        (None, 2, None, 2, 0, True),
        (None, 2, None, 2, -1, True),
        ((5,), 2, (3,), 2, 0, True),
        ((5,), 2, (3,), 2, -2, True),
        ((2,), 5, (2,), 3, 1, True),
        pytest.param(
            (2,),
            5,
            (2,),
            5,
            0,
            False,
            marks=pytest.mark.xfail(
                reason="cannot measure dimshuffled multivariate RVs"
            ),
        ),
        pytest.param(
            (2,),
            5,
            (2,),
            5,
            1,
            False,
            marks=pytest.mark.xfail(
                reason="cannot measure dimshuffled multivariate RVs"
            ),
        ),
    ],
)
def test_measurable_join_multivariate(
    size1, supp_size1, size2, supp_size2, axis, concatenate
):
    base1_rv = at.random.multivariate_normal(
        np.zeros(supp_size1), np.eye(supp_size1), size=size1, name="base1"
    )
    base2_rv = at.random.dirichlet(np.ones(supp_size2), size=size2, name="base2")
    if concatenate:
        y_rv = at.concatenate((base1_rv, base2_rv), axis=axis)
    else:
        y_rv = at.stack((base1_rv, base2_rv), axis=axis)
    y_rv.name = "y"

    base1_vv = base1_rv.clone()
    base2_vv = base2_rv.clone()
    y_vv = y_rv.clone()
    base_logps = [
        at.atleast_1d(logp)
        for logp in factorized_joint_logprob(
            {base1_rv: base1_vv, base2_rv: base2_vv}
        ).values()
    ]

    if concatenate:
        axis_norm = np.core.numeric.normalize_axis_index(axis, base1_rv.ndim)
        base_logps = at.concatenate(base_logps, axis=axis_norm - 1)
    else:
        axis_norm = np.core.numeric.normalize_axis_index(axis, base1_rv.ndim + 1)
        base_logps = at.stack(base_logps, axis=axis_norm - 1)
    y_logp = joint_logprob({y_rv: y_vv}, sum=False)

    base1_testval = base1_rv.eval()
    base2_testval = base2_rv.eval()
    if concatenate:
        y_testval = np.concatenate((base1_testval, base2_testval), axis=axis)
    else:
        y_testval = np.stack((base1_testval, base2_testval), axis=axis)
    np.testing.assert_allclose(
        base_logps.eval({base1_vv: base1_testval, base2_vv: base2_testval}),
        y_logp.eval({y_vv: y_testval}),
    )


def test_join_mixed_ndim_supp():
    base1_rv = at.random.normal(size=3, name="base1")
    base2_rv = at.random.dirichlet(np.ones(3), name="base2")
    y_rv = at.concatenate((base1_rv, base2_rv), axis=0)

    y_vv = y_rv.clone()
    with pytest.raises(
        ValueError, match="Joined logps have different number of dimensions"
    ):
        joint_logprob({y_rv: y_vv})


@aesara.config.change_flags(cxx="")
@pytest.mark.parametrize(
    "ds_order",
    [
        (0, 2, 1),  # Swap
        (2, 1, 0),  # Swap
        (0, 1, 2, "x"),  # Expand
        ("x", 0, 1, 2),  # Expand
        (
            0,
            2,
        ),  # Drop
        (2, 0),  # Swap and drop
        (2, 1, "x", 0),  # Swap and expand
        ("x", 0, 2),  # Expand and drop
        (2, "x", 0),  # Swap, expand and drop
    ],
)
@pytest.mark.parametrize("multivariate", (False, True))
def test_measurable_dimshuffle(ds_order, multivariate):
    if multivariate:
        base_rv = at.random.dirichlet([1, 2, 3], size=(2, 1))
    else:
        base_rv = at.exp(at.random.beta(1, 2, size=(2, 1, 3)))

    ds_rv = base_rv.dimshuffle(ds_order)
    base_vv = base_rv.clone()
    ds_vv = ds_rv.clone()

    # Remove support dimension axis from ds_order (i.e., 2, for multivariate)
    if multivariate:
        logp_ds_order = [o for o in ds_order if o == "x" or o < 2]
    else:
        logp_ds_order = ds_order

    ref_logp = joint_logprob({base_rv: base_vv}, sum=False).dimshuffle(logp_ds_order)
    ds_logp = joint_logprob({ds_rv: ds_vv}, sum=False)
    assert ds_logp is not None

    ref_logp_fn = aesara.function([base_vv], ref_logp)
    ds_logp_fn = aesara.function([ds_vv], ds_logp)

    base_test_value = base_rv.eval()
    ds_test_value = at.constant(base_test_value).dimshuffle(ds_order).eval()

    np.testing.assert_array_equal(
        ref_logp_fn(base_test_value), ds_logp_fn(ds_test_value)
    )
