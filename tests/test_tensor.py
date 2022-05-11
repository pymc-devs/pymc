import numpy as np
from aesara import tensor as at
from aesara.graph import optimize_graph
from aesara.graph.opt import in2out
from aesara.tensor.extra_ops import BroadcastTo
from scipy import stats as st

from aeppl import factorized_joint_logprob
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
