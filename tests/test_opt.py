import aesara
import aesara.tensor as at
import numpy as np
import scipy.stats as st
from aesara.graph.opt import in2out
from aesara.graph.opt_utils import optimize_graph
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.subtensor import Subtensor

from aeppl import factorized_joint_logprob
from aeppl.dists import DiracDelta, dirac_delta
from aeppl.opt import local_lift_DiracDelta, naive_bcast_rv_lift


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


def test_local_lift_DiracDelta():
    c_at = at.vector()
    dd_at = dirac_delta(c_at)

    Z_at = at.cast(dd_at, "int64")

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Elemwise)

    Z_at = dd_at.dimshuffle("x", 0)

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, DimShuffle)

    Z_at = dd_at[0]

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert isinstance(res.owner.op, DiracDelta)
    assert isinstance(res.owner.inputs[0].owner.op, Subtensor)

    # Don't lift multi-output `Op`s
    c_at = at.matrix()
    dd_at = dirac_delta(c_at)
    Z_at = at.nlinalg.svd(dd_at)[0]

    res = optimize_graph(Z_at, custom_opt=in2out(local_lift_DiracDelta), clone=False)
    assert res is Z_at


def test_local_remove_DiracDelta():
    c_at = at.vector()
    dd_at = dirac_delta(c_at)

    fn = aesara.function([c_at], dd_at)
    assert not any(
        isinstance(node.op, DiracDelta) for node in fn.maker.fgraph.toposort()
    )
