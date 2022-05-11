from typing import Optional

import aesara
from aesara import tensor as at
from aesara.graph.op import compute_test_value
from aesara.graph.opt import local_optimizer
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.opt import local_dimshuffle_rv_lift, local_rv_size_lift

from aeppl.opt import PreserveRVMappings, measurable_ir_rewrites_db


@local_optimizer([BroadcastTo])
def naive_bcast_rv_lift(fgraph, node):
    """Lift a ``BroadcastTo`` through a ``RandomVariable`` ``Op``.

    XXX: This implementation simply broadcasts the ``RandomVariable``'s
    parameters, which won't always work (e.g. multivariate distributions).

    TODO: Instead, it should use ``RandomVariable.ndim_supp``--and the like--to
    determine which dimensions of each parameter need to be broadcasted.
    Also, this doesn't need to remove ``size`` to perform the lifting, like it
    currently does.
    """

    if not (
        isinstance(node.op, BroadcastTo)
        and node.inputs[0].owner
        and isinstance(node.inputs[0].owner.op, RandomVariable)
    ):
        return None  # pragma: no cover

    bcast_shape = node.inputs[1:]

    rv_var = node.inputs[0]
    rv_node = rv_var.owner

    if hasattr(fgraph, "dont_touch_vars") and rv_var in fgraph.dont_touch_vars:
        return None  # pragma: no cover

    # Do not replace RV if it is associated with a value variable
    rv_map_feature: Optional[PreserveRVMappings] = getattr(
        fgraph, "preserve_rv_mappings", None
    )
    if rv_map_feature is not None and rv_var in rv_map_feature.rv_values:
        return None

    if not bcast_shape:
        # The `BroadcastTo` is broadcasting a scalar to a scalar (i.e. doing nothing)
        assert rv_var.ndim == 0
        return [rv_var]

    size_lift_res = local_rv_size_lift.transform(fgraph, rv_node)
    if size_lift_res is None:
        lifted_node = rv_node
    else:
        _, lifted_rv = size_lift_res
        lifted_node = lifted_rv.owner

    rng, size, dtype, *dist_params = lifted_node.inputs

    new_dist_params = [
        at.broadcast_to(
            param,
            at.broadcast_shape(
                tuple(param.shape), tuple(bcast_shape), arrays_are_shapes=True
            ),
        )
        for param in dist_params
    ]
    bcasted_node = lifted_node.op.make_node(rng, size, dtype, *new_dist_params)

    if aesara.config.compute_test_value != "off":
        compute_test_value(bcasted_node)

    return [bcasted_node.outputs[1]]


measurable_ir_rewrites_db.register(
    "dimshuffle_lift", local_dimshuffle_rv_lift, -5, "basic", "tensor"
)

measurable_ir_rewrites_db.register(
    "broadcast_to_lift", naive_bcast_rv_lift, -5, "basic", "tensor"
)
