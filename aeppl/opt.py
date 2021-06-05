from typing import Dict

import aesara
import aesara.tensor as at
from aesara.graph.features import Feature
from aesara.graph.op import compute_test_value
from aesara.graph.opt import EquilibriumOptimizer, local_optimizer
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.opt import (
    local_dimshuffle_rv_lift,
    local_rv_size_lift,
    local_subtensor_rv_lift,
)
from aesara.tensor.var import TensorVariable


class PreserveRVMappings(Feature):
    r"""Keep track of random variable replacements in a map.

    When a `Variable` that is replaced by optimizations, this `Feature` updates
    the key entries in a map to reflect the new transformed `Variable`\s.
    """

    def __init__(self, rv_values: Dict[TensorVariable, TensorVariable]):
        """
        Parameters
        ==========
        rv_values
            Mappings between random variables and their value variables.
            The keys of this map are what this `Feature` keeps updated.
            The ``dict`` is updated in-place.
        """
        self.rv_values = rv_values

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        r_value_var = self.rv_values.pop(r, None)
        if r_value_var is not None:
            self.rv_values[new_r] = r_value_var


class RVSinker(EquilibriumOptimizer):
    """Sink `RandomVariable` `Op`s so that log-probabilities can be determined."""

    def __init__(self):
        super().__init__(
            [
                local_dimshuffle_rv_lift,
                local_subtensor_rv_lift,
                naive_bcast_rv_lift,
            ],
            ignore_newtrees=False,
            tracks_on_change_inputs=True,
            max_use_ratio=10000,
        )


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
        return

    bcast_shape = node.inputs[1:]

    assert len(bcast_shape) > 0

    rv_var = node.inputs[0]
    rv_node = rv_var.owner

    if hasattr(fgraph, "dont_touch_vars") and rv_var in fgraph.dont_touch_vars:
        return

    size_lift_res = local_rv_size_lift.transform(fgraph, rv_node)
    if size_lift_res is None:
        lifted_node = rv_node
    else:
        _, lifted_rv = size_lift_res
        lifted_node = lifted_rv.owner

    rng, size, dtype, *dist_params = lifted_node.inputs

    new_dist_params = [at.broadcast_to(param, bcast_shape) for param in dist_params]
    bcasted_node = lifted_node.op.make_node(rng, size, dtype, *new_dist_params)

    if aesara.config.compute_test_value != "off":
        compute_test_value(bcasted_node)

    return [bcasted_node.outputs[1]]
