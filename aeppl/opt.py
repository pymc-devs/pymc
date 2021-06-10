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
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    IncSubtensor,
)
from aesara.tensor.var import TensorVariable

from aeppl.utils import indices_from_subtensor

inc_subtensor_ops = (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)


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

    def on_attach(self, fgraph):
        if hasattr(fgraph, "preserve_rv_mappings"):
            raise ValueError(
                f"{fgraph} already has the `PreserveRVMappings` feature attached."
            )

        fgraph.preserve_rv_mappings = self

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        r_value_var = self.rv_values.pop(r, None)
        if r_value_var is not None:
            self.rv_values[new_r] = r_value_var


class RVSinker(EquilibriumOptimizer):
    """Sink `RandomVariable` `Op`s so that log-probabilities can be determined.

    This optimizer is essentially a collection of `RandomVariable`-based rewrites
    that are needed in order to compute log-probabilities for non-`RandomVariable`
    graphs.
    """

    def __init__(self):
        super().__init__(
            [
                local_dimshuffle_rv_lift,
                local_subtensor_rv_lift,
                naive_bcast_rv_lift,
                incsubtensor_rv_replace,
            ],
            ignore_newtrees=False,
            tracks_on_change_inputs=True,
            max_use_ratio=10000,
        )

    # If we wanted to support the `.tag.value_var` approach,
    # something like the following would be reasonable:
    # def add_requirements(self, fgraph):
    #     if not hasattr(fgraph, "preserve_rv_mappings"):
    #         fgraph.attach_feature(PreserveRVMappings({}))


@local_optimizer(inc_subtensor_ops)
def incsubtensor_rv_replace(fgraph, node):
    r"""Replace `*IncSubtensor*` `Op`\s and their value variables for log-probability calculations.

    This is used to derive the log-probability graph for ``Y[idx] = data``, where
    ``Y`` is a `RandomVariable`, ``idx`` indices, and ``data`` some arbitrary data.

    To compute the log-probability of a statement like ``Y[idx] = data``, we must
    first realize that our objective is equivalent to computing ``logprob(Y, z)``,
    where ``z = at.set_subtensor(y[idx], data)`` and ``y`` is the value variable
    for ``Y``.

    In other words, the log-probability for an `*IncSubtensor*` is the log-probability
    of the underlying `RandomVariable` evaluated at ``data`` for the indices
    given by ``idx`` and at the value variable for ``~idx``.

    This provides a means of specifying "missing data", for instance.
    """
    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return

    if not isinstance(node.op, inc_subtensor_ops):
        return

    rv_var = node.inputs[0]

    if not (rv_var.owner and isinstance(rv_var.owner.op, RandomVariable)):
        return

    data = node.inputs[1]
    idx = indices_from_subtensor(getattr(node.op, "idx_list", None), node.inputs[2:])

    # Create a new value variable with the indices `idx` set to `data`
    z = at.set_subtensor(rv_map_feature.rv_values[rv_var][idx], data)

    # Replace `rv_var`'s value variable with the new one
    rv_map_feature.rv_values[rv_var] = z

    # Return the `RandomVariable` being indexed
    return [rv_var]


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
