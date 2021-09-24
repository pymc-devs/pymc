from typing import Dict, Optional

import aesara
import aesara.tensor as at
from aesara.compile.mode import optdb
from aesara.graph.features import Feature
from aesara.graph.op import compute_test_value
from aesara.graph.opt import local_optimizer
from aesara.graph.optdb import EquilibriumDB, SequenceDB
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
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from aesara.tensor.var import TensorVariable

from aeppl.abstract import MeasurableVariable
from aeppl.utils import indices_from_subtensor

inc_subtensor_ops = (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)
subtensor_ops = (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)


class PreserveRVMappings(Feature):
    r"""Keeps track of random variables and their respective value variables during
    graph rewrites in `rv_values`

    When a random variable is replaced in a rewrite, this `Feature` automatically
    updates the `rv_values` mapping, so that the new variable is linked to the
    original value variable.

    In addition this `Feature` provides functionality to manually update a random
    and/or value variable. A mapping from the transformed value variables to the
    the original value variables is kept in `original_values`.
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
        self.original_values = {v: v for v in rv_values.values()}

    def on_attach(self, fgraph):
        if hasattr(fgraph, "preserve_rv_mappings"):
            raise ValueError(
                f"{fgraph} already has the `PreserveRVMappings` feature attached."
            )

        fgraph.preserve_rv_mappings = self

    def update_rv_maps(
        self,
        old_rv: TensorVariable,
        new_value: TensorVariable,
        new_rv: Optional[TensorVariable] = None,
    ):
        """Update mappings for a random variable.

        It also creates/updates a map from new value variables to their
        original value variables.

        Parameters
        ==========
        old_rv
            The random variable whose mappings will be updated.
        new_value
            The new value variable that will replace the current one assigned
            to `old_rv`.
        new_rv
            When non-``None``, `old_rv` will also be replaced with `new_rv` in
            the mappings, as well.
        """
        old_value = self.rv_values.pop(old_rv)
        original_value = self.original_values.pop(old_value)

        if new_rv is None:
            new_rv = old_rv

        self.rv_values[new_rv] = new_value
        self.original_values[new_value] = original_value

    def on_change_input(self, fgraph, node, i, r, new_r, reason=None):
        """
        Whenever a node is replaced during rewrite, we check if it had a value
        variable associated with it and map it to the new node.
        """
        r_value_var = self.rv_values.pop(r, None)
        if r_value_var is not None:
            self.rv_values[new_r] = r_value_var


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
        return None  # pragma: no cover

    if not isinstance(node.op, inc_subtensor_ops):
        return None  # pragma: no cover

    rv_var = node.outputs[0]
    if rv_var not in rv_map_feature.rv_values:
        return None  # pragma: no cover

    base_rv_var = node.inputs[0]

    if not (
        base_rv_var.owner
        and isinstance(base_rv_var.owner.op, MeasurableVariable)
        and base_rv_var not in rv_map_feature.rv_values
    ):
        return None  # pragma: no cover

    data = node.inputs[1]
    idx = indices_from_subtensor(getattr(node.op, "idx_list", None), node.inputs[2:])

    # Create a new value variable with the indices `idx` set to `data`
    value_var = rv_map_feature.rv_values[rv_var]
    new_value_var = at.set_subtensor(value_var[idx], data)
    rv_map_feature.update_rv_maps(rv_var, new_value_var, base_rv_var)

    # Return the `RandomVariable` being indexed
    return [base_rv_var]


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

    assert len(bcast_shape) > 0

    rv_var = node.inputs[0]
    rv_node = rv_var.owner

    if hasattr(fgraph, "dont_touch_vars") and rv_var in fgraph.dont_touch_vars:
        return None  # pragma: no cover

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


logprob_rewrites_db = SequenceDB()
logprob_rewrites_db.name = "logprob_rewrites_db"
logprob_rewrites_db.register(
    "pre-canonicalize", optdb.query("+canonicalize"), -10, "basic"
)


class RVSinkingDB(EquilibriumDB):
    r"""This `EquilibriumDB` doesn't hide its exceptions.

    By setting `failure_callback` to ``None`` in the `EquilibriumOptimizer`\s
    that `EquilibriumDB` generates, we're able to directly emit the desired
    exceptions from within the `LocalOptimization`\s themselves.
    """

    def query(self, *tags, **kwtags):
        res = super().query(*tags, **kwtags)
        res.failure_callback = None
        return res


rv_sinking_db = RVSinkingDB()
rv_sinking_db.name = "rv_sinking_db"
rv_sinking_db.register("dimshuffle_lift", local_dimshuffle_rv_lift, -5, "basic")
rv_sinking_db.register("subtensor_lift", local_subtensor_rv_lift, -5, "basic")
rv_sinking_db.register("broadcast_to_lift", naive_bcast_rv_lift, -5, "basic")
rv_sinking_db.register("incsubtensor_lift", incsubtensor_rv_replace, -5, "basic")

logprob_rewrites_db.register("sinking", rv_sinking_db, -10, "basic")
logprob_rewrites_db.register(
    "post-canonicalize", optdb.query("+canonicalize"), 10, "basic"
)
