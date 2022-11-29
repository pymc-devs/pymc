from typing import Dict, Optional, Tuple

import aesara.tensor as at
from aesara.compile.mode import optdb
from aesara.graph.basic import Variable
from aesara.graph.features import Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import GraphRewriter, node_rewriter
from aesara.graph.rewriting.db import EquilibriumDB, RewriteDatabaseQuery, SequenceDB
from aesara.tensor.elemwise import DimShuffle, Elemwise
from aesara.tensor.extra_ops import BroadcastTo
from aesara.tensor.random.rewriting import local_subtensor_rv_lift
from aesara.tensor.rewriting.basic import register_canonicalize, register_useless
from aesara.tensor.rewriting.shape import ShapeFeature
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
from aeppl.dists import DiracDelta
from aeppl.utils import indices_from_subtensor

inc_subtensor_ops = (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)
subtensor_ops = (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)


class NoCallbackEquilibriumDB(EquilibriumDB):
    r"""This `EquilibriumDB` doesn't hide its exceptions.

    By setting `failure_callback` to ``None`` in the `EquilibriumGraphRewriter`\s
    that `EquilibriumDB` generates, we're able to directly emit the desired
    exceptions from within the `NodeRewriter`\s themselves.
    """

    def query(self, *tags, **kwtags):
        res = super().query(*tags, **kwtags)
        res.failure_callback = None
        return res


class PreserveRVMappings(Feature):
    r"""Keeps track of random variables and their respective value variables during
    graph rewrites in `rv_values`

    When a random variable is replaced in a rewrite, this `Feature` automatically
    updates the `rv_values` mapping, so that the new variable is linked to the
    original value variable.

    In addition this `Feature` provides functionality to manually update a random
    and/or value variable. A mapping from the transformed value variables to the
    the original value variables is kept in `original_values`.

    Likewise, a `measurable_conversions` map is maintained, which holds
    information about un-valued and un-measurable variables that were replaced
    with measurable variables.  This information can be used to revert these
    rewrites.

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
        self.measurable_conversions: Dict[Variable, Variable] = {}

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
        elif (
            new_r not in self.rv_values
            and r.owner
            and new_r.owner
            and not isinstance(r.owner.op, MeasurableVariable)
            and isinstance(new_r.owner.op, MeasurableVariable)
        ):
            self.measurable_conversions[r] = new_r


@register_canonicalize
@node_rewriter((Elemwise, BroadcastTo, DimShuffle) + subtensor_ops)
def local_lift_DiracDelta(fgraph, node):
    r"""Lift basic `Op`\s through `DiracDelta`\s."""

    if len(node.outputs) > 1:
        return

    # Only handle scalar `Elemwise` `Op`s
    if isinstance(node.op, Elemwise) and len(node.inputs) != 1:
        return

    dd_inp = node.inputs[0]

    if dd_inp.owner is None or not isinstance(dd_inp.owner.op, DiracDelta):
        return

    dd_val = dd_inp.owner.inputs[0]

    new_value_node = node.op.make_node(dd_val, *node.inputs[1:])
    new_node = dd_inp.owner.op.make_node(new_value_node.outputs[0])
    return new_node.outputs


@register_useless
@node_rewriter((DiracDelta,))
def local_remove_DiracDelta(fgraph, node):
    r"""Remove `DiracDelta`\s."""
    dd_val = node.inputs[0]
    return [dd_val]


@node_rewriter(inc_subtensor_ops)
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


logprob_rewrites_db = SequenceDB()
logprob_rewrites_db.name = "logprob_rewrites_db"
logprob_rewrites_db.register(
    "pre-canonicalize", optdb.query("+canonicalize"), -10, "basic"
)

# These rewrites convert un-measurable variables into their measurable forms,
# but they need to be reapplied, because some of the measurable forms require
# their inputs to be measurable.
measurable_ir_rewrites_db = NoCallbackEquilibriumDB()
measurable_ir_rewrites_db.name = "measurable_ir_rewrites_db"

logprob_rewrites_db.register(
    "measurable_ir_rewrites", measurable_ir_rewrites_db, -10, "basic"
)

# These rewrites push random/measurable variables "down", making them closer to
# (or eventually) the graph outputs.  Often this is done by lifting other `Op`s
# "up" through the random/measurable variables and into their inputs.
measurable_ir_rewrites_db.register(
    "subtensor_lift", local_subtensor_rv_lift, -5, "basic"
)
measurable_ir_rewrites_db.register(
    "incsubtensor_lift", incsubtensor_rv_replace, -5, "basic"
)

logprob_rewrites_db.register(
    "post-canonicalize", optdb.query("+canonicalize"), 10, "basic"
)


def construct_ir_fgraph(
    rv_values: Dict[Variable, Variable],
    ir_rewriter: Optional[GraphRewriter] = None,
) -> Tuple[FunctionGraph, Dict[Variable, Variable], Dict[Variable, Variable]]:
    r"""Construct a `FunctionGraph` in measurable IR form for the keys in `rv_values`.

    A custom IR rewriter can be specified. By default,
    `logprob_rewrites_db.query(RewriteDatabaseQuery(include=["basic"]))` is used.

    Our measurable IR takes the form of an Aesara graph that is more-or-less
    equivalent to a given Aesara graph (i.e. the keys of `rv_values`) but
    contains `Op`s that are subclasses of the `MeasurableVariable` type in
    place of ones that do not inherit from `MeasurableVariable` in the original
    graph but are nevertheless measurable.

    `MeasurableVariable`\s are mapped to log-probabilities, so this IR is how
    non-trivial log-probabilities are constructed, especially when the
    "measurability" of a term depends on the measurability of its inputs
    (e.g. a mixture).

    In some cases, entire sub-graphs in the original graph are replaced with a
    single measurable node.  In other cases, the relevant nodes are already
    measurable and there is no difference between the resulting measurable IR
    graph and the original.  In general, some changes will be present,
    because--at the very least--canonicalization is always performed and the
    measurable IR includes manipulations that are not applicable to outside of
    the context of measurability/log-probabilities.

    For instance, some `Op`s will be lifted through `MeasurableVariable`\s in
    this IR, and the resulting graphs will not be computationally sound,
    because they wouldn't produce independent samples when the original graph
    would.  See https://github.com/aesara-devs/aeppl/pull/78.

    Returns
    -------
    A `FunctionGraph` of the measurable IR, a copy of `rv_values` containing
    the new, cloned versions of the original variables in `rv_values`, and
    a ``dict`` mapping all the original variables to their cloned values in
    `FunctionGraph`.
    """

    # Since we're going to clone the entire graph, we need to keep a map from
    # the old nodes to the new ones; otherwise, we won't be able to use
    # `rv_values`.
    # We start the `dict` with mappings from the value variables to themselves,
    # to prevent them from being cloned.
    memo = {v: v for v in rv_values.values()}

    # We add `ShapeFeature` because it will get rid of references to the old
    # `RandomVariable`s that have been lifted; otherwise, it will be difficult
    # to give good warnings when an unaccounted for `RandomVariable` is
    # encountered
    fgraph = FunctionGraph(
        outputs=list(rv_values.keys()),
        clone=True,
        memo=memo,
        copy_orphans=False,
        copy_inputs=False,
        features=[ShapeFeature()],
    )

    # Update `rv_values` so that it uses the new cloned variables
    rv_values = {memo[k]: v for k, v in rv_values.items()}

    # This `Feature` preserves the relationships between the original
    # random variables (i.e. keys in `rv_values`) and the new ones
    # produced when `Op`s are lifted through them.
    rv_remapper = PreserveRVMappings(rv_values)
    fgraph.attach_feature(rv_remapper)

    if ir_rewriter is None:
        ir_rewriter = logprob_rewrites_db.query(RewriteDatabaseQuery(include=["basic"]))
    ir_rewriter.rewrite(fgraph)

    if rv_remapper.measurable_conversions:
        # Undo un-valued measurable IR rewrites
        new_to_old = tuple(
            (v, k) for k, v in rv_remapper.measurable_conversions.items()
        )
        fgraph.replace_all(new_to_old)

    return fgraph, rv_values, memo
