#   Copyright 2023 The PyMC Developers
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
#
#   MIT License
#
#   Copyright (c) 2021-2022 aesara-devs
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
import warnings

from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import pytensor.tensor as pt

from pytensor import config
from pytensor.compile.mode import optdb
from pytensor.graph.basic import (
    Constant,
    Variable,
    ancestors,
    io_toposort,
    truncated_graph_inputs,
)
from pytensor.graph.features import Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import (
    ChangeTracker,
    EquilibriumGraphRewriter,
    GraphRewriter,
    node_rewriter,
    out2in,
)
from pytensor.graph.rewriting.db import (
    LocalGroupDB,
    RewriteDatabase,
    RewriteDatabaseQuery,
    SequenceDB,
    TopoDB,
)
from pytensor.tensor.basic import Alloc
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.random.rewriting import local_subtensor_rv_lift
from pytensor.tensor.rewriting.basic import register_canonicalize
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.rewriting.uncanonicalize import local_max_and_argmax
from pytensor.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import MeasurableVariable
from pymc.logprob.utils import DiracDelta, indices_from_subtensor

inc_subtensor_ops = (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)
subtensor_ops = (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)


class MeasurableEquilibriumGraphRewriter(EquilibriumGraphRewriter):
    """EquilibriumGraphRewriter focused on IR measurable rewrites.

    This is a stripped down version of the EquilibriumGraphRewriter,
    which specifically targets nodes in `PreserveRVMAppings.needs_measuring`
    that are not yet measurable.

    """

    def apply(self, fgraph):
        rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)
        if not rv_map_feature:
            return None

        change_tracker = ChangeTracker()
        fgraph.attach_feature(change_tracker)

        changed = True
        max_use_abort = False
        rewriter_name = None
        global_process_count = {}

        for rewriter in self.global_rewriters + list(self.get_node_rewriters()):
            global_process_count.setdefault(rewriter, 0)

        while changed and not max_use_abort:
            changed = False
            max_nb_nodes = len(fgraph.apply_nodes)
            max_use = max_nb_nodes * self.max_use_ratio

            # Apply global rewriters
            for grewrite in self.global_rewriters:
                change_tracker.reset()
                grewrite.apply(fgraph)
                if change_tracker.changed:
                    global_process_count[grewrite] += 1
                    changed = True
                    if global_process_count[grewrite] > max_use:
                        max_use_abort = True
                        rewriter_name = getattr(grewrite, "name", None) or getattr(
                            grewrite, "__name__", ""
                        )

            # Apply local node rewriters
            q = deque(io_toposort(fgraph.inputs, fgraph.outputs))
            while q:
                node = q.pop()
                if node not in fgraph.apply_nodes:
                    continue
                # This is where we filter only those nodes we care about:
                # Nodes that have variables that we want to measure and are not yet measurable
                if isinstance(node.op, MeasurableVariable):
                    continue
                if not any(out in rv_map_feature.needs_measuring for out in node.outputs):
                    continue
                for node_rewriter in self.node_tracker.get_trackers(node.op):
                    node_rewriter_change = self.process_node(fgraph, node, node_rewriter)
                    if not node_rewriter_change:
                        continue
                    global_process_count[node_rewriter] += 1
                    changed = True
                    if global_process_count[node_rewriter] > max_use:
                        max_use_abort = True
                        rewriter_name = getattr(node_rewriter, "name", None) or getattr(
                            node_rewriter, "__name__", ""
                        )
                    # If we converted to a MeasurableVariable we're done here!
                    if node not in fgraph.apply_nodes or isinstance(node.op, MeasurableVariable):
                        # go to next node
                        break

        if max_use_abort:
            msg = (
                f"{type(self).__name__} max'ed out by {rewriter_name}."
                "You can safely raise the current threshold of "
                f"{config.optdb__max_use_ratio} with the option `optdb__max_use_ratio`."
            )
            if config.on_opt_error == "raise":
                raise AssertionError(msg)
            else:
                warnings.warn(msg)
        fgraph.remove_feature(change_tracker)


class MeasurableEquilibriumDB(RewriteDatabase):
    """A database of rewrites that should be applied until equilibrium is reached.

    This will return a MeasurableEquilibriumGraphRewriter when queried.

    """

    def query(self, *tags, **kwtags):
        rewriters = super().query(*tags, **kwtags)
        return MeasurableEquilibriumGraphRewriter(
            rewriters,
            max_use_ratio=config.optdb__max_use_ratio,
        )


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
        ----------
        rv_values
            Mappings between random variables and their value variables.
            The keys of this map are what this `Feature` keeps updated.
            The ``dict`` is updated in-place.
        """
        self.rv_values = rv_values
        self.original_values = {v: v for v in rv_values.values()}
        self.needs_measuring = set(rv_values.keys())

    def on_attach(self, fgraph):
        if hasattr(fgraph, "preserve_rv_mappings"):
            raise ValueError(f"{fgraph} already has the `PreserveRVMappings` feature attached.")

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
        ----------
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
            self.needs_measuring.add(new_r)
            if new_r.name is None:
                new_r.name = r.name

    def request_measurable(self, vars: Sequence[Variable]) -> List[Variable]:
        measurable = []
        for var in vars:
            # Input vars or valued vars can't be measured for derived expressions
            if not var.owner or var in self.rv_values:
                continue
            if isinstance(var.owner.op, MeasurableVariable):
                measurable.append(var)
            else:
                self.needs_measuring.add(var)
        return measurable


@register_canonicalize
@node_rewriter((Elemwise, Alloc, DimShuffle) + subtensor_ops)
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


@node_rewriter([DiracDelta])
def remove_DiracDelta(fgraph, node):
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
    where ``z = pt.set_subtensor(y[idx], data)`` and ``y`` is the value variable
    for ``Y``.

    In other words, the log-probability for an `*IncSubtensor*` is the log-probability
    of the underlying `RandomVariable` evaluated at ``data`` for the indices
    given by ``idx`` and at the value variable for ``~idx``.

    This provides a means of specifying "missing data", for instance.
    """
    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    rv_var = node.outputs[0]
    if rv_var not in rv_map_feature.rv_values:
        return None  # pragma: no cover

    base_rv_var = node.inputs[0]

    if not rv_map_feature.request_measurable([base_rv_var]):
        return None

    data = node.inputs[1]
    idx = indices_from_subtensor(getattr(node.op, "idx_list", None), node.inputs[2:])

    # Create a new value variable with the indices `idx` set to `data`
    value_var = rv_map_feature.rv_values[rv_var]
    new_value_var = pt.set_subtensor(value_var[idx], data)
    rv_map_feature.update_rv_maps(rv_var, new_value_var, base_rv_var)

    # Return the `RandomVariable` being indexed
    return [base_rv_var]


logprob_rewrites_db = SequenceDB()
logprob_rewrites_db.name = "logprob_rewrites_db"
logprob_rewrites_db.register("pre-canonicalize", optdb.query("+canonicalize"), "basic")
logprob_rewrites_db.register("local_max_and_argmax", out2in(local_max_and_argmax), "basic")

# These rewrites convert un-measurable variables into their measurable forms,
# but they need to be reapplied, because some of the measurable forms require
# their inputs to be measurable.
measurable_ir_rewrites_db = MeasurableEquilibriumDB()
measurable_ir_rewrites_db.name = "measurable_ir_rewrites_db"

logprob_rewrites_db.register("measurable_ir_rewrites", measurable_ir_rewrites_db, "basic")

# These rewrites push random/measurable variables "down", making them closer to
# (or eventually) the graph outputs.  Often this is done by lifting other `Op`s
# "up" through the random/measurable variables and into their inputs.
measurable_ir_rewrites_db.register("subtensor_lift", local_subtensor_rv_lift, "basic")
measurable_ir_rewrites_db.register("incsubtensor_lift", incsubtensor_rv_replace, "basic")

logprob_rewrites_db.register("post-canonicalize", optdb.query("+canonicalize"), "basic")

# Rewrites that remove IR Ops
cleanup_ir_rewrites_db = LocalGroupDB()
cleanup_ir_rewrites_db.name = "cleanup_ir_rewrites_db"
logprob_rewrites_db.register(
    "cleanup_ir_rewrites",
    TopoDB(cleanup_ir_rewrites_db, order="out_to_in", ignore_newtrees=True, failure_callback=None),
    "cleanup",
)

cleanup_ir_rewrites_db.register("remove_DiracDelta", remove_DiracDelta, "cleanup")


def construct_ir_fgraph(
    rv_values: Dict[Variable, Variable],
    ir_rewriter: Optional[GraphRewriter] = None,
) -> Tuple[FunctionGraph, Dict[Variable, Variable], Dict[Variable, Variable]]:
    r"""Construct a `FunctionGraph` in measurable IR form for the keys in `rv_values`.

    A custom IR rewriter can be specified. By default,
    `logprob_rewrites_db.query(RewriteDatabaseQuery(include=["basic"]))` is used.

    Our measurable IR takes the form of an PyTensor graph that is more-or-less
    equivalent to a given PyTensor graph (i.e. the keys of `rv_values`) but
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
    # to prevent them from being cloned. This also includes ancestors
    memo = {v: v for v in ancestors(rv_values.values()) if not isinstance(v, Constant)}

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

    return fgraph, rv_values, memo


def cleanup_ir(vars: Sequence[Variable]) -> None:
    fgraph = FunctionGraph(outputs=vars, clone=False)
    ir_rewriter = logprob_rewrites_db.query(RewriteDatabaseQuery(include=["cleanup"]))
    ir_rewriter.rewrite(fgraph)


def assume_measured_ir_outputs(
    inputs: Sequence[TensorVariable], outputs: Sequence[TensorVariable]
) -> Sequence[TensorVariable]:
    """Run IR rewrite assuming each output is measured.

    IR variables could depend on each other in a way that looks unmeasurable without a value variable assigned to each.
    For instance `join([add(x, z), z])` is a potentially measurable join, but `add(x, z)` can look unmeasurable
    because neither `x` and `z` are valued in the IR representation.
    This helper runs an inner ir rewrite after giving each output a dummy value variable.
    We replace inputs by dummies and then undo it so that any dependency on outer variables is preserved.
    """
    # Replace inputs by dummy variables
    replaced_inputs = {
        var: var.type()
        for var in truncated_graph_inputs(outputs, ancestors_to_include=inputs)
        if var in inputs
    }
    cloned_outputs = clone_replace(outputs, replace=replaced_inputs)

    dummy_rv_values = {base_var: base_var.type() for base_var in cloned_outputs}
    fgraph, *_ = construct_ir_fgraph(dummy_rv_values)

    # Replace dummy variables by inputs
    fgraph.replace_all(
        tuple((repl, orig) for orig, repl in replaced_inputs.items()),
        import_missing=True,
    )

    return fgraph.outputs
