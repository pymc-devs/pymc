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

from typing import Dict, Optional, Sequence, Tuple

from pytensor.compile.mode import optdb
from pytensor.graph.basic import Variable, ancestors, truncated_graph_inputs
from pytensor.graph.features import Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import GraphRewriter, node_rewriter, out2in
from pytensor.graph.rewriting.db import (
    EquilibriumDB,
    LocalGroupDB,
    RewriteDatabaseQuery,
    SequenceDB,
    TopoDB,
)
from pytensor.tensor.basic import Alloc
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.random.rewriting import local_subtensor_rv_lift
from pytensor.tensor.rewriting.basic import register_canonicalize
from pytensor.tensor.rewriting.math import local_exp_over_1_plus_exp
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

from pymc.logprob.abstract import (
    MeasurableVariable,
    PromisedValuedRV,
    ValuedRV,
    valued_rv,
)
from pymc.logprob.utils import DiracDelta
from pymc.pytensorf import toposort_replace

inc_subtensor_ops = (IncSubtensor, AdvancedIncSubtensor, AdvancedIncSubtensor1)
subtensor_ops = (AdvancedSubtensor, AdvancedSubtensor1, Subtensor)


class PreserveRVMappings(Feature):
    pass


MeasurableVariable.register(ValuedRV)


@node_rewriter([ValuedRV, PromisedValuedRV])
def remove_ValuedRV(fgraph, node):
    rv = node.inputs[0]
    return [rv]


remove_valued_rvs = out2in(remove_ValuedRV)


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


logprob_rewrites_db = SequenceDB()
logprob_rewrites_db.name = "logprob_rewrites_db"

# Rewrites that must run before any canonicalization or the main Equilibrium DB
early_measurable_ir_rewrites_db = LocalGroupDB()
early_measurable_ir_rewrites_db.name = "early_measurable_rewrites_db"
logprob_rewrites_db.register(
    "early_ir_rewrites",
    TopoDB(
        early_measurable_ir_rewrites_db,
        order="out_to_in",
        ignore_newtrees=False,
        failure_callback=None,
    ),
    "basic",
)

# Introduce sigmoid. We do it before canonicalization so that useless mul are removed next
logprob_rewrites_db.register(
    "local_exp_over_1_plus_exp", out2in(local_exp_over_1_plus_exp), "basic"
)
logprob_rewrites_db.register("pre-canonicalize", optdb.query("+canonicalize"), "basic")
# Split max_and_argmax
logprob_rewrites_db.register("local_max_and_argmax", out2in(local_max_and_argmax), "basic")

# These rewrites convert un-measurable variables into their measurable forms,
# but they need to be reapplied, because some of the measurable forms require
# their inputs to be measurable.
measurable_ir_rewrites_db = EquilibriumDB()
measurable_ir_rewrites_db.name = "measurable_ir_rewrites_db"

logprob_rewrites_db.register("measurable_ir_rewrites", measurable_ir_rewrites_db, "basic")

# These rewrites push random/measurable variables "down", making them closer to
# (or eventually) the graph outputs.  Often this is done by lifting other `Op`s
# "up" through the random/measurable variables and into their inputs.
measurable_ir_rewrites_db.register("subtensor_lift", local_subtensor_rv_lift, "basic")

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
cleanup_ir_rewrites_db.register("remove_ValuedVar", remove_ValuedRV, "cleanup")


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
    # the old nodes to the new ones; otherwise, we won't be able to use `rv_values`.
    memo = {}

    # We add `ShapeFeature` because it will get rid of references to the old
    # `RandomVariable`s that have been lifted; otherwise, it will be difficult
    # to give good warnings when an unaccounted for `RandomVariable` is encountered
    fgraph = FunctionGraph(
        outputs=list(rv_values.keys()),
        clone=True,
        memo=memo,
        copy_orphans=False,
        copy_inputs=False,
        features=[ShapeFeature()],
    )

    # Replace valued RVs by ValuedVar Ops so that rewrites are aware of conditioning points
    # We use clones of the value variables so that they are not affected by rewrites
    cloned_values = {v: v.clone() for v in rv_values.values()}
    ir_rv_values = {memo[k]: cloned_values[v] for k, v in rv_values.items()}

    replacements = tuple((rv, valued_rv(rv, value)) for rv, value in ir_rv_values.items())
    toposort_replace(fgraph, replacements, reverse=True)

    if ir_rewriter is None:
        ir_rewriter = logprob_rewrites_db.query(RewriteDatabaseQuery(include=["basic"]))
    ir_rewriter.rewrite(fgraph)

    # Reintroduce original value variables
    replacements = tuple((cloned_v, v) for v, cloned_v in cloned_values.items())
    toposort_replace(fgraph, replacements=replacements, reverse=True)

    return fgraph, rv_values, memo


def cleanup_ir(vars: Sequence[Variable]) -> None:
    fgraph = FunctionGraph(outputs=vars, clone=False)
    ir_rewriter = logprob_rewrites_db.query(RewriteDatabaseQuery(include=["cleanup"]))
    ir_rewriter.rewrite(fgraph)


def assume_valued_outputs(
    outputs: Sequence[TensorVariable],
) -> Sequence[TensorVariable]:
    """Run IR rewrite assuming each output is measured.

    IR variables could depend on each other in a way that looks unmeasurable without a value variable assigned to each.
    For instance `join([add(x, z), z])` is a potentially measurable join, but `add(x, z)` can look unmeasurable
    because neither `x` and `z` are valued in the IR representation.
    This helper runs an inner ir rewrite after giving each output a dummy value variable.
    We replace inputs by dummies and then undo it so that any dependency on outer variables is preserved.
    """
    # Replace inputs by dummy variables (so they are not affected)
    inputs = [
        valued_var
        for valued_var in ancestors(outputs)
        if (valued_var.owner and isinstance(valued_var.owner.op, ValuedRV))
    ]
    replaced_inputs = {
        var: var.type()
        for var in truncated_graph_inputs(outputs, ancestors_to_include=inputs)
        if var in inputs
    }
    cloned_outputs = clone_replace(outputs, replace=replaced_inputs)

    dummy_rv_values = {base_var: base_var.type() for base_var in cloned_outputs}
    fgraph, *_ = construct_ir_fgraph(dummy_rv_values)
    remove_valued_rvs.apply(fgraph)

    # Replace dummy variables by original inputs
    fgraph.replace_all(
        tuple((repl, orig) for orig, repl in replaced_inputs.items()),
        import_missing=True,
    )

    return fgraph.outputs
