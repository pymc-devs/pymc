#   Copyright 2024 - present The PyMC Developers
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


import pytensor.tensor as pt

from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import EquilibriumGraphRewriter
from pytensor.graph.traversal import walk
from pytensor.ifelse import IfElse, ifelse
from pytensor.raise_op import CheckAndRaise
from pytensor.scalar import Switch
from pytensor.tensor.basic import Join, MakeVector
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.rewriting import (
    local_dimshuffle_rv_lift,
    local_rv_size_lift,
    local_subtensor_rv_lift,
)
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.rewriting.shape import ShapeFeature
from pytensor.tensor.subtensor import (
    AdvancedSubtensor,
    Subtensor,
    unflatten_index_variables,
)
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import (
    LogprobQuery,
    ValuedRV,
    logprob_query,
)
from pymc.logprob.query import (
    UnsupportedObservation,
    bind_value,
    contains_random,
    density_sources,
    derive_graph,
    infer_support_axes,
    other_query_uses,
    query_parts,
    rewrite_logprob_query,
    support_axes,
)
from pymc.logprob.rewriting import (
    local_lift_DiracDelta,
    subtensor_ops,
)


def rv_pull_down(x: TensorVariable) -> TensorVariable:
    """Pull a ``RandomVariable`` ``Op`` down through a graph, when possible."""
    fgraph = FunctionGraph(outputs=[x], clone=False, features=[ShapeFeature()])
    rewrites = [
        local_rv_size_lift,
        local_dimshuffle_rv_lift,
        local_subtensor_rv_lift,
        local_lift_DiracDelta,
    ]
    EquilibriumGraphRewriter(rewrites, max_use_ratio=100).rewrite(fgraph)
    return fgraph.outputs[0]


def get_stack_mixture_vars(
    node: Apply,
) -> tuple[list[TensorVariable] | None, int | None]:
    """Extract the components and stacking axis from an indexed stack."""
    assert isinstance(node.op, subtensor_ops)

    joined_rvs = node.inputs[0]

    # First, make sure that it's some sort of concatenation
    if not (joined_rvs.owner and isinstance(joined_rvs.owner.op, MakeVector | Join)):
        return None, None

    if isinstance(joined_rvs.owner.op, MakeVector):
        join_axis = None
        mixture_rvs = joined_rvs.owner.inputs

    elif isinstance(joined_rvs.owner.op, Join):
        join_axis = joined_rvs.owner.op.axis
        mixture_rvs = joined_rvs.owner.inputs

    return mixture_rvs, join_axis


@rewrite_logprob_query.register(Switch)
def rewrite_switch_logprob(op, fgraph, query, **kwargs):
    from pymc.logprob.switch import rewrite_switch_non_overlapping

    rv, _ = query_parts(query)
    condition, *components = rv.owner.inputs
    if contains_random(condition):
        return rewrite_switch_non_overlapping(fgraph, query, **kwargs)
    if any(contains_random(comp) and comp.broadcastable != rv.broadcastable for comp in components):
        return None
    return rewrite_conditional_logprob(fgraph, query, **kwargs)


@infer_support_axes.register(IfElse)
def measure_ifelse(op, var):
    left = var.owner.inputs[1 + var.index]
    right = var.owner.inputs[1 + op.n_outs + var.index]
    if support_axes(left) != support_axes(right):
        raise UnsupportedObservation("IfElse branches have different measures")
    return support_axes(left)


@infer_support_axes.register(AdvancedSubtensor)
@infer_support_axes.register(Subtensor)
def measure_subtensor(op, var):
    from pytensor.scan.op import Scan

    base = var.owner.inputs[0]
    if isinstance(base.owner_op, Scan):
        return support_axes(base)
    components, _ = get_stack_mixture_vars(var.owner)
    if components is None:
        if support_axes(base) == ():
            return ()
        raise UnsupportedObservation("Indexing support dimensions needs an explicit measure")
    if any(support_axes(comp) for comp in components):
        raise UnsupportedObservation("Indexing support dimensions needs an explicit measure")
    return ()


@rewrite_logprob_query.register(AdvancedSubtensor)
@rewrite_logprob_query.register(Subtensor)
def rewrite_subtensor_logprob(op, fgraph, query, **kwargs):
    from pytensor.scan.op import Scan

    from pymc.logprob.scan import scan_slice_source

    rv, value = query_parts(query)
    node = rv.owner
    if isinstance(node.inputs[0].owner_op, Scan):
        source = scan_slice_source(node)
        if source is None:
            return None
        return [logprob_query(bind_value(fgraph, source, value))]
    components, join_axis = get_stack_mixture_vars(node)
    if components is None:
        return None
    indices = unflatten_index_variables(list(node.inputs[1:]), op.idx_list)
    if join_axis is None:
        if len(indices) != 1:
            return None
        (index,) = indices
    else:
        ndim = node.inputs[0].ndim
        axis = join_axis % ndim
        indices = (*indices, *(slice(None) for _ in range(ndim - len(indices))))
        index = indices[axis]
        if not all(component.broadcastable[axis] for component in components):
            return None
        component_indices = indices[:axis] + indices[axis + 1 :]
        components = [pt.squeeze(component, axis=axis) for component in components]
        if any(not isinstance(idx, slice) or idx != slice(None) for idx in component_indices):
            components = [rv_pull_down(component[component_indices]) for component in components]
    if not (
        isinstance(index, Variable)
        and (index.ndim == 0 or all(index.type.broadcastable))
        and index.dtype.startswith("int")
        and not contains_random(index)
    ):
        return None
    index = pt.as_tensor_variable(index)
    index_ndim = index.ndim
    index = pt.squeeze(index)
    if index_ndim:
        position = 0 if join_axis is None else sum(isinstance(idx, slice) for idx in indices[:axis])
        order = list(range(components[0].ndim))
        components = [
            comp.dimshuffle(order[:position] + ["x"] * index_ndim + order[position:])
            for comp in components
        ]
    outputs = conditioning_scope(fgraph, [rv])
    pending_layout = any(isinstance(out.owner_op, ValuedRV) for out in outputs)
    scope = FunctionGraph(outputs=[*outputs, *components], clone=False)
    branches = []
    for i in range(len(components)):
        branch, equiv = scope.clone_get_equiv(copy_inputs=False)
        chosen = branch.outputs[len(outputs) + i]
        for _ in components:
            branch.remove_output(len(outputs))
        selected = equiv[rv]
        bindings = [
            client.outputs[0]
            for client, index in branch.clients[selected]
            if isinstance(client.op, ValuedRV) and index == 0
        ]
        branch.replace(selected, chosen, reason="select indexed component", import_missing=True)
        bind_selected_values(branch, bindings)
        if pending_layout:
            derive_graph(branch, **kwargs)
        branches.append(branch.outputs)
    if len(branches) == 1:
        return {
            out: pt.stack([term])[index] for out, term in zip(outputs, branches[0], strict=True)
        }
    index = CheckAndRaise(IndexError, "mixture index out of bounds")(
        index, index >= -len(branches), index < len(branches)
    )
    index = index % len(branches)
    # Inverses can make parameters invalid in inactive branches; evaluate only the selected density.
    terms = branches[-1]
    for i in reversed(range(len(branches) - 1)):
        terms = ifelse(pt.eq(index, i), branches[i], terms)
    return dict(zip(outputs, terms, strict=True))


@rewrite_logprob_query.register(IfElse)
def rewrite_ifelse_logprob(op, fgraph, query, **kwargs):
    return rewrite_conditional_logprob(fgraph, query, **kwargs)


def rewrite_conditional_logprob(fgraph, query, **kwargs):
    branch_node = query_parts(query)[0].owner
    guard = branch_node.inputs[0]
    if contains_random(guard):
        return None
    if guard.ndim and all(guard.broadcastable):
        # Elemwise pads a scalar Switch condition to its components' rank.
        # It still selects one whole branch, including scalar event densities.
        guard = pt.squeeze(guard)
    outputs = conditioning_scope(fgraph, branch_node.outputs)
    pending_layout = any(isinstance(out.owner_op, ValuedRV) for out in outputs)
    scope = FunctionGraph(outputs=outputs, clone=False)
    if guard.ndim:
        components = branch_node.inputs[1:]
        if any(support_axes(component) for component in components):
            return None
        sources = [set(density_sources(component)) for component in components]
        exclusive = set.union(*sources) - set.intersection(*sources)
        if other_query_uses(fgraph, exclusive, {query}):
            return None
        if len(outputs) > 1 or set.intersection(*sources):
            # Per-entry guards only compose with factors that preserve those entries.
            if any(out.ndim != branch_node.outputs[0].ndim for out in outputs):
                return None
            for node in scope.toposort():
                if any(contains_random(out) for out in node.outputs) and not (
                    isinstance(node.op, Elemwise | ValuedRV | LogprobQuery)
                    or (isinstance(node.op, RandomVariable) and node.op.ndim_supp == 0)
                ):
                    return None
    # Each branch gets its own value bindings; its unresolved queries stay in the rewrite graph.
    branches = []
    for take_then in (True, False):
        branch, equiv = scope.clone_get_equiv(copy_inputs=False)
        branch_guard = equiv[branch_node].inputs[0]
        selected_bindings = []
        # Every IfElse using this guard belongs to the same conditioning scope.
        for node in reversed(branch.toposort()):
            if (
                (
                    isinstance(node.op, IfElse)
                    or (isinstance(node.op, Elemwise) and isinstance(node.op.scalar_op, Switch))
                )
                and node.inputs[0] is branch_guard
                and node in branch.apply_nodes
            ):
                offset = 1 if take_then else 1 + len(node.outputs)
                for out, chosen in zip(
                    node.outputs, node.inputs[offset : offset + len(node.outputs)], strict=True
                ):
                    bindings = [
                        client.outputs[0]
                        for client, index in branch.clients[out]
                        if isinstance(client.op, ValuedRV) and index == 0
                    ]
                    selected_bindings.extend(bindings)
                    if isinstance(node.op, Elemwise):
                        chosen = pt.cast(chosen, out.dtype)
                        if chosen.broadcastable != out.broadcastable and bindings:
                            chosen = pt.broadcast_to(chosen, bindings[0].owner.inputs[1].shape)
                    branch.replace(
                        out, chosen, reason="select conditional branch", import_missing=True
                    )
        bind_selected_values(branch, selected_bindings)
        if pending_layout:
            # These observations cannot become typed queries until this branch
            # supplies their inverse values. Resolve the branch before building
            # the conditional expression, whose density output types are now known.
            derive_graph(branch, **kwargs)
        branches.append(branch.outputs)
    terms = (
        [pt.switch(guard, left, right) for left, right in zip(*branches, strict=True)]
        if guard.ndim
        else ifelse(guard, branches[0], branches[1])
    )
    return dict(zip(outputs, terms, strict=True))


def bind_selected_values(branch, bindings):
    # The selected source can also be a parameter of another variable in this branch.
    for binding in bindings:
        source, value = binding.owner.inputs
        if isinstance(source.owner_op, ValuedRV):
            continue
        branch.replace(
            binding,
            bind_value(branch, source, value),
            reason="bind selected branch",
            import_missing=True,
        )


def conditioning_scope(fgraph, sources):
    """Find terms connected by unbound randomness; independent factors need no branching."""
    pending_roots = {out for out in fgraph.outputs if isinstance(out.owner_op, ValuedRV)}

    def inputs(var):
        if var in pending_roots:
            return var.owner.inputs
        if var.owner is None or isinstance(var.owner_op, ValuedRV):
            return ()
        if isinstance(var.owner_op, LogprobQuery):
            return query_parts(var.owner)
        return [inp for inp in var.owner.inputs if not isinstance(inp.type, RandomType)]

    def dependencies(roots):
        result = set()
        for var in walk(roots, inputs):
            if contains_random(var):
                result.update((var, var.owner))
        return result

    connected = dependencies(sources)
    terms = [node.outputs[0] for node in fgraph.toposort() if isinstance(node.op, LogprobQuery)]
    terms.extend(out for out in fgraph.outputs if out in pending_roots or contains_random(out))
    remaining = {term: dependencies([term]) for term in terms}
    selected = set()
    while remaining:
        found = [out for out, deps in remaining.items() if connected.intersection(deps)]
        if not found:
            break
        for out in found:
            connected.update(remaining.pop(out))
            selected.add(out)
    return list(dict.fromkeys(term for term in terms if term in selected))
