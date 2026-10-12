#   Copyright 2024 - present The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
"""Density queries, conditioning boundaries, and joint producer dispatch.

Tensor rules propagate observations upstream. Primitive density rules receive
all observations of a producer together. Recognized operations own their final
metadata; unresolved requests wait for observations and upstream recognition.
"""

from collections import deque
from contextvars import ContextVar
from functools import singledispatch

from pytensor.configdefaults import config
from pytensor.graph.features import Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import graph_replace
from pytensor.graph.traversal import ancestors, graph_inputs, walk
from pytensor.ifelse import IfElse
from pytensor.scan.op import Scan
from pytensor.tensor.random.type import RandomType

from pymc.logprob.abstract import (
    LogprobQuery,
    MeasurableOp,
    MeasurableTensorOp,
    ValuedRV,
    _logprob,
    logprob_query,
    supp_axes,
    valued_rv,
)


class UnsupportedObservation(NotImplementedError):
    """The available rules cannot derive this observation's density."""


_ir_rewriter = ContextVar("density_ir_rewriter", default=None)


def is_random_source(var):
    return (
        not isinstance(var.type, RandomType)
        and isinstance(var.owner_op, MeasurableOp)
        and not isinstance(var.owner_op, MeasurableTensorOp)
    )


def random_inputs(var):
    if var.owner is None or isinstance(var.owner_op, ValuedRV | LogprobQuery):
        return ()
    return tuple(inp for inp in var.owner.inputs if not isinstance(inp.type, RandomType))


def contains_random(var):
    """Whether a variable depends on randomness outside conditioning boundaries."""
    memo = {}
    pending = [(var, False)]
    while pending:
        current, visited = pending.pop()
        if current in memo:
            continue
        if isinstance(current.type, RandomType) or isinstance(current.owner_op, ValuedRV):
            memo[current] = False
        elif is_random_source(current):
            memo[current] = True
        elif isinstance(current.owner_op, Scan):
            mapping = current.owner.op.get_oinp_iinp_iout_oout_mappings()
            inners = mapping["inner_out_from_outer_out"].get(current.index, ())
            memo[current] = any(contains_random(current.owner.op.inner_outputs[i]) for i in inners)
        elif visited:
            memo[current] = any(memo[inp] for inp in random_inputs(current))
        else:
            pending.append((current, True))
            pending.extend((inp, False) for inp in random_inputs(current) if inp not in memo)
    return memo[var]


def density_sources(var):
    """Stochastic producers of an expression, excluding their distribution parameters."""

    def inputs(current):
        return () if is_random_source(current) else random_inputs(current)

    return [current for current in walk([var], inputs) if is_random_source(current)]


@singledispatch
def infer_support_axes(op, var):
    axes = supp_axes(var)
    if axes is None:
        raise UnsupportedObservation(f"Support axes are unknown for {op}")
    return tuple(axis % var.ndim for axis in axes)


@singledispatch
def infer_density_ndim(op, var):
    axes = support_axes(var)
    if axes is None:
        raise UnsupportedObservation(f"Density rank is unknown for {op}")
    return var.ndim - len(axes)


def support_axes(var):
    if isinstance(var.owner_op, ValuedRV):
        return support_axes(var.owner.inputs[0])
    if var.owner is None:
        return ()
    op = var.owner.op
    if isinstance(op, MeasurableTensorOp) and op.supp_axes is not None:
        axes = op.supp_axes[var.index]
        return None if axes is None else tuple(axis % var.ndim for axis in axes)
    return infer_support_axes(op, var)


def density_ndim(var):
    if isinstance(var.owner_op, ValuedRV):
        return density_ndim(var.owner.inputs[0])
    if var.owner is None:
        return var.ndim
    op = var.owner.op
    if isinstance(op, MeasurableTensorOp) and op.logp_ndims is not None:
        ndim = op.logp_ndims[var.index]
        if ndim is not None:
            return ndim
    return infer_density_ndim(op, var)


def query_parts(query):
    return query.inputs[0].owner.inputs


def bind_value(fgraph, rv, value):
    """Retain a density source and condition its other uses on an inferred value."""
    if isinstance(rv.owner_op, ValuedRV):
        return rv
    pending = deque([(rv, value)])
    bound = {}
    while pending:
        source, source_value = pending.popleft()
        if source in bound:
            continue
        clients = list(fgraph.clients.get(source, ()))
        binding = next(
            (
                client.outputs[0]
                for client, index in clients
                if isinstance(client.op, ValuedRV) and index == 0
            ),
            None,
        )
        if binding is None:
            binding = valued_rv(source, source_value)
        bound[source] = binding
        replacement = source.type.filter_variable(binding, allow_convert=True)
        affected = set()
        for client, index in clients:
            if isinstance(client.op, ValuedRV) and index == 0:
                continue
            fgraph.change_node_input(
                client, index, replacement, reason="infer observation", import_missing=True
            )
            affected.add(client)
        # A recognized operation keeps its density interpretation. When all
        # its inputs acquire values, publish the resulting value as a separate
        # conditioning boundary instead of reclassifying that operation.
        for client in affected:
            if isinstance(client.op, MeasurableTensorOp) and not any(
                contains_random(inp) for inp in client.inputs
            ):
                if all(
                    any(
                        isinstance(user.op, ValuedRV) and index == 0
                        for user, index in fgraph.clients.get(out, ())
                    )
                    for out in client.outputs
                ):
                    continue
                values = client.op.base_op.make_node(*explicit_values(client.inputs)).outputs
                pending.extend(zip(client.outputs, values, strict=True))
    return bound[rv]


def output_queries(fgraph, producer):
    """Collect direct requests through each output's conditioning boundaries."""
    queries = {}
    for rv in producer.outputs:
        for binding, index in fgraph.clients.get(rv, ()):
            if not isinstance(binding.op, ValuedRV) or index != 0:
                continue
            for node, _ in fgraph.clients.get(binding.outputs[0], ()):
                if not isinstance(node.op, LogprobQuery):
                    continue
                if rv in queries and queries[rv] is not node:
                    raise UnsupportedObservation("Multiple observations of the same random output")
                queries[rv] = node
    return queries


def other_query_uses(fgraph, sources, excluded):
    sources = set(sources)
    pending_roots = {out for out in fgraph.outputs if isinstance(out.owner_op, ValuedRV)}

    def inputs(var):
        if var in pending_roots:
            return var.owner.inputs
        if isinstance(var.owner_op, LogprobQuery):
            return () if var.owner in excluded else query_parts(var.owner)
        return random_inputs(var)

    return any(var in sources for var in walk(fgraph.outputs, inputs))


def query_at_values(rv, values, **kwargs):
    """Derive one density template and substitute alternative observation values.

    Distribution parameters remain inputs of this scope, preserving their
    identity so later observations in the enclosing graph can condition them.
    """
    sources = density_sources(rv)
    source_set = set(sources)
    if len({source.owner for source in sources}) != len(sources) or any(
        var in source_set
        for source in sources
        for var in walk(random_inputs(source), random_inputs)
    ):
        raise UnsupportedObservation(
            "Alternative inverses of coupled density sources need a joint inverse rule"
        )
    boundaries = list(
        dict.fromkeys(
            [
                *graph_inputs([rv]),
                *(inp for source in sources for inp in source.owner.inputs),
            ]
        )
    )
    placeholders = {boundary: boundary.clone() for boundary in boundaries}
    graph = FunctionGraph(
        inputs=boundaries, outputs=[rv], clone=True, copy_inputs=False, memo=placeholders
    )
    value = rv.type()
    graph.replace(
        graph.outputs[0], logprob_query(valued_rv(graph.outputs[0], value)), import_missing=True
    )
    derive_graph(graph, ir_rewriter=_ir_rewriter.get(), **kwargs)
    parameters = {placeholders[boundary]: boundary for boundary in boundaries}
    return [
        graph_replace(graph.outputs[0], {**parameters, value: alternative}, strict=False)
        for alternative in values
    ]


def query_at_value(rv, value):
    return query_at_values(rv, [value])[0]


def explicit_values(variables):
    bindings = {
        var: var.owner.inputs[1]
        for var in ancestors(variables)
        if isinstance(var.owner_op, ValuedRV)
    }
    return graph_replace(variables, bindings, strict=False) if bindings else variables


@singledispatch
def rewrite_logprob_query(op, fgraph, query, **kwargs):
    rv, _ = query_parts(query)
    if not is_random_source(rv):
        return None
    producer = rv.owner
    queries = output_queries(fgraph, producer)
    missing = [
        out
        for out in producer.outputs
        if out not in queries and not isinstance(out.type, RandomType)
    ]
    if other_query_uses(fgraph, missing, set(queries.values())):
        return None
    outputs = [out for out in producer.outputs if out in queries]
    values = [query_parts(queries[out])[1] for out in outputs]
    terms = _logprob(op, explicit_values(values), *explicit_values(producer.inputs), **kwargs)
    if not isinstance(terms, list | tuple):
        terms = [terms]
    if len(terms) != len(outputs):
        raise ValueError(f"{op} returned {len(terms)} density terms for {len(outputs)} values")
    return {queries[out].outputs[0]: term for out, term in zip(outputs, terms, strict=True)}


class QueryQueue(Feature):
    """Schedule unresolved density requests; accepted metadata is never invalidated."""

    def __init__(self):
        self.tensor = deque()
        self.primitive = deque()
        self.queued = set()
        self.waiting = {}
        self.revision = 0

    def clone(self):
        return type(self)()

    def enqueue(self, node):
        if not isinstance(node.op, LogprobQuery) or node in self.queued:
            return
        queue = self.primitive if is_random_source(query_parts(node)[0]) else self.tensor
        queue.append(node)
        self.queued.add(node)

    def on_import(self, fgraph, node, reason):
        self.revision += 1
        self.enqueue(node)

    def on_change_input(self, fgraph, node, i, old, new, reason=None):
        self.revision += 1
        self.enqueue(node)

    def on_prune(self, fgraph, node, reason):
        self.waiting.pop(node, None)

    def resume(self):
        for node in self.waiting:
            self.enqueue(node)
        self.waiting.clear()

    def pop(self):
        node = (self.tensor or self.primitive).popleft()
        self.queued.discard(node)
        return node

    def __bool__(self):
        return bool(self.tensor or self.primitive)


def derive_graph(fgraph, ir_rewriter=None, **kwargs):
    """Resolve each request once, deferring requests whose prerequisites are pending."""
    from pymc.logprob.rewriting import logprob_rewrites_basic_query, logprob_rewrites_db

    if ir_rewriter is None:
        ir_rewriter = _ir_rewriter.get()
    if ir_rewriter is None:
        ir_rewriter = logprob_rewrites_db.query(logprob_rewrites_basic_query)
    queue = QueryQueue()
    fgraph.attach_feature(queue)
    for node in reversed(fgraph.toposort()):
        queue.enqueue(node)
    pending_outputs = {
        i: "" for i, out in enumerate(fgraph.outputs) if isinstance(out.owner_op, ValuedRV)
    }
    rewriter_token = _ir_rewriter.set(ir_rewriter)
    recognition_revision = -1
    try:
        while queue or queue.waiting or pending_outputs:
            for index in tuple(pending_outputs):
                binding = fgraph.outputs[index]
                try:
                    request = logprob_query(binding)
                except UnsupportedObservation as error:
                    pending_outputs[index] = str(error)
                    continue
                # Only this output port changes from an observation to its
                # density request; existing tensor consumers keep their types.
                output_node, _ = fgraph.get_output_client(index)
                fgraph.change_node_input(
                    output_node,
                    0,
                    request,
                    check=False,
                    import_missing=True,
                    reason="resolve density layout",
                )
                del pending_outputs[index]
            if not queue:
                if queue.revision != recognition_revision and (
                    pending_outputs
                    or any(not is_random_source(query_parts(node)[0]) for node in queue.waiting)
                ):
                    # Only unresolved operations participate in recognition. New
                    # bindings can make an input a parameter and unlock a rule.
                    ir_rewriter.rewrite(fgraph)
                    recognition_revision = queue.revision
                    queue.resume()
                    continue
                detail = "; ".join(
                    dict.fromkeys(
                        msg for msg in (*queue.waiting.values(), *pending_outputs.values()) if msg
                    )
                )
                raise UnsupportedObservation(
                    detail
                    or "No applicable density rule; a shared unvalued source requires marginalization"
                )
            query = queue.pop()
            if query not in fgraph.apply_nodes:
                continue
            rv, _ = query_parts(query)
            error = ""
            try:
                if not contains_random(rv):
                    from pymc.logprob.utils import DiracDelta

                    result = [_logprob(DiracDelta(), [query_parts(query)[1]], rv)]
                elif isinstance(rv.owner_op, MeasurableOp | IfElse):
                    # Selecting a known IfElse branch is structural: a typed
                    # output query can select it without committing metadata
                    # for other outputs whose layouts are still unresolved.
                    result = rewrite_logprob_query(rv.owner.op, fgraph, query, **kwargs)
                else:
                    result = None
            except UnsupportedObservation as exc:
                error = str(exc)
                result = None
            if result is None:
                queue.waiting[query] = error
                continue
            replacements = (
                result
                if isinstance(result, dict)
                else dict(zip(query.outputs, result, strict=True))
            )
            for old, new in replacements.items():
                new = new.astype(config.floatX)
                pending_indices = [
                    index for index in pending_outputs if fgraph.outputs[index] is old
                ]
                if pending_indices:
                    # A conditional scope can resolve an untyped observation
                    # directly to its density. Its other tensor consumers still
                    # need the observation boundary, not this density tensor.
                    for index in pending_indices:
                        output_node, _ = fgraph.get_output_client(index)
                        fgraph.change_node_input(
                            output_node,
                            0,
                            new,
                            check=False,
                            import_missing=True,
                            reason="derive pending density",
                        )
                        del pending_outputs[index]
                    continue
                fgraph.replace(old, new, reason="derive density", import_missing=True)
            queue.resume()
        for node in reversed(fgraph.toposort()):
            if isinstance(node.op, ValuedRV) and node in fgraph.apply_nodes:
                fgraph.replace(
                    node.outputs[0],
                    node.inputs[1],
                    reason="resolve observation",
                    import_missing=True,
                )
    finally:
        _ir_rewriter.reset(rewriter_token)
        fgraph.remove_feature(queue)
