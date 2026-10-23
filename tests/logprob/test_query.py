#   Copyright 2024 - present The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.graph import Apply, Op
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.traversal import ancestors
from pytensor.ifelse import ifelse

from pymc.logprob import query, transforms
from pymc.logprob.abstract import (
    MeasurableOp,
    MeasurableTensorOp,
    ValuedRV,
    _logprob,
    logprob_query,
    valued_rv,
)
from pymc.logprob.basic import conditional_logp
from pymc.logprob.measurable import find_measurable_tensor, measurable_node
from pymc.logprob.query import (
    UnsupportedObservation,
    bind_value,
    contains_random,
    density_ndim,
    derive_graph,
    infer_support_axes,
    query_at_values,
    support_axes,
)
from pymc.logprob.rewriting import construct_ir_fgraph
from pymc.testing import assert_no_rvs


def test_intermediate_metadata_is_reused(monkeypatch):
    x = pt.random.normal(size=(2, 5))
    observed = pt.clip(x, -1, 1).T
    graph = construct_ir_fgraph({observed: observed.type()})
    rv = graph.outputs[0].owner.inputs[0]
    assert isinstance(rv.owner.op, MeasurableTensorOp)

    def unexpected_inference(*args):
        raise AssertionError("Stored metadata should be read without rederivation")

    monkeypatch.setattr(query, "infer_support_axes", unexpected_inference)
    monkeypatch.setattr(query, "infer_density_ndim", unexpected_inference)
    for _ in range(10):
        assert support_axes(rv) == ()
        assert density_ndim(rv) == 2


def test_metadata_survives_native_op_reconstruction():
    source = pt.random.normal(size=(2, 3))
    node = measurable_node(pt.max(source, axis=1).owner)
    op = node.op
    assert isinstance(op, MeasurableTensorOp)
    assert op.logp_ndims == (1,)
    assert op.supp_axes == ((),)

    cloned = op.clone()
    assert cloned is not op
    assert cloned == op
    assert hash(cloned) == hash(op)
    assert cloned.metadata is op.metadata
    rebuilt = cloned.make_node(source)
    assert rebuilt.op.metadata == op.metadata
    assert density_ndim(rebuilt.outputs[0]) == 1

    # Max.make_node normalizes axes by cloning its Op. Changed native
    # parameters must establish new metadata, not carry the old rank along.
    changed = op.clone(axis=None)
    assert not isinstance(changed, MeasurableTensorOp)
    reduced = measurable_node(changed.make_node(source))
    assert isinstance(reduced.op, MeasurableTensorOp)
    assert reduced.op.logp_ndims == (0,)
    assert density_ndim(reduced.outputs[0]) == 0
    assert op.logp_ndims == (1,)

    # Parameter/value substitution preserves the declared interpretation.
    reconstructed = op.make_node(pt.matrix("known"))
    assert reconstructed.op is op
    assert reconstructed.op.metadata is op.metadata


def test_recognition_is_final_across_cloning_and_parameter_substitution():
    calls = []

    class Identity(Op):
        def make_node(self, x):
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = inputs[0]

    @infer_support_axes.register(Identity)
    def identity_axes(op, var):
        calls.append(var)
        return support_axes(var.owner.inputs[0])

    source = pt.random.normal(size=3)
    graph = FunctionGraph(outputs=[Identity()(Identity()(source))], clone=False)
    recognize = in2out(find_measurable_tensor)
    recognize.rewrite(graph)
    assert len(calls) == 2
    assert density_ndim(graph.outputs[0]) == 1
    recognize.rewrite(graph)
    assert len(calls) == 2

    other = graph.clone()
    original_op = graph.outputs[0].owner.op
    recognize.rewrite(other)
    assert len(calls) == 2
    # A structural change creates a fresh native operation and is recognized
    # separately. Existing nodes retain their own interpretation.
    replacement = pt.random.multivariate_normal(np.zeros(3), np.eye(3))
    new_node = measurable_node(original_op.base_op.make_node(replacement))
    assert support_axes(new_node.outputs[0]) == (0,)
    assert density_ndim(new_node.outputs[0]) == 0
    assert density_ndim(other.outputs[0]) == 1
    assert other.outputs[0].owner.op is original_op
    assert original_op.logp_ndims == (1,)


def test_known_values_do_not_reclassify_recognized_nodes():
    source = pt.random.normal(size=3)
    observed = source.dimshuffle("x", 0) + pt.random.normal(size=(2, 3))
    graph = FunctionGraph(outputs=[valued_rv(observed, observed.type())], clone=False)
    recognize = in2out(find_measurable_tensor)
    recognize.rewrite(graph)
    observed = graph.outputs[0].owner.inputs[0]
    dimshuffle = observed.owner.inputs[0]
    op = dimshuffle.owner.op
    assert isinstance(op, MeasurableTensorOp)
    bind_value(graph, source, source.type("value"))
    boundary = observed.owner.inputs[0]
    assert isinstance(boundary.owner.op, ValuedRV)
    assert boundary.owner.inputs[0] is dimshuffle
    assert dimshuffle.owner.op is op
    assert op.logp_ndims == (2,)
    assert not contains_random(boundary)
    recognize.rewrite(graph)
    assert isinstance(graph.outputs[0].owner.inputs[0].owner.op, MeasurableTensorOp)
    assert dimshuffle.owner.op is op


def test_lowering_reuses_the_selected_transform(monkeypatch):
    source = pt.random.normal()
    shift = pt.scalar("shift")
    observed = pt.exp(source + shift)
    value = pt.scalar("value")
    graph = construct_ir_fgraph({observed: value})
    binding = graph.outputs[0]
    rv = binding.owner.inputs[0]
    assert rv.owner.op.metadata.transform is not None
    assert rv.owner.inputs[0].owner.op.metadata.transform is not None
    graph.replace(binding, logprob_query(binding), import_missing=True)

    def unexpected_recognition(*args):
        raise AssertionError("A resolved transform must not be recognized again")

    monkeypatch.setattr(transforms, "select_transform", unexpected_recognition)
    derive_graph(graph)
    assert not any(isinstance(var.owner_op, query.LogprobQuery) for var in ancestors(graph.outputs))
    np.testing.assert_allclose(
        graph.outputs[0].eval({value: np.e, shift: 1.0}), -0.5 * np.log(2 * np.pi) - 1
    )


def test_request_waits_for_its_density_layout():
    # Both operands are vectors, but their densities have different ranks.
    # The sum's measured input is only known after exp(x) supplies x's value.
    x = pt.random.normal(size=3)
    z = pt.random.multivariate_normal(np.zeros(3), np.eye(3))
    y, w = x + z, pt.exp(x)
    y_value, w_value = pt.vectors("y_value", "w_value")
    terms = conditional_logp({y: y_value, w: w_value})
    assert terms[y_value].ndim == 0
    assert terms[w_value].ndim == 1
    point = {y_value: [1.0, 2.0, 3.0], w_value: np.exp([0.1, 0.2, 0.3])}
    expected = -0.5 * np.square(np.array([0.9, 1.8, 2.7])).sum() - 1.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(terms[y_value].eval(point), expected)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("selection", ["ifelse", "switch", "index_first", "index_last"])
def test_pending_density_layout_participates_in_conditional_scope(reverse, selection):
    x = pt.random.normal(size=3)
    z = pt.random.multivariate_normal(np.zeros(3), np.eye(3))
    y = x + z
    selector = pt.scalar("selector", dtype="bool")
    left, right = pt.exp(x), pt.exp(2 * x)
    if selection == "ifelse":
        w = ifelse(selector, left, right)
    elif selection == "switch":
        w = pt.switch(selector, left, right)
    else:
        index = pt.cast(selector, "int64")
        w = (
            pt.stack([right, left])[index]
            if selection == "index_first"
            else pt.stack([right, left], axis=1)[:, index]
        )
    y_value, w_value = pt.vectors("y_value", "w_value")
    pairs = [(y, y_value), (w, w_value)]
    terms = conditional_logp(dict(pairs[::-1] if reverse else pairs))
    assert terms[y_value].ndim == 0
    assert terms[w_value].ndim == 1
    for term in terms.values():
        assert_no_rvs(term)
    fn = pytensor.function([selector, y_value, w_value], [terms[y_value], terms[w_value]])
    point_y = np.array([1.0, 2.0, 3.0])
    point_w = np.exp([0.1, 0.2, 0.3])
    for choice, scale in [(True, 1), (False, 2)]:
        inverse = np.log(point_w) / scale
        expected_y = -0.5 * np.square(point_y - inverse).sum() - 1.5 * np.log(2 * np.pi)
        expected_w = -0.5 * np.square(inverse) - 0.5 * np.log(2 * np.pi)
        expected_w -= np.log(scale * point_w)
        actual_y, actual_w = fn(choice, point_y, point_w)
        np.testing.assert_allclose(actual_y, expected_y)
        np.testing.assert_allclose(actual_w, expected_w)


@pytest.mark.parametrize("reverse", [False, True])
def test_multi_output_recognition_waits_for_every_stochastic_layout(reverse):
    x = pt.random.normal(size=3)
    z1 = pt.random.multivariate_normal(np.zeros(3), np.eye(3))
    z2 = pt.random.multivariate_normal(np.zeros(3), 2 * np.eye(3))
    selector = pt.scalar("selector", dtype="bool")
    y, _ = ifelse(
        selector,
        [x + z1, pt.random.normal()],
        [x + z2, pt.random.normal()],
    )
    # The second output's known scalar layout must not freeze an unknown
    # layout for the first output into a partially recognized operation.
    assert measurable_node(y.owner) is y.owner
    y_value, w_value = pt.vectors("y_value", "w_value")
    pairs = [(y, y_value), (pt.exp(x), w_value)]
    terms = conditional_logp(dict(pairs[::-1] if reverse else pairs))
    for term in terms.values():
        assert_no_rvs(term)
    fn = pytensor.function([selector, y_value, w_value], terms[y_value])
    point_y = np.array([1.0, 2.0, 3.0])
    inverse = np.array([0.1, 0.2, 0.3])
    for choice, variance in [(True, 1), (False, 2)]:
        expected = -0.5 * np.square(point_y - inverse).sum() / variance
        expected -= 1.5 * np.log(2 * np.pi * variance)
        np.testing.assert_allclose(fn(choice, point_y, np.exp(inverse)), expected)


@pytest.mark.parametrize("transform", [pt.abs, lambda x: x**2])
def test_alternative_inverses_reject_coupled_sources(transform):
    x = pt.random.normal()
    y = pt.random.normal(x, 1)
    observed = transform(pt.stack([x, y]))
    with pytest.raises(UnsupportedObservation, match="coupled density sources"):
        conditional_logp({observed: pt.vector("value")})


@pytest.mark.parametrize("observe_all", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_ready_ifelse_output_can_select_unresolved_sibling(observe_all, reverse):
    x = pt.random.normal(size=3)
    z1 = pt.random.multivariate_normal(np.zeros(3), np.eye(3))
    z2 = pt.random.multivariate_normal(np.zeros(3), 2 * np.eye(3))
    selector = pt.scalar("selector", dtype="bool")
    w, y = ifelse(selector, [pt.exp(x), x + z1], [pt.exp(2 * x), x + z2])
    w_value, y_value = pt.vectors("w_value", "y_value")
    pairs = [(w, w_value)]
    if observe_all:
        pairs.append((y, y_value))
    terms = conditional_logp(dict(pairs[::-1] if reverse else pairs))
    for term in terms.values():
        assert_no_rvs(term)
    values = [w_value, y_value] if observe_all else [w_value]
    fn = pytensor.function([selector, *values], [terms[value] for value in values])
    point_w = np.exp([0.1, 0.2, 0.3])
    point_y = np.array([1.0, 2.0, 3.0])
    for choice, scale in [(True, 1), (False, 2)]:
        inverse = np.log(point_w) / scale
        point = [point_w, point_y] if observe_all else [point_w]
        actual = fn(choice, *point)
        expected_w = -0.5 * np.square(inverse) - 0.5 * np.log(2 * np.pi)
        expected_w -= np.log(scale * point_w)
        np.testing.assert_allclose(actual[0], expected_w)
        if observe_all:
            expected_y = -0.5 * np.square(point_y - inverse).sum() / scale
            expected_y -= 1.5 * np.log(2 * np.pi * scale)
            np.testing.assert_allclose(actual[1], expected_y)


def test_alternative_inverses_preserve_independent_sources():
    observed = pt.abs(pt.stack([pt.random.normal(-1, 1), pt.random.normal(2, 1)]))
    value = pt.vector("value")
    term = conditional_logp({observed: value})[value]
    assert_no_rvs(term)
    point = np.array([0.7, 1.2])
    means = np.array([-1.0, 2.0])
    expected = np.logaddexp(-0.5 * (point - means) ** 2, -0.5 * (-point - means) ** 2)
    expected -= 0.5 * np.log(2 * np.pi)
    np.testing.assert_allclose(term.eval({value: point}), expected)


def test_alternative_values_reuse_density_template():
    calls = []

    class Source(MeasurableOp, Op):
        ndim_supp = 0

        def make_node(self, parameter):
            return Apply(self, [parameter], [parameter.type()])

        def perform(self, node, inputs, outputs):
            raise AssertionError("The density should not sample")

    @_logprob.register(Source)
    def source_logp(op, values, parameter, **kwargs):
        calls.append(values)
        return -((values[0] - parameter) ** 2)

    parameter = pt.exp(pt.random.normal())
    value = pt.scalar("value")
    terms = query_at_values(Source()(parameter), [value, -value])
    assert len(calls) == 1
    assert parameter in set(ancestors(terms))
    np.testing.assert_allclose(terms[0].eval({parameter: 2.0, value: 1.0}), -1.0)
    np.testing.assert_allclose(terms[1].eval({parameter: 2.0, value: 1.0}), -9.0)
