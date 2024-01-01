#   Copyright 2024 The PyMC Developers
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

from typing import Dict, Optional, Sequence, Union

import numpy as np

from pytensor.gradient import DisconnectedType
from pytensor.graph import Apply, Op
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import GraphRewriter, in2out, node_rewriter
from pytensor.scan.op import Scan
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import MeasurableVariable, _logprob
from pymc.logprob.rewriting import PreserveRVMappings, cleanup_ir_rewrites_db
from pymc.logprob.transforms import Transform


class TransformedValue(Op):
    """A no-op that pairs the original value with its transformed version.

    This is introduced by the `TransformValuesRewrite`
    """

    view_map = {0: [0]}

    def make_node(self, tran_value: TensorVariable, value: TensorVariable):
        return Apply(self, [tran_value, value], [tran_value.type()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError("These `Op`s should be removed from graphs used for computation.")

    def connection_pattern(self, node):
        return [[True], [False]]

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]

    def grad(self, args, g_outs):
        return g_outs[0], DisconnectedType()()


transformed_value = TransformedValue()


class TransformedValueRV(Op):
    """A no-op that identifies RVs whose values were transformed.

    This is introduced by the `TransformValuesRewrite`
    """

    view_map = {0: [0]}

    __props__ = ("transforms",)

    def __init__(self, transforms: Sequence[Transform]):
        self.transforms = tuple(transforms)
        super().__init__()

    def make_node(self, *rv_outputs):
        return Apply(self, rv_outputs, [out.type() for out in rv_outputs])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            "`TransformedRV` `Op`s should be removed from graphs used for computation."
        )

    def connection_pattern(self, node):
        return [[True] for _ in node.outputs]

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes


MeasurableVariable.register(TransformedValueRV)


@_logprob.register(TransformedValueRV)
def transformed_value_logprob(op, values, *rv_outs, use_jacobian=True, **kwargs):
    """Compute the log-probability graph for a `TransformedRV`.

    This is introduced by the `TransformValuesRewrite`
    """
    rv_op = rv_outs[0].owner.op
    rv_inputs = rv_outs[0].owner.inputs
    logprobs = _logprob(rv_op, values, *rv_inputs, **kwargs)

    if not isinstance(logprobs, Sequence):
        logprobs = [logprobs]

    # Handle jacobian
    assert len(values) == len(logprobs) == len(op.transforms)
    logprobs_jac = []
    for value, transform, logp in zip(values, op.transforms, logprobs):
        if transform is None:
            logprobs_jac.append(logp)
            continue

        assert isinstance(value.owner.op, TransformedValue)
        original_forward_value = value.owner.inputs[1]
        log_jac_det = transform.log_jac_det(original_forward_value, *rv_inputs).copy()
        # The jacobian determinant has less dims than the logp
        # when a multivariate transform (like Simplex or Ordered) is applied to univariate distributions.
        # In this case we have to reduce the last logp dimensions, as they are no longer independent
        if log_jac_det.ndim < logp.ndim:
            diff_ndims = logp.ndim - log_jac_det.ndim
            logp = logp.sum(axis=np.arange(-diff_ndims, 0))
        # This case is sometimes, but not always, trivial to accomodate depending on the "space rank" of the
        # multivariate distribution. See https://proceedings.mlr.press/v130/radul21a.html
        elif log_jac_det.ndim > logp.ndim:
            raise NotImplementedError(
                f"Univariate transform {transform} cannot be applied to multivariate {rv_op}"
            )
        # Check there is no broadcasting between logp and jacobian
        if logp.type.broadcastable != log_jac_det.type.broadcastable:
            raise ValueError(
                f"The logp of {rv_op} and log_jac_det of {transform} are not allowed to broadcast together. "
                "There is a bug in the implementation of either one."
            )

        if use_jacobian:
            if value.name:
                log_jac_det.name = f"{value.name}_jacobian"
            logprobs_jac.append(logp + log_jac_det)
        else:
            # We still want to use the reduced logp, even though the jacobian isn't included
            logprobs_jac.append(logp)

    return logprobs_jac


@node_rewriter(tracks=None)
def transform_values(fgraph: FunctionGraph, node: Apply) -> Optional[list[Apply]]:
    """Apply transforms to value variables.

    It is assumed that the input value variables correspond to forward
    transformations, usually chosen in such a way that the values are
    unconstrained on the real line.

    For example, if ``Y = halfnormal(...)``, we assume the respective value
    variable is specified on the log scale and back-transform it to obtain
    ``Y`` on the natural scale.
    """

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)
    values_to_transforms: Optional[TransformValuesMapping] = getattr(
        fgraph, "values_to_transforms", None
    )

    if rv_map_feature is None or values_to_transforms is None:
        return None  # pragma: no cover

    rv_vars = []
    value_vars = []

    for out in node.outputs:
        value = rv_map_feature.rv_values.get(out, None)
        if value is None:
            continue
        rv_vars.append(out)
        value_vars.append(value)

    if not value_vars:
        return None

    transforms = [values_to_transforms.get(value_var, None) for value_var in value_vars]

    if all(transform is None for transform in transforms):
        return None

    transformed_rv_op = TransformedValueRV(transforms)
    # Clone outputs so that rewrite doesn't reference original variables circularly
    cloned_outputs = node.clone().outputs
    transformed_rv_node = transformed_rv_op.make_node(*cloned_outputs)

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    for rv_var, value_var, transform in zip(rv_vars, value_vars, transforms):
        rv_var_out_idx = node.outputs.index(rv_var)

        if transform is None:
            continue

        new_value_var = transformed_value(
            transform.backward(value_var, *node.inputs),
            value_var,
        )

        if value_var.name and getattr(transform, "name", None):
            new_value_var.name = f"{value_var.name}_{transform.name}"

        rv_map_feature.update_rv_maps(
            rv_var, new_value_var, transformed_rv_node.outputs[rv_var_out_idx]
        )

    return transformed_rv_node.outputs


@node_rewriter(tracks=[Scan])
def transform_scan_values(fgraph: FunctionGraph, node: Apply) -> Optional[list[Apply]]:
    """Apply transforms to Scan value variables.

    This specialized rewrite is needed because Scan replaces the original value variables
    by a more complex graph. We want to apply the transform to the original value variable
    in this subgraph, leaving the rest intact
    """

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)
    values_to_transforms: Optional[TransformValuesMapping] = getattr(
        fgraph, "values_to_transforms", None
    )

    if rv_map_feature is None or values_to_transforms is None:
        return None  # pragma: no cover

    rv_vars = []
    value_vars = []

    for out in node.outputs:
        value = rv_map_feature.rv_values.get(out, None)
        if value is None:
            continue
        rv_vars.append(out)
        value_vars.append(value)

    if not value_vars:
        return None

    transforms = [
        values_to_transforms.get(rv_map_feature.original_values[value_var], None)
        for value_var in value_vars
    ]

    if all(transform is None for transform in transforms):
        return None

    transformed_rv_op = TransformedValueRV(transforms)
    # Clone outputs so that rewrite doesn't reference original variables circularly
    cloned_outputs = node.clone().outputs
    transformed_rv_node = transformed_rv_op.make_node(*cloned_outputs)

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    for rv_var, value_var, transform in zip(rv_vars, value_vars, transforms):
        rv_var_out_idx = node.outputs.index(rv_var)

        if transform is None:
            continue

        # We access the original value variable and apply the transform to that
        original_value_var = rv_map_feature.original_values[value_var]
        trans_original_value_var = transform.backward(
            original_value_var, *transformed_rv_node.inputs
        )

        # We then replace the reference to the original value variable in the scan value
        # variable by the back-transform projection computed above

        # The first input corresponds to the original value variable. We are careful to
        # only clone_replace that part of the graph, as we don't want to break the
        # mappings between other rvs that are likely to be present in the rest of the
        # scan value variable graph
        # TODO: Is it true that the original value only appears in the first input
        #  and that no other RV can appear there?
        (trans_original_value_var,) = clone_replace(
            (value_var.owner.inputs[0],),
            replace={original_value_var: trans_original_value_var},
        )
        transformed_value_var = value_var.owner.clone_with_new_inputs(
            inputs=[trans_original_value_var] + value_var.owner.inputs[1:]
        ).default_output()

        new_value_var = transformed_value(transformed_value_var, original_value_var)

        if value_var.name and getattr(transform, "name", None):
            new_value_var.name = f"{value_var.name}_{transform.name}"

        rv_map_feature.update_rv_maps(
            rv_var, new_value_var, transformed_rv_node.outputs[rv_var_out_idx]
        )

    return transformed_rv_node.outputs


class TransformValuesMapping(Feature):
    r"""A `Feature` that maintains a map between value variables and their transforms."""

    def __init__(self, values_to_transforms):
        self.values_to_transforms = values_to_transforms.copy()

    def on_attach(self, fgraph):
        if hasattr(fgraph, "values_to_transforms"):
            raise AlreadyThere()

        fgraph.values_to_transforms = self.values_to_transforms


class TransformValuesRewrite(GraphRewriter):
    r"""Transforms value variables according to a map."""

    transform_rewrite = in2out(transform_values, ignore_newtrees=True)
    scan_transform_rewrite = in2out(transform_scan_values, ignore_newtrees=True)

    def __init__(
        self,
        values_to_transforms: Dict[TensorVariable, Union[Transform, None]],
    ):
        """
        Parameters
        ----------
        values_to_transforms
            Mapping between value variables and their transformations.  Each
            value variable can be assigned one of `RVTransform`, or ``None``.
            If a transform is not specified for a specific value variable it will
            not be transformed.

        """

        self.values_to_transforms = values_to_transforms

    def add_requirements(self, fgraph):
        values_transforms_feature = TransformValuesMapping(self.values_to_transforms)
        fgraph.attach_feature(values_transforms_feature)

    def apply(self, fgraph: FunctionGraph):
        self.transform_rewrite.rewrite(fgraph)
        self.scan_transform_rewrite.rewrite(fgraph)


@node_rewriter([TransformedValue])
def remove_TransformedValues(fgraph, node):
    return [node.inputs[0]]


@node_rewriter([TransformedValueRV])
def remove_TransformedValueRVs(fgraph, node):
    return node.inputs


cleanup_ir_rewrites_db.register(
    "remove_TransformedValues",
    remove_TransformedValues,
    "cleanup",
    "transform",
)


cleanup_ir_rewrites_db.register(
    "remove_TransformedValueRVs",
    remove_TransformedValueRVs,
    "cleanup",
    "transform",
)
