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

from copy import copy
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
from pymc.logprob.transforms import RVTransform


class TransformedVariable(Op):
    """A no-op that identifies a transform and its un-transformed input."""

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


transformed_variable = TransformedVariable()


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

    new_op = _create_transformed_rv_op(node.op, transforms)
    # Create a new `Apply` node and outputs
    trans_node = node.clone()
    trans_node.op = new_op

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    for rv_var, value_var, transform in zip(rv_vars, value_vars, transforms):
        rv_var_out_idx = node.outputs.index(rv_var)

        if transform is None:
            continue

        new_value_var = transformed_variable(
            transform.backward(value_var, *trans_node.inputs), value_var
        )

        if value_var.name and getattr(transform, "name", None):
            new_value_var.name = f"{value_var.name}_{transform.name}"

        rv_map_feature.update_rv_maps(rv_var, new_value_var, trans_node.outputs[rv_var_out_idx])

    return trans_node.outputs


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

    new_op = _create_transformed_rv_op(node.op, transforms)
    trans_node = node.clone()
    trans_node.op = new_op

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    for rv_var, value_var, transform in zip(rv_vars, value_vars, transforms):
        rv_var_out_idx = node.outputs.index(rv_var)

        if transform is None:
            continue

        # We access the original value variable and apply the transform to that
        original_value_var = rv_map_feature.original_values[value_var]
        trans_original_value_var = transform.backward(original_value_var, *trans_node.inputs)

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
        trans_value_var = value_var.owner.clone_with_new_inputs(
            inputs=[trans_original_value_var] + value_var.owner.inputs[1:]
        ).default_output()

        new_value_var = transformed_variable(trans_value_var, original_value_var)

        if value_var.name and getattr(transform, "name", None):
            new_value_var.name = f"{value_var.name}_{transform.name}"

        rv_map_feature.update_rv_maps(rv_var, new_value_var, trans_node.outputs[rv_var_out_idx])

    return trans_node.outputs


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
        values_to_transforms: Dict[TensorVariable, Union[RVTransform, None]],
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


def _create_transformed_rv_op(
    rv_op: Op,
    transforms: Union[RVTransform, Sequence[Union[None, RVTransform]]],
    *,
    cls_dict_extra: Optional[Dict] = None,
) -> Op:
    """Create a new transformed variable instance given a base `RandomVariable` `Op`.

    This will essentially copy the `type` of the given `Op` instance, create a
    copy of said `Op` instance and change it's `type` to the new one.

    In the end, we have an `Op` instance that will map to a `RVTransform` while
    also behaving exactly as it did before.

    Parameters
    ----------
    rv_op
        The `RandomVariable` for which we want to construct a `TransformedRV`.
    transform
        The `RVTransform` for `rv_op`.
    cls_dict_extra
        Additional class members to add to the constructed `TransformedRV`.

    """

    if not isinstance(transforms, Sequence):
        transforms = (transforms,)

    trans_names = [
        getattr(transform, "name", "transformed") if transform is not None else "None"
        for transform in transforms
    ]
    rv_op_type = type(rv_op)
    rv_type_name = rv_op_type.__name__
    cls_dict = rv_op_type.__dict__.copy()
    rv_name = cls_dict.get("name", "")
    if rv_name:
        cls_dict["name"] = f"{rv_name}_{'_'.join(trans_names)}"
    cls_dict["transforms"] = transforms

    if cls_dict_extra is not None:
        cls_dict.update(cls_dict_extra)

    new_op_type = type(f"Transformed{rv_type_name}", (rv_op_type,), cls_dict)

    MeasurableVariable.register(new_op_type)

    @_logprob.register(new_op_type)
    def transformed_logprob(op, values, *inputs, use_jacobian=True, **kwargs):
        """Compute the log-likelihood graph for a `TransformedRV`.

        We assume that the value variable was back-transformed to be on the natural
        support of the respective random variable.
        """
        logprobs = _logprob(rv_op, values, *inputs, **kwargs)

        if not isinstance(logprobs, Sequence):
            logprobs = [logprobs]

        # Handle jacobian
        assert len(values) == len(logprobs) == len(op.transforms)
        logprobs_jac = []
        for value, transform, logp in zip(values, op.transforms, logprobs):
            if transform is None:
                logprobs_jac.append(logp)
                continue

            assert isinstance(value.owner.op, TransformedVariable)
            original_forward_value = value.owner.inputs[1]
            log_jac_det = transform.log_jac_det(original_forward_value, *inputs).copy()
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
            else:
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

    new_op = copy(rv_op)
    new_op.__class__ = new_op_type

    return new_op


@node_rewriter([TransformedVariable])
def remove_TransformedVariables(fgraph, node):
    if isinstance(node.op, TransformedVariable):
        return [node.inputs[0]]


cleanup_ir_rewrites_db.register(
    "remove_TransformedVariables",
    remove_TransformedVariables,
    "cleanup",
    "transform",
)
