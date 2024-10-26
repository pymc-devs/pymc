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

from collections.abc import Sequence

import numpy as np

from pytensor.graph import Apply, Op
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import GraphRewriter, in2out, node_rewriter
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import MeasurableOp, ValuedRV, _logprob, valued_rv
from pymc.logprob.rewriting import cleanup_ir_rewrites_db
from pymc.logprob.transforms import Transform
from pymc.logprob.utils import get_related_valued_nodes


class TransformedValue(Op):
    """A no-op that pairs the original value with its transformed version.

    This is introduced by the `TransformValuesRewrite`
    """

    view_map = {0: [0]}

    def make_node(self, tran_value: TensorVariable, value: TensorVariable):
        return Apply(self, [tran_value, value], [tran_value.type()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError("These `Op`s should be removed from graphs used for computation.")

    def infer_shape(self, fgraph, node, input_shapes):
        return [input_shapes[0]]


transformed_value = TransformedValue()


class TransformedValueRV(MeasurableOp, Op):
    """A no-op that identifies RVs whose values were transformed.

    This is introduced by the `TransformValuesRewrite`
    """

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

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes


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


@node_rewriter(tracks=[ValuedRV])
def transform_values(fgraph: FunctionGraph, node: Apply) -> list[Apply] | None:
    """Apply transforms to value variables.

    It is assumed that the input value variables correspond to forward
    transformations, usually chosen in such a way that the values are
    unconstrained on the real line.

    For example, if ``Y = halfnormal(...)``, we assume the respective value
    variable is specified on the log scale and back-transform it to obtain
    ``Y`` on the natural scale.
    """
    values_to_transforms: TransformValuesMapping | None = getattr(
        fgraph, "values_to_transforms", None
    )

    if values_to_transforms is None:
        return None

    rv_node = node.inputs[0].owner
    valued_nodes = get_related_valued_nodes(fgraph, rv_node)
    rvs = [valued_var.inputs[0] for valued_var in valued_nodes]
    values = [valued_var.inputs[1] for valued_var in valued_nodes]
    transforms = [values_to_transforms.get(value, None) for value in values]

    if all(transform is None for transform in transforms):
        return None

    transformed_rv_op = TransformedValueRV(transforms)
    transformed_rv_node = transformed_rv_op.make_node(*rvs)

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    replacements = {}
    for valued_node, transformed_rv, transform in zip(
        valued_nodes, transformed_rv_node.outputs, transforms
    ):
        rv, value = valued_node.inputs
        [val_rv] = valued_node.outputs

        if transform is None:
            transformed_val = value

        else:
            transformed_val = transformed_value(
                transform.backward(value, *rv.owner.inputs),
                value,
            )

            value_name = value.name
            transform_name = getattr(transform, "name", None)
            if value_name and transform_name:
                transformed_val.name = f"{value_name}_{transform.name}"

        replacements[val_rv] = valued_rv(transformed_rv, transformed_val)

    return replacements


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

    def __init__(
        self,
        values_to_transforms: dict[TensorVariable, Transform | None],
    ):
        """Create the rewriter.

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
