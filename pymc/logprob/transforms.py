#   Copyright 2022- The PyMC Developers
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

import abc

from copy import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import pytensor.tensor as at

from pytensor.gradient import DisconnectedType, jacobian
from pytensor.graph.basic import Apply, Node, Variable
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.rewriting.basic import GraphRewriter, in2out, node_rewriter
from pytensor.scalar import Add, Exp, Log, Mul
from pytensor.tensor.math import add, exp, log, mul
from pytensor.tensor.rewriting.basic import (
    register_specialize,
    register_stabilize,
    register_useless,
)
from pytensor.tensor.var import TensorVariable

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    _get_measurable_outputs,
    _logprob,
    assign_custom_measurable_outputs,
    logprob,
)
from pymc.logprob.rewriting import PreserveRVMappings, measurable_ir_rewrites_db
from pymc.logprob.utils import walk_model


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


@register_specialize
@register_stabilize
@register_useless
@node_rewriter([TransformedVariable])
def remove_TransformedVariables(fgraph, node):
    if isinstance(node.op, TransformedVariable):
        return [node.inputs[0]]


class RVTransform(abc.ABC):
    @abc.abstractmethod
    def forward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Apply the transformation."""

    @abc.abstractmethod
    def backward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Invert the transformation."""

    def log_jac_det(self, value: TensorVariable, *inputs) -> TensorVariable:
        """Construct the log of the absolute value of the Jacobian determinant."""
        # jac = at.reshape(
        #     gradient(at.sum(self.backward(value, *inputs)), [value]), value.shape
        # )
        # return at.log(at.abs(jac))
        phi_inv = self.backward(value, *inputs)
        return at.log(at.abs(at.nlinalg.det(at.atleast_2d(jacobian(phi_inv, [value])[0]))))


@node_rewriter(tracks=None)
def transform_values(fgraph: FunctionGraph, node: Node) -> Optional[List[Node]]:
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
        trans_node.outputs[rv_var_out_idx].name = rv_var.name

        if transform is None:
            continue

        new_value_var = transformed_variable(
            transform.backward(value_var, *trans_node.inputs), value_var
        )

        if value_var.name and getattr(transform, "name", None):
            new_value_var.name = f"{value_var.name}_{transform.name}"

        rv_map_feature.update_rv_maps(rv_var, new_value_var, trans_node.outputs[rv_var_out_idx])

    return trans_node.outputs


class TransformValuesMapping(Feature):
    r"""A `Feature` that maintains a map between value variables and their transforms."""

    def __init__(self, values_to_transforms):
        self.values_to_transforms = values_to_transforms

    def on_attach(self, fgraph):
        if hasattr(fgraph, "values_to_transforms"):
            raise AlreadyThere()

        fgraph.values_to_transforms = self.values_to_transforms


class TransformValuesRewrite(GraphRewriter):
    r"""Transforms value variables according to a map."""

    transform_rewrite = in2out(transform_values, ignore_newtrees=True)

    def __init__(
        self,
        values_to_transforms: Dict[TensorVariable, Union[RVTransform, None]],
    ):
        """
        Parameters
        ==========
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
        return self.transform_rewrite.rewrite(fgraph)


class MeasurableTransform(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a transformed measurable variable"""

    valid_scalar_types = (Exp, Log, Add, Mul)

    # Cannot use `transform` as name because it would clash with the property added by
    # the `TransformValuesRewrite`
    transform_elemwise: RVTransform
    measurable_input_idx: int

    def __init__(self, *args, transform: RVTransform, measurable_input_idx: int, **kwargs):
        self.transform_elemwise = transform
        self.measurable_input_idx = measurable_input_idx
        super().__init__(*args, **kwargs)


@_get_measurable_outputs.register(MeasurableTransform)
def _get_measurable_outputs_Transform(op, node):
    return [node.default_output()]


@_logprob.register(MeasurableTransform)
def measurable_transform_logprob(op: MeasurableTransform, values, *inputs, **kwargs):
    """Compute the log-probability graph for a `MeasurabeTransform`."""
    # TODO: Could other rewrites affect the order of inputs?
    (value,) = values
    other_inputs = list(inputs)
    measurable_input = other_inputs.pop(op.measurable_input_idx)

    # The value variable must still be back-transformed to be on the natural support of
    # the respective measurable input.
    backward_value = op.transform_elemwise.backward(value, *other_inputs)
    input_logprob = logprob(measurable_input, backward_value, **kwargs)

    jacobian = op.transform_elemwise.log_jac_det(value, *other_inputs)

    return input_logprob + jacobian


@node_rewriter([exp, log, add, mul])
def find_measurable_transforms(fgraph: FunctionGraph, node: Node) -> Optional[List[Node]]:
    """Find measurable transformations from Elemwise operators."""

    # Node was already converted
    if isinstance(node.op, MeasurableVariable):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    # Check that we have a single source of measurement
    measurable_inputs = [
        inp
        for idx, inp in enumerate(node.inputs)
        if inp.owner
        and isinstance(inp.owner.op, MeasurableVariable)
        and inp not in rv_map_feature.rv_values
    ]

    if len(measurable_inputs) != 1:
        return None

    measurable_input: TensorVariable = measurable_inputs[0]

    # Do not apply rewrite to discrete variables
    if measurable_input.type.dtype.startswith("int"):
        return None

    # Check that other inputs are not potentially measurable, in which case this rewrite
    # would be invalid
    other_inputs = tuple(inp for inp in node.inputs if inp is not measurable_input)
    if any(
        ancestor_node
        for ancestor_node in walk_model(
            other_inputs,
            walk_past_rvs=False,
            stop_at_vars=set(rv_map_feature.rv_values),
        )
        if (
            ancestor_node.owner
            and isinstance(ancestor_node.owner.op, MeasurableVariable)
            and ancestor_node not in rv_map_feature.rv_values
        )
    ):
        return None

    # Make base_measure outputs unmeasurable
    # This seems to be the only thing preventing nested rewrites from being erased
    measurable_input = assign_custom_measurable_outputs(measurable_input.owner)

    scalar_op = node.op.scalar_op
    measurable_input_idx = 0
    transform_inputs: Tuple[TensorVariable, ...] = (measurable_input,)
    transform: RVTransform
    if isinstance(scalar_op, Exp):
        transform = ExpTransform()
    elif isinstance(scalar_op, Log):
        transform = LogTransform()
    elif isinstance(scalar_op, Add):
        transform_inputs = (measurable_input, at.add(*other_inputs))
        transform = LocTransform(
            transform_args_fn=lambda *inputs: inputs[-1],
        )
    else:
        transform_inputs = (measurable_input, at.mul(*other_inputs))
        transform = ScaleTransform(
            transform_args_fn=lambda *inputs: inputs[-1],
        )

    transform_op = MeasurableTransform(
        scalar_op=scalar_op,
        transform=transform,
        measurable_input_idx=measurable_input_idx,
    )
    transform_out = transform_op.make_node(*transform_inputs).default_output()
    transform_out.name = node.outputs[0].name

    return [transform_out]


measurable_ir_rewrites_db.register(
    "find_measurable_transforms",
    find_measurable_transforms,
    "basic",
    "transform",
)


class LocTransform(RVTransform):
    name = "loc"

    def __init__(self, transform_args_fn):
        self.transform_args_fn = transform_args_fn

    def forward(self, value, *inputs):
        loc = self.transform_args_fn(*inputs)
        return value + loc

    def backward(self, value, *inputs):
        loc = self.transform_args_fn(*inputs)
        return value - loc

    def log_jac_det(self, value, *inputs):
        return at.zeros_like(value)


class ScaleTransform(RVTransform):
    name = "scale"

    def __init__(self, transform_args_fn):
        self.transform_args_fn = transform_args_fn

    def forward(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return value * scale

    def backward(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return value / scale

    def log_jac_det(self, value, *inputs):
        scale = self.transform_args_fn(*inputs)
        return -at.log(at.abs(scale))


class LogTransform(RVTransform):
    name = "log"

    def forward(self, value, *inputs):
        return at.log(value)

    def backward(self, value, *inputs):
        return at.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


class ExpTransform(RVTransform):
    name = "exp"

    def forward(self, value, *inputs):
        return at.exp(value)

    def backward(self, value, *inputs):
        return at.log(value)

    def log_jac_det(self, value, *inputs):
        return -at.log(value)


class IntervalTransform(RVTransform):
    name = "interval"

    def __init__(self, args_fn: Callable[..., Tuple[Optional[Variable], Optional[Variable]]]):
        """

        Parameters
        ==========
        args_fn
            Function that expects inputs of RandomVariable and returns the lower
            and upper bounds for the interval transformation. If one of these is
            None, the RV is considered to be unbounded on the respective edge.
        """
        self.args_fn = args_fn

    def forward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            return at.log(value - a) - at.log(b - value)
        elif a is not None:
            return at.log(value - a)
        elif b is not None:
            return at.log(b - value)
        else:
            raise ValueError("Both edges of IntervalTransform cannot be None")

    def backward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            sigmoid_x = at.sigmoid(value)
            return sigmoid_x * b + (1 - sigmoid_x) * a
        elif a is not None:
            return at.exp(value) + a
        elif b is not None:
            return b - at.exp(value)
        else:
            raise ValueError("Both edges of IntervalTransform cannot be None")

    def log_jac_det(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            s = at.softplus(-value)
            return at.log(b - a) - 2 * s - value
        elif a is None and b is None:
            raise ValueError("Both edges of IntervalTransform cannot be None")
        else:
            return value


class LogOddsTransform(RVTransform):
    name = "logodds"

    def backward(self, value, *inputs):
        return at.expit(value)

    def forward(self, value, *inputs):
        return at.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = at.sigmoid(value)
        return at.log(sigmoid_value) + at.log1p(-sigmoid_value)


class SimplexTransform(RVTransform):
    name = "simplex"

    def forward(self, value, *inputs):
        log_value = at.log(value)
        shift = at.sum(log_value, -1, keepdims=True) / value.shape[-1]
        return log_value[..., :-1] - shift

    def backward(self, value, *inputs):
        value = at.concatenate([value, -at.sum(value, -1, keepdims=True)], axis=-1)
        exp_value_max = at.exp(value - at.max(value, -1, keepdims=True))
        return exp_value_max / at.sum(exp_value_max, -1, keepdims=True)

    def log_jac_det(self, value, *inputs):
        N = value.shape[-1] + 1
        sum_value = at.sum(value, -1, keepdims=True)
        value_sum_expanded = value + sum_value
        value_sum_expanded = at.concatenate([value_sum_expanded, at.zeros(sum_value.shape)], -1)
        logsumexp_value_expanded = at.logsumexp(value_sum_expanded, -1, keepdims=True)
        res = at.log(N) + (N * sum_value) - (N * logsumexp_value_expanded)
        return at.sum(res, -1)


class CircularTransform(RVTransform):
    name = "circular"

    def backward(self, value, *inputs):
        return at.arctan2(at.sin(value), at.cos(value))

    def forward(self, value, *inputs):
        return at.as_tensor_variable(value)

    def log_jac_det(self, value, *inputs):
        return at.zeros(value.shape)


class ChainedTransform(RVTransform):
    name = "chain"

    def __init__(self, transform_list, base_op):
        self.transform_list = transform_list
        self.base_op = base_op

    def forward(self, value, *inputs):
        for transform in self.transform_list:
            value = transform.forward(value, *inputs)
        return value

    def backward(self, value, *inputs):
        for transform in reversed(self.transform_list):
            value = transform.backward(value, *inputs)
        return value

    def log_jac_det(self, value, *inputs):
        value = at.as_tensor_variable(value)
        det_list = []
        ndim0 = value.ndim
        for transform in reversed(self.transform_list):
            det_ = transform.log_jac_det(value, *inputs)
            det_list.append(det_)
            ndim0 = min(ndim0, det_.ndim)
            value = transform.backward(value, *inputs)
        # match the shape of the smallest jacobian_det
        det = 0.0
        for det_ in det_list:
            if det_.ndim > ndim0:
                det += det_.sum(axis=-1)
            else:
                det += det_
        return det


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
    ==========
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

        if use_jacobian:
            assert len(values) == len(logprobs) == len(op.transforms)
            logprobs_jac = []
            for value, transform, logprob in zip(values, op.transforms, logprobs):
                if transform is None:
                    logprobs_jac.append(logprob)
                    continue
                assert isinstance(value.owner.op, TransformedVariable)
                original_forward_value = value.owner.inputs[1]
                jacobian = transform.log_jac_det(original_forward_value, *inputs).copy()
                if value.name:
                    jacobian.name = f"{value.name}_jacobian"
                logprobs_jac.append(logprob + jacobian)
            logprobs = logprobs_jac

        return logprobs

    new_op = copy(rv_op)
    new_op.__class__ = new_op_type

    return new_op
