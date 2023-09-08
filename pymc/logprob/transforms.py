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

import abc

from copy import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytensor.tensor as pt

from pytensor import scan
from pytensor.gradient import DisconnectedType, jacobian
from pytensor.graph.basic import Apply, Node, Variable
from pytensor.graph.features import AlreadyThere, Feature
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import Op
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.basic import GraphRewriter, in2out, node_rewriter
from pytensor.scalar import (
    Abs,
    Add,
    ArcCosh,
    ArcSinh,
    ArcTanh,
    Cosh,
    Erf,
    Erfc,
    Erfcx,
    Exp,
    Exp2,
    Expm1,
    Log,
    Log1mexp,
    Log1p,
    Log2,
    Log10,
    Mul,
    Pow,
    Sigmoid,
    Sinh,
    Softplus,
    Sqr,
    Sqrt,
    Tanh,
)
from pytensor.scan.op import Scan
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import (
    abs,
    add,
    arccosh,
    arcsinh,
    arctanh,
    cosh,
    erf,
    erfc,
    erfcx,
    exp,
    exp2,
    expm1,
    log,
    log1mexp,
    log1p,
    log2,
    log10,
    mul,
    neg,
    pow,
    reciprocal,
    sigmoid,
    sinh,
    softplus,
    sqr,
    sqrt,
    sub,
    tanh,
    true_div,
)
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    _icdf,
    _icdf_helper,
    _logcdf,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import (
    PreserveRVMappings,
    cleanup_ir_rewrites_db,
    measurable_ir_rewrites_db,
)
from pymc.logprob.utils import check_potential_measurability


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


class RVTransform(abc.ABC):
    ndim_supp = None

    @abc.abstractmethod
    def forward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Apply the transformation."""

    @abc.abstractmethod
    def backward(
        self, value: TensorVariable, *inputs: Variable
    ) -> Union[TensorVariable, Tuple[TensorVariable, ...]]:
        """Invert the transformation. Multiple values may be returned when the
        transformation is not 1-to-1"""

    def log_jac_det(self, value: TensorVariable, *inputs) -> TensorVariable:
        """Construct the log of the absolute value of the Jacobian determinant."""
        if self.ndim_supp not in (0, 1):
            raise NotImplementedError(
                f"RVTransform default log_jac_det only implemented for ndim_supp in (0, 1), got {self.ndim_supp=}"
            )
        if self.ndim_supp == 0:
            jac = pt.reshape(pt.grad(pt.sum(self.backward(value, *inputs)), [value]), value.shape)
            return pt.log(pt.abs(jac))
        else:
            phi_inv = self.backward(value, *inputs)
            return pt.log(pt.abs(pt.nlinalg.det(pt.atleast_2d(jacobian(phi_inv, [value])[0]))))

    def __str__(self):
        return f"{self.__class__.__name__}"


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
def transform_scan_values(fgraph: FunctionGraph, node: Node) -> Optional[List[Node]]:
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


class MeasurableTransform(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a transformed measurable variable"""

    valid_scalar_types = (
        Exp,
        Log,
        Add,
        Mul,
        Pow,
        Abs,
        Sinh,
        Cosh,
        Tanh,
        ArcSinh,
        ArcCosh,
        ArcTanh,
        Erf,
        Erfc,
        Erfcx,
    )

    # Cannot use `transform` as name because it would clash with the property added by
    # the `TransformValuesRewrite`
    transform_elemwise: RVTransform
    measurable_input_idx: int

    def __init__(self, *args, transform: RVTransform, measurable_input_idx: int, **kwargs):
        self.transform_elemwise = transform
        self.measurable_input_idx = measurable_input_idx
        super().__init__(*args, **kwargs)


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

    # Some transformations, like squaring may produce multiple backward values
    if isinstance(backward_value, tuple):
        input_logprob = pt.logaddexp(
            *(
                _logprob_helper(measurable_input, backward_val, **kwargs)
                for backward_val in backward_value
            )
        )
    else:
        input_logprob = _logprob_helper(measurable_input, backward_value)

    jacobian = op.transform_elemwise.log_jac_det(value, *other_inputs)

    if input_logprob.ndim < value.ndim:
        # For multivariate variables, the Jacobian is diagonal.
        # We can get the right result by summing the last dimensions
        # of `transform_elemwise.log_jac_det`
        ndim_supp = value.ndim - input_logprob.ndim
        jacobian = jacobian.sum(axis=tuple(range(-ndim_supp, 0)))

    # The jacobian is used to ensure a value in the supported domain was provided
    return pt.switch(pt.isnan(jacobian), -np.inf, input_logprob + jacobian)


MONOTONICALLY_INCREASING_OPS = (Exp, Log, Add, Sinh, Tanh, ArcSinh, ArcCosh, ArcTanh, Erf)
MONOTONICALLY_DECREASING_OPS = (Erfc, Erfcx)


@_logcdf.register(MeasurableTransform)
def measurable_transform_logcdf(op: MeasurableTransform, value, *inputs, **kwargs):
    """Compute the log-CDF graph for a `MeasurabeTransform`."""
    other_inputs = list(inputs)
    measurable_input = other_inputs.pop(op.measurable_input_idx)

    backward_value = op.transform_elemwise.backward(value, *other_inputs)

    # Fail if transformation is not injective
    # A TensorVariable is returned in 1-to-1 inversions, and a tuple in 1-to-many
    if isinstance(backward_value, tuple):
        raise NotImplementedError

    logcdf = _logcdf_helper(measurable_input, backward_value)
    logccdf = pt.log1mexp(logcdf)

    if isinstance(op.scalar_op, MONOTONICALLY_INCREASING_OPS):
        pass
    elif isinstance(op.scalar_op, MONOTONICALLY_DECREASING_OPS):
        logcdf = logccdf
    # mul is monotonically increasing for scale > 0, and monotonically decreasing otherwise
    elif isinstance(op.scalar_op, Mul):
        [scale] = other_inputs
        logcdf = pt.switch(pt.ge(scale, 0), logcdf, logccdf)
    # pow is increasing if pow > 0, and decreasing otherwise (even powers are rejected above)!
    # Care must be taken to handle negative values (https://math.stackexchange.com/a/442362/783483)
    elif isinstance(op.scalar_op, Pow):
        if op.transform_elemwise.power < 0:
            logcdf_zero = _logcdf_helper(measurable_input, 0)
            logcdf = pt.switch(
                pt.lt(backward_value, 0),
                pt.log(pt.exp(logcdf_zero) - pt.exp(logcdf)),
                pt.logaddexp(logccdf, logcdf_zero),
            )
    else:
        # We don't know if this Op is monotonically increasing/decreasing
        raise NotImplementedError

    # The jacobian is used to ensure a value in the supported domain was provided
    jacobian = op.transform_elemwise.log_jac_det(value, *other_inputs)

    return pt.switch(pt.isnan(jacobian), -np.inf, logcdf)


@_icdf.register(MeasurableTransform)
def measurable_transform_icdf(op: MeasurableTransform, value, *inputs, **kwargs):
    """Compute the inverse CDF graph for a `MeasurabeTransform`."""
    other_inputs = list(inputs)
    measurable_input = other_inputs.pop(op.measurable_input_idx)

    if isinstance(op.scalar_op, MONOTONICALLY_INCREASING_OPS):
        pass
    elif isinstance(op.scalar_op, MONOTONICALLY_DECREASING_OPS):
        value = 1 - value
    elif isinstance(op.scalar_op, Mul):
        [scale] = other_inputs
        value = pt.switch(pt.lt(scale, 0), 1 - value, value)
    elif isinstance(op.scalar_op, Pow):
        if op.transform_elemwise.power < 0:
            raise NotImplementedError
    else:
        raise NotImplementedError

    input_icdf = _icdf_helper(measurable_input, value)
    icdf = op.transform_elemwise.forward(input_icdf, *other_inputs)

    # Fail if transformation is not injective
    # A TensorVariable is returned in 1-to-1 inversions, and a tuple in 1-to-many
    if isinstance(op.transform_elemwise.backward(icdf, *other_inputs), tuple):
        raise NotImplementedError

    return icdf


@node_rewriter([reciprocal])
def measurable_reciprocal_to_power(fgraph, node):
    """Convert reciprocal of `MeasurableVariable`s to power."""
    [inp] = node.inputs
    return [pt.pow(inp, -1.0)]


@node_rewriter([sqr, sqrt])
def measurable_sqrt_sqr_to_power(fgraph, node):
    """Convert square root or square of `MeasurableVariable`s to power form."""
    [inp] = node.inputs

    if isinstance(node.op.scalar_op, Sqr):
        return [pt.pow(inp, 2)]

    if isinstance(node.op.scalar_op, Sqrt):
        return [pt.pow(inp, 1 / 2)]


@node_rewriter([true_div])
def measurable_div_to_product(fgraph, node):
    """Convert divisions involving `MeasurableVariable`s to products."""
    numerator, denominator = node.inputs

    # Check if numerator is 1
    try:
        if pt.get_scalar_constant_value(numerator) == 1:
            # We convert the denominator directly to a power transform as this
            # must be the measurable input
            return [pt.pow(denominator, -1)]
    except NotScalarConstantError:
        pass
    # We don't convert the denominator directly to a power transform as
    # it might not be measurable (and therefore not needed)
    return [pt.mul(numerator, pt.reciprocal(denominator))]


@node_rewriter([neg])
def measurable_neg_to_product(fgraph, node):
    """Convert negation of `MeasurableVariable`s to product with `-1`."""
    inp = node.inputs[0]
    return [pt.mul(inp, -1.0)]


@node_rewriter([sub])
def measurable_sub_to_neg(fgraph, node):
    """Convert subtraction involving `MeasurableVariable`s to addition with neg"""
    minuend, subtrahend = node.inputs
    return [pt.add(minuend, pt.neg(subtrahend))]


@node_rewriter([log1p, softplus, log1mexp, log2, log10])
def measurable_special_log_to_log(fgraph, node):
    """Convert log1p, log1mexp, softplus, log2, log10 of `MeasurableVariable`s to log form."""
    [inp] = node.inputs

    if isinstance(node.op.scalar_op, Log1p):
        return [pt.log(1 + inp)]
    if isinstance(node.op.scalar_op, Softplus):
        return [pt.log(1 + pt.exp(inp))]
    if isinstance(node.op.scalar_op, Log1mexp):
        return [pt.log(1 - pt.exp(inp))]
    if isinstance(node.op.scalar_op, Log2):
        return [pt.log(inp) / pt.log(2)]
    if isinstance(node.op.scalar_op, Log10):
        return [pt.log(inp) / pt.log(10)]


@node_rewriter([expm1, sigmoid, exp2])
def measurable_special_exp_to_exp(fgraph, node):
    """Convert expm1, sigmoid, and exp2 of `MeasurableVariable`s to xp form."""
    [inp] = node.inputs
    if isinstance(node.op.scalar_op, Exp2):
        return [pt.exp(pt.log(2) * inp)]
    if isinstance(node.op.scalar_op, Expm1):
        return [pt.add(pt.exp(inp), -1)]
    if isinstance(node.op.scalar_op, Sigmoid):
        return [1 / (1 + pt.exp(-inp))]


@node_rewriter(
    [
        exp,
        log,
        add,
        mul,
        pow,
        abs,
        sinh,
        cosh,
        tanh,
        arcsinh,
        arccosh,
        arctanh,
        erf,
        erfc,
        erfcx,
    ]
)
def find_measurable_transforms(fgraph: FunctionGraph, node: Node) -> Optional[List[Node]]:
    """Find measurable transformations from Elemwise operators."""

    # Node was already converted
    if isinstance(node.op, MeasurableVariable):
        return None  # pragma: no cover

    rv_map_feature: Optional[PreserveRVMappings] = getattr(fgraph, "preserve_rv_mappings", None)
    if rv_map_feature is None:
        return None  # pragma: no cover

    # Check that we have a single source of measurement
    measurable_inputs = rv_map_feature.request_measurable(node.inputs)

    if len(measurable_inputs) != 1:
        return None

    [measurable_input] = measurable_inputs

    # Do not apply rewrite to discrete variables
    if measurable_input.type.dtype.startswith("int"):
        return None

    # Check that other inputs are not potentially measurable, in which case this rewrite
    # would be invalid
    other_inputs = tuple(inp for inp in node.inputs if inp is not measurable_input)

    if check_potential_measurability(other_inputs, rv_map_feature.rv_values.keys()):
        return None

    scalar_op = node.op.scalar_op
    measurable_input_idx = 0
    transform_inputs: Tuple[TensorVariable, ...] = (measurable_input,)
    transform: RVTransform

    transform_dict = {
        Exp: ExpTransform(),
        Log: LogTransform(),
        Abs: AbsTransform(),
        Sinh: SinhTransform(),
        Cosh: CoshTransform(),
        Tanh: TanhTransform(),
        ArcSinh: ArcsinhTransform(),
        ArcCosh: ArccoshTransform(),
        ArcTanh: ArctanhTransform(),
        Erf: ErfTransform(),
        Erfc: ErfcTransform(),
        Erfcx: ErfcxTransform(),
    }
    transform = transform_dict.get(type(scalar_op), None)
    if isinstance(scalar_op, Pow):
        # We only allow for the base to be measurable
        if measurable_input_idx != 0:
            return None
        try:
            (power,) = other_inputs
            power = pt.get_underlying_scalar_constant_value(power).item()
        # Power needs to be a constant
        except NotScalarConstantError:
            return None
        transform_inputs = (measurable_input, power)
        transform = PowerTransform(power=power)
    elif isinstance(scalar_op, Add):
        transform_inputs = (measurable_input, pt.add(*other_inputs))
        transform = LocTransform(
            transform_args_fn=lambda *inputs: inputs[-1],
        )
    elif transform is None:
        transform_inputs = (measurable_input, pt.mul(*other_inputs))
        transform = ScaleTransform(
            transform_args_fn=lambda *inputs: inputs[-1],
        )
    transform_op = MeasurableTransform(
        scalar_op=scalar_op,
        transform=transform,
        measurable_input_idx=measurable_input_idx,
    )
    transform_out = transform_op.make_node(*transform_inputs).default_output()
    return [transform_out]


measurable_ir_rewrites_db.register(
    "measurable_reciprocal_to_power",
    measurable_reciprocal_to_power,
    "basic",
    "transform",
)


measurable_ir_rewrites_db.register(
    "measurable_sqrt_sqr_to_power",
    measurable_sqrt_sqr_to_power,
    "basic",
    "transform",
)


measurable_ir_rewrites_db.register(
    "measurable_div_to_product",
    measurable_div_to_product,
    "basic",
    "transform",
)


measurable_ir_rewrites_db.register(
    "measurable_neg_to_product",
    measurable_neg_to_product,
    "basic",
    "transform",
)

measurable_ir_rewrites_db.register(
    "measurable_sub_to_neg",
    measurable_sub_to_neg,
    "basic",
    "transform",
)

measurable_ir_rewrites_db.register(
    "measurable_special_log_to_log",
    measurable_special_log_to_log,
    "basic",
    "transform",
)

measurable_ir_rewrites_db.register(
    "measurable_special_exp_to_exp",
    measurable_special_exp_to_exp,
    "basic",
    "transform",
)


measurable_ir_rewrites_db.register(
    "find_measurable_transforms",
    find_measurable_transforms,
    "basic",
    "transform",
)


class SinhTransform(RVTransform):
    name = "sinh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.sinh(value)

    def backward(self, value, *inputs):
        return pt.arcsinh(value)


class CoshTransform(RVTransform):
    name = "cosh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.cosh(value)

    def backward(self, value, *inputs):
        back_value = pt.arccosh(value)
        return (-back_value, back_value)

    def log_jac_det(self, value, *inputs):
        return pt.switch(
            value < 1,
            np.nan,
            -pt.log(pt.sqrt(value**2 - 1)),
        )


class TanhTransform(RVTransform):
    name = "tanh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.tanh(value)

    def backward(self, value, *inputs):
        return pt.arctanh(value)


class ArcsinhTransform(RVTransform):
    name = "arcsinh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arcsinh(value)

    def backward(self, value, *inputs):
        return pt.sinh(value)


class ArccoshTransform(RVTransform):
    name = "arccosh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arccosh(value)

    def backward(self, value, *inputs):
        return pt.cosh(value)


class ArctanhTransform(RVTransform):
    name = "arctanh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arctanh(value)

    def backward(self, value, *inputs):
        return pt.tanh(value)


class ErfTransform(RVTransform):
    name = "erf"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erf(value)

    def backward(self, value, *inputs):
        return pt.erfinv(value)


class ErfcTransform(RVTransform):
    name = "erfc"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erfc(value)

    def backward(self, value, *inputs):
        return pt.erfcinv(value)


class ErfcxTransform(RVTransform):
    name = "erfcx"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erfcx(value)

    def backward(self, value, *inputs):
        # computes the inverse of erfcx, this was adapted from
        # https://tinyurl.com/4mxfd3cz
        x = pt.switch(value <= 1, 1.0 / (value * pt.sqrt(np.pi)), -pt.sqrt(pt.log(value)))

        def calc_delta_x(value, prior_result):
            return prior_result - (pt.erfcx(prior_result) - value) / (
                2 * prior_result * pt.erfcx(prior_result) - 2 / pt.sqrt(np.pi)
            )

        result, updates = scan(
            fn=calc_delta_x,
            outputs_info=pt.ones_like(x),
            non_sequences=value,
            n_steps=10,
        )
        return result[-1]


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
        return pt.zeros_like(value)


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
        return -pt.log(pt.abs(pt.broadcast_to(scale, value.shape)))


class LogTransform(RVTransform):
    name = "log"

    def forward(self, value, *inputs):
        return pt.log(value)

    def backward(self, value, *inputs):
        return pt.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


class ExpTransform(RVTransform):
    name = "exp"

    def forward(self, value, *inputs):
        return pt.exp(value)

    def backward(self, value, *inputs):
        return pt.log(value)

    def log_jac_det(self, value, *inputs):
        return -pt.log(value)


class AbsTransform(RVTransform):
    name = "abs"

    def forward(self, value, *inputs):
        return pt.abs(value)

    def backward(self, value, *inputs):
        value = pt.switch(value >= 0, value, np.nan)
        return -value, value

    def log_jac_det(self, value, *inputs):
        return pt.switch(value >= 0, 0, np.nan)


class PowerTransform(RVTransform):
    name = "power"

    def __init__(self, power=None):
        if not isinstance(power, (int, float)):
            raise TypeError(f"Power must be integer or float, got {type(power)}")
        if power == 0:
            raise ValueError("Power cannot be 0")
        self.power = power
        super().__init__()

    def forward(self, value, *inputs):
        return pt.power(value, self.power)

    def backward(self, value, *inputs):
        inv_power = 1 / self.power

        # Powers that don't admit negative values
        if (np.abs(self.power) < 1) or (self.power % 2 == 0):
            backward_value = pt.switch(value >= 0, pt.power(value, inv_power), np.nan)
        # Powers that admit negative values require special logic, because (-1)**(1/3) returns `nan` in PyTensor
        else:
            backward_value = pt.power(pt.abs(value), inv_power) * pt.switch(value >= 0, 1, -1)

        # In this case the transform is not 1-to-1
        if self.power % 2 == 0:
            return -backward_value, backward_value
        else:
            return backward_value

    def log_jac_det(self, value, *inputs):
        inv_power = 1 / self.power

        # Note: This fails for value==0
        res = np.log(np.abs(inv_power)) + (inv_power - 1) * pt.log(pt.abs(value))

        # Powers that don't admit negative values
        if (np.abs(self.power) < 1) or (self.power % 2 == 0):
            res = pt.switch(value >= 0, res, np.nan)

        return res


class IntervalTransform(RVTransform):
    name = "interval"

    def __init__(self, args_fn: Callable[..., Tuple[Optional[Variable], Optional[Variable]]]):
        """

        Parameters
        ----------
        args_fn
            Function that expects inputs of RandomVariable and returns the lower
            and upper bounds for the interval transformation. If one of these is
            None, the RV is considered to be unbounded on the respective edge.
        """
        self.args_fn = args_fn

    def forward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            return pt.log(value - a) - pt.log(b - value)
        elif a is not None:
            return pt.log(value - a)
        elif b is not None:
            return pt.log(b - value)
        else:
            raise ValueError("Both edges of IntervalTransform cannot be None")

    def backward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            sigmoid_x = pt.sigmoid(value)
            return sigmoid_x * b + (1 - sigmoid_x) * a
        elif a is not None:
            return pt.exp(value) + a
        elif b is not None:
            return b - pt.exp(value)
        else:
            raise ValueError("Both edges of IntervalTransform cannot be None")

    def log_jac_det(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            s = pt.softplus(-value)
            return pt.log(b - a) - 2 * s - value
        elif a is None and b is None:
            raise ValueError("Both edges of IntervalTransform cannot be None")
        else:
            return value


class LogOddsTransform(RVTransform):
    name = "logodds"

    def backward(self, value, *inputs):
        return pt.expit(value)

    def forward(self, value, *inputs):
        return pt.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = pt.sigmoid(value)
        return pt.log(sigmoid_value) + pt.log1p(-sigmoid_value)


class SimplexTransform(RVTransform):
    name = "simplex"

    def forward(self, value, *inputs):
        value = pt.as_tensor(value)
        log_value = pt.log(value)
        N = value.shape[-1].astype(value.dtype)
        shift = pt.sum(log_value, -1, keepdims=True) / N
        return log_value[..., :-1] - shift

    def backward(self, value, *inputs):
        value = pt.concatenate([value, -pt.sum(value, -1, keepdims=True)], axis=-1)
        exp_value_max = pt.exp(value - pt.max(value, -1, keepdims=True))
        return exp_value_max / pt.sum(exp_value_max, -1, keepdims=True)

    def log_jac_det(self, value, *inputs):
        value = pt.as_tensor(value)
        N = value.shape[-1] + 1
        N = N.astype(value.dtype)
        sum_value = pt.sum(value, -1, keepdims=True)
        value_sum_expanded = value + sum_value
        value_sum_expanded = pt.concatenate([value_sum_expanded, pt.zeros(sum_value.shape)], -1)
        logsumexp_value_expanded = pt.logsumexp(value_sum_expanded, -1, keepdims=True)
        res = pt.log(N) + (N * sum_value) - (N * logsumexp_value_expanded)
        return pt.sum(res, -1)


class CircularTransform(RVTransform):
    name = "circular"

    def backward(self, value, *inputs):
        return pt.arctan2(pt.sin(value), pt.cos(value))

    def forward(self, value, *inputs):
        return pt.as_tensor_variable(value)

    def log_jac_det(self, value, *inputs):
        return pt.zeros(value.shape)


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
        value = pt.as_tensor_variable(value)
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
                ndim_diff = det_.ndim - ndim0
                det += det_.sum(axis=tuple(range(-ndim_diff, 0)))
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
