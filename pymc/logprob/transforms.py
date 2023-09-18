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

from copy import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytensor.tensor as pt

from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply, Node
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

from pymc.distributions.transforms import (
    AbsTransform,
    ArccoshTransform,
    ArcsinhTransform,
    ArctanhTransform,
    CoshTransform,
    ErfcTransform,
    ErfcxTransform,
    ErfTransform,
    ExpTransform,
    LocTransform,
    LogTransform,
    PowerTransform,
    ScaleTransform,
    SinhTransform,
    TanhTransform,
    Transform,
)
from pymc.logprob.abstract import (
    MeasurableElemwise,
    MeasurableVariable,
    ValuedRV,
    _icdf,
    _icdf_helper,
    _logcdf,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import cleanup_ir_rewrites_db, measurable_ir_rewrites_db
from pymc.logprob.utils import (
    CheckParameterValue,
    check_potential_measurability,
    filter_measurable_variables,
    get_related_valued_nodes,
)


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


@node_rewriter(tracks=[ValuedRV])
def transform_values(fgraph: FunctionGraph, valued_node: Node) -> Optional[List[Node]]:
    """Apply transforms to value variables.

    It is assumed that the input value variables correspond to forward
    transformations, usually chosen in such a way that the values are
    unconstrained on the real line.

    For example, if ``Y = halfnormal(...)``, we assume the respective value
    variable is specified on the log scale and back-transform it to obtain
    ``Y`` on the natural scale.
    """

    values_to_transforms: Optional[TransformValuesMapping] = getattr(
        fgraph, "values_to_transforms", None
    )

    if values_to_transforms is None:
        return None  # pragma: no cover

    rv_node = valued_node.inputs[0].owner
    valued_nodes = get_related_valued_nodes(rv_node, fgraph)
    rvs = [valued_var.inputs[0] for valued_var in valued_nodes]
    values = [valued_var.inputs[1] for valued_var in valued_nodes]
    transforms = [values_to_transforms.get(value, None) for value in values]

    if all(transform is None for transform in transforms):
        return None

    # Create a new RV Op whose logprob respects the transformed value variable
    transformed_rv_op = _create_transformed_rv_op(rv_node.op, transforms)
    # Create a new `Apply` node and outputs
    transformed_node = rv_node.clone()
    transformed_node.op = transformed_rv_op

    replacements = dict(zip(rv_node.outputs, transformed_node.outputs))

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    for rv, value, transform in zip(rvs, values, transforms):
        if transform is None:
            continue

        new_value = transformed_variable(transform.backward(value, *transformed_node.inputs), value)

        if value.name and getattr(transform, "name", "transformed"):
            new_value.name = f"{value.name}_{transform.name}"

        replacements[value] = new_value

    return replacements


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
    transform_elemwise: Transform
    measurable_input_idx: int

    def __init__(self, *args, transform: Transform, measurable_input_idx: int, **kwargs):
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
    if filter_measurable_variables(node.inputs):
        [inp] = node.inputs
        return [pt.pow(inp, -1.0)]


@node_rewriter([sqr, sqrt])
def measurable_sqrt_sqr_to_power(fgraph, node):
    """Convert square root or square of `MeasurableVariable`s to power form."""
    if not filter_measurable_variables(node.inputs):
        return None

    [inp] = node.inputs

    if isinstance(node.op.scalar_op, Sqr):
        return [pt.pow(inp, 2)]

    if isinstance(node.op.scalar_op, Sqrt):
        return [pt.pow(inp, 1 / 2)]


@node_rewriter([true_div])
def measurable_div_to_product(fgraph, node):
    """Convert divisions involving `MeasurableVariable`s to products."""
    if not filter_measurable_variables(node.inputs):
        return None

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
    if filter_measurable_variables(node.inputs):
        inp = node.inputs[0]
        return [pt.mul(inp, -1)]


@node_rewriter([sub])
def measurable_sub_to_neg(fgraph, node):
    """Convert subtraction involving `MeasurableVariable`s to addition with neg"""
    if filter_measurable_variables(node.inputs):
        minuend, subtrahend = node.inputs
        return [pt.add(minuend, pt.neg(subtrahend))]


@node_rewriter([log1p, softplus, log1mexp, log2, log10])
def measurable_special_log_to_log(fgraph, node):
    """Convert log1p, log1mexp, softplus, log2, log10 of `MeasurableVariable`s to log form."""
    if not filter_measurable_variables(node.inputs):
        return None

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
    if not filter_measurable_variables(node.inputs):
        return None

    [inp] = node.inputs

    if isinstance(node.op.scalar_op, Exp2):
        return [pt.exp(pt.log(2) * inp)]
    if isinstance(node.op.scalar_op, Expm1):
        return [pt.add(pt.exp(inp), -1)]
    if isinstance(node.op.scalar_op, Sigmoid):
        return [1 / (1 + pt.exp(-inp))]


@node_rewriter([pow])
def measurable_power_exponent_to_exp(fgraph, node):
    """Convert power(base, rv) of `MeasurableVariable`s to exp(log(base) * rv) form."""
    base, inp_exponent = node.inputs

    if not filter_measurable_variables([inp_exponent]):
        return None

    # When the base is measurable we have `power(rv, exponent)`, which should be handled by `PowerTransform` and needs no further rewrite.
    # Here we change only the cases where exponent is measurable `power(base, rv)` which is not supported by the `PowerTransform`
    if check_potential_measurability([base]):
        return None

    base = CheckParameterValue("base >= 0")(base, pt.all(pt.ge(base, 0.0)))

    return [pt.exp(pt.log(base) * inp_exponent)]


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

    # Check that we have a single source of measurement
    measurable_inputs = filter_measurable_variables(node.inputs)

    if len(measurable_inputs) != 1:
        return None

    [measurable_input] = measurable_inputs

    # Do not apply rewrite to discrete variables
    if measurable_input.type.dtype.startswith("int"):
        return None

    # Check that other inputs are not potentially measurable, in which case this rewrite
    # would be invalid
    other_inputs = tuple(inp for inp in node.inputs if inp is not measurable_input)

    if check_potential_measurability(other_inputs):
        return None

    scalar_op = node.op.scalar_op
    measurable_input_idx = 0
    transform_inputs: Tuple[TensorVariable, ...] = (measurable_input,)
    transform: Transform

    if isinstance(scalar_op, Pow):
        # We only allow for the base to be measurable
        if measurable_input_idx != 0:
            return None
        try:
            (power,) = other_inputs
            power = pt.get_underlying_scalar_constant_value(power).item()
        # Power needs to be a constant, if not then proceed to the other case power(base, rv)
        except NotScalarConstantError:
            return None
        transform_inputs = (measurable_input, power)
        transform = PowerTransform(power=power)
    elif isinstance(scalar_op, Add):
        transform_inputs = (measurable_input, pt.add(*other_inputs))
        transform = LocTransform(
            transform_args_fn=lambda *inputs: inputs[-1],
        )
    elif isinstance(scalar_op, Mul):
        transform_inputs = (measurable_input, pt.mul(*other_inputs))
        transform = ScaleTransform(
            transform_args_fn=lambda *inputs: inputs[-1],
        )
    else:
        transform = {
            Exp: ExpTransform,
            Log: LogTransform,
            Abs: AbsTransform,
            Sinh: SinhTransform,
            Cosh: CoshTransform,
            Tanh: TanhTransform,
            ArcSinh: ArcsinhTransform,
            ArcCosh: ArccoshTransform,
            ArcTanh: ArctanhTransform,
            Erf: ErfTransform,
            Erfc: ErfcTransform,
            Erfcx: ErfcxTransform,
        }[type(scalar_op)]()
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
    "measurable_power_expotent_to_exp",
    measurable_power_exponent_to_exp,
    "basic",
    "transform",
)

measurable_ir_rewrites_db.register(
    "find_measurable_transforms",
    find_measurable_transforms,
    "basic",
    "transform",
)


def _create_transformed_rv_op(
    rv_op: Op,
    transforms: Union[Transform, Sequence[Union[None, Transform]]],
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
