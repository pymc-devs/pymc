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

from collections.abc import Callable

import numpy as np
import pytensor.tensor as pt

from pytensor import scan
from pytensor.gradient import jacobian
from pytensor.graph.basic import Node, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
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
    MeasurableOp,
    _icdf,
    _icdf_helper,
    _logcdf,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import (
    CheckParameterValue,
    check_potential_measurability,
    filter_measurable_variables,
    find_negated_var,
)


class Transform(abc.ABC):
    ndim_supp = None

    @abc.abstractmethod
    def forward(self, value: TensorVariable, *inputs: Variable) -> TensorVariable:
        """Apply the transformation."""

    @abc.abstractmethod
    def backward(
        self, value: TensorVariable, *inputs: Variable
    ) -> TensorVariable | tuple[TensorVariable, ...]:
        """Invert the transformation.

        Multiple values may be returned when the transformation is not 1-to-1.
        """

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
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}"


class MeasurableTransform(MeasurableElemwise):
    """A placeholder used to specify a log-likelihood for a transformed measurable variable."""

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

    is_discrete = measurable_input.type.dtype.startswith("int")

    logcdf = _logcdf_helper(measurable_input, backward_value)
    if is_discrete:
        logccdf = pt.log1mexp(_logcdf_helper(measurable_input, backward_value - 1))
    else:
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

    if is_discrete:
        return logcdf

    # The jacobian is used to ensure a value in the supported domain was provided
    jacobian = op.transform_elemwise.log_jac_det(value, *other_inputs)
    return pt.switch(pt.isnan(jacobian), -np.inf, logcdf)


@_icdf.register(MeasurableTransform)
def measurable_transform_icdf(op: MeasurableTransform, value, *inputs, **kwargs):
    """Compute the inverse CDF graph for a `MeasurabeTransform`."""
    other_inputs = list(inputs)
    measurable_input = other_inputs.pop(op.measurable_input_idx)

    # Do not apply rewrite to discrete variables
    if measurable_input.type.dtype.startswith("int"):
        raise NotImplementedError("icdf of transformed discrete variables not implemented")

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
    if not filter_measurable_variables(node.inputs):
        return None

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
    if not filter_measurable_variables(node.inputs):
        return None

    inp = node.inputs[0]
    return [pt.mul(inp, -1)]


@node_rewriter([sub])
def measurable_sub_to_neg(fgraph, node):
    """Convert subtraction involving `MeasurableVariable`s to addition with neg."""
    if not filter_measurable_variables(node.inputs):
        return None

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
    if not filter_measurable_variables(node.inputs):
        return None

    base, inp_exponent = node.inputs

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
def find_measurable_transforms(fgraph: FunctionGraph, node: Node) -> list[Node] | None:
    """Find measurable transformations from Elemwise operators."""
    # Node was already converted
    if isinstance(node.op, MeasurableOp):
        return None

    # Check that we have a single source of measurement
    measurable_inputs = filter_measurable_variables(node.inputs)

    if len(measurable_inputs) != 1:
        return None

    [measurable_input] = measurable_inputs
    [measurable_output] = node.outputs

    # Do not apply rewrite to discrete variables except for their addition and negation
    if measurable_input.type.dtype.startswith("int"):
        if not (find_negated_var(measurable_output) or isinstance(node.op.scalar_op, Add)):
            return None
        # Do not allow rewrite if output is cast to a float, because we don't have meta-info on the type of the MeasurableVariable
        if not measurable_output.type.dtype.startswith("int"):
            return None

    # Check that other inputs are not potentially measurable, in which case this rewrite
    # would be invalid
    other_inputs = tuple(inp for inp in node.inputs if inp is not measurable_input)

    if check_potential_measurability(other_inputs):
        return None

    scalar_op = node.op.scalar_op
    measurable_input_idx = 0
    transform_inputs: tuple[TensorVariable, ...] = (measurable_input,)
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


class SinhTransform(Transform):
    name = "sinh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.sinh(value)

    def backward(self, value, *inputs):
        return pt.arcsinh(value)


class CoshTransform(Transform):
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


class TanhTransform(Transform):
    name = "tanh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.tanh(value)

    def backward(self, value, *inputs):
        return pt.arctanh(value)


class ArcsinhTransform(Transform):
    name = "arcsinh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arcsinh(value)

    def backward(self, value, *inputs):
        return pt.sinh(value)


class ArccoshTransform(Transform):
    name = "arccosh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arccosh(value)

    def backward(self, value, *inputs):
        return pt.cosh(value)


class ArctanhTransform(Transform):
    name = "arctanh"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arctanh(value)

    def backward(self, value, *inputs):
        return pt.tanh(value)


class ErfTransform(Transform):
    name = "erf"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erf(value)

    def backward(self, value, *inputs):
        return pt.erfinv(value)


class ErfcTransform(Transform):
    name = "erfc"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erfc(value)

    def backward(self, value, *inputs):
        return pt.erfcinv(value)


class ErfcxTransform(Transform):
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


class LocTransform(Transform):
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


class ScaleTransform(Transform):
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


class LogTransform(Transform):
    name = "log"

    def forward(self, value, *inputs):
        return pt.log(value)

    def backward(self, value, *inputs):
        return pt.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


class ExpTransform(Transform):
    name = "exp"

    def forward(self, value, *inputs):
        return pt.exp(value)

    def backward(self, value, *inputs):
        return pt.log(value)

    def log_jac_det(self, value, *inputs):
        return -pt.log(value)


class AbsTransform(Transform):
    name = "abs"

    def forward(self, value, *inputs):
        return pt.abs(value)

    def backward(self, value, *inputs):
        value = pt.switch(value >= 0, value, np.nan)
        return -value, value

    def log_jac_det(self, value, *inputs):
        return pt.switch(value >= 0, 0, np.nan)


class PowerTransform(Transform):
    name = "power"

    def __init__(self, power=None):
        if not isinstance(power, int | float):
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


class IntervalTransform(Transform):
    name = "interval"

    def __init__(self, args_fn: Callable[..., tuple[Variable | None, Variable | None]]):
        """Create the IntervalTransform object.

        Parameters
        ----------
        args_fn
            Function that expects inputs of RandomVariable and returns the lower
            and upper bounds for the interval transformation. If one of these is
            None, the RV is considered to be unbounded on the respective edge.
        """
        self.args_fn = args_fn

    def get_a_and_b(self, inputs):
        """Return interval bound values.

        Also returns two boolean variables indicating whether the transform is known to be statically bounded.
        This is used to generate smaller graphs in the transform methods.
        """
        a, b = self.args_fn(*inputs)
        lower_bounded, upper_bounded = True, True
        if a is None:
            a = -pt.inf
            lower_bounded = False
        if b is None:
            b = pt.inf
            upper_bounded = False
        return a, b, lower_bounded, upper_bounded

    def forward(self, value, *inputs):
        a, b, lower_bounded, upper_bounded = self.get_a_and_b(inputs)

        log_lower_distance = pt.log(value - a)
        log_upper_distance = pt.log(b - value)

        if lower_bounded and upper_bounded:
            return pt.where(
                pt.and_(pt.neq(a, -pt.inf), pt.neq(b, pt.inf)),
                log_lower_distance - log_upper_distance,
                pt.where(
                    pt.neq(a, -pt.inf),
                    log_lower_distance,
                    pt.where(
                        pt.neq(b, pt.inf),
                        log_upper_distance,
                        value,
                    ),
                ),
            )
        elif lower_bounded:
            return log_lower_distance
        elif upper_bounded:
            return log_upper_distance
        else:
            return value

    def backward(self, value, *inputs):
        a, b, lower_bounded, upper_bounded = self.get_a_and_b(inputs)

        exp_value = pt.exp(value)
        sigmoid_x = pt.sigmoid(value)
        lower_distance = exp_value + a
        upper_distance = b - exp_value

        if lower_bounded and upper_bounded:
            return pt.where(
                pt.and_(pt.neq(a, -pt.inf), pt.neq(b, pt.inf)),
                sigmoid_x * b + (1 - sigmoid_x) * a,
                pt.where(
                    pt.neq(a, -pt.inf),
                    lower_distance,
                    pt.where(
                        pt.neq(b, pt.inf),
                        upper_distance,
                        value,
                    ),
                ),
            )
        elif lower_bounded:
            return lower_distance
        elif upper_bounded:
            return upper_distance
        else:
            return value

    def log_jac_det(self, value, *inputs):
        a, b, lower_bounded, upper_bounded = self.get_a_and_b(inputs)

        if lower_bounded and upper_bounded:
            s = pt.softplus(-value)

            return pt.where(
                pt.and_(pt.neq(a, -pt.inf), pt.neq(b, pt.inf)),
                pt.log(b - a) - 2 * s - value,
                pt.where(
                    pt.or_(pt.neq(a, -pt.inf), pt.neq(b, pt.inf)),
                    value,
                    pt.zeros_like(value),
                ),
            )
        elif lower_bounded or upper_bounded:
            return value
        else:
            return pt.zeros_like(value)


class LogOddsTransform(Transform):
    name = "logodds"

    def backward(self, value, *inputs):
        return pt.expit(value)

    def forward(self, value, *inputs):
        return pt.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = pt.sigmoid(value)
        return pt.log(sigmoid_value) + pt.log1p(-sigmoid_value)


class SimplexTransform(Transform):
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
        value_sum_expanded = pt.concatenate([value_sum_expanded, pt.zeros_like(sum_value)], -1)
        logsumexp_value_expanded = pt.logsumexp(value_sum_expanded, -1, keepdims=True)
        res = pt.log(N) + (N * sum_value) - (N * logsumexp_value_expanded)
        return pt.sum(res, -1)


class CircularTransform(Transform):
    name = "circular"

    def backward(self, value, *inputs):
        return pt.arctan2(pt.sin(value), pt.cos(value))

    def forward(self, value, *inputs):
        return pt.as_tensor_variable(value)

    def log_jac_det(self, value, *inputs):
        return pt.zeros_like(value)


class ChainedTransform(Transform):
    name = "chain"

    def __init__(self, transform_list):
        self.transform_list = transform_list

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
