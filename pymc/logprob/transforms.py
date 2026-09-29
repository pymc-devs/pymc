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
import abc

from collections.abc import Callable

import numpy as np
import pytensor.scalar as ps
import pytensor.tensor as pt

from pytensor import scan
from pytensor.gradient import jacobian
from pytensor.graph.basic import Variable
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.traversal import walk
from pytensor.scalar import (
    Abs,
    Add,
    ArcCos,
    ArcCosh,
    ArcSin,
    ArcSinh,
    ArcTan,
    ArcTanh,
    Cast,
    Cosh,
    Erf,
    Erfc,
    Erfcinv,
    Erfcx,
    Erfinv,
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
from pytensor.tensor.basic import Alloc
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.exceptions import NotScalarConstantError
from pytensor.tensor.math import (
    exp2,
    expm1,
    log1mexp,
    log1p,
    log2,
    log10,
    neg,
    pow,
    reciprocal,
    sigmoid,
    softplus,
    sqr,
    sqrt,
    sub,
    true_div,
    variadic_add,
    variadic_mul,
)
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import (
    _icdf,
    _icdf_helper,
    _logccdf_helper,
    _logcdf,
    _logcdf_helper,
    logprob_query,
)
from pymc.logprob.query import (
    UnsupportedObservation,
    bind_value,
    contains_random,
    density_sources,
    infer_support_axes,
    is_random_source,
    other_query_uses,
    query_at_value,
    query_parts,
    random_inputs,
    rewrite_logprob_query,
    support_axes,
)
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.utils import (
    CheckParameterValue,
    filter_measurable_variables,
    find_negated_var,
)
from pymc.math import logdiffexp


class Transform(abc.ABC):
    ndim_supp: int | None = None

    def support_axes(self, value):
        if self.ndim_supp is None:
            raise NotImplementedError(f"{self} must declare ndim_supp")
        return tuple(range(value.ndim - self.ndim_supp, value.ndim))

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
            return pt.log(pt.abs(pt.linalg.det(pt.atleast_2d(jacobian(phi_inv, [value])[0]))))

    def __str__(self):
        """Return a string representation of the object."""
        return f"{self.__class__.__name__}"


def select_transform(scalar_op, inputs):
    """Identify the one input whose value can be inverted with the current conditioning."""
    candidates = filter_measurable_variables(inputs)
    if len(candidates) != 1:
        return None
    (base,) = candidates
    if any(contains_random(inp) for inp in inputs if inp is not base):
        return None

    # Follow the deterministic path to its stochastic sources. Dtype relabelling
    # and broadcast copies do not establish a continuous change of variables.
    def inputs_to_check(var):
        if not contains_random(var) or is_random_source(var):
            return ()
        return random_inputs(var)

    for var in walk([base], inputs_to_check):
        if not contains_random(var):
            continue
        if isinstance(var.owner_op, Alloc):
            return None
        if isinstance(var.owner_op, Elemwise) and isinstance(var.owner.op.scalar_op, Cast):
            if var.owner.inputs[0].dtype.startswith(
                ("int", "uint", "bool")
            ) and var.dtype.startswith("float"):
                return None
    others = tuple(inp for inp in inputs if inp is not base)
    if base.dtype.startswith(("int", "uint", "bool")):
        negated = isinstance(scalar_op, ps.Neg) or (
            isinstance(scalar_op, Mul)
            and len(others) == 1
            and find_negated_var(Elemwise(scalar_op)(*inputs)) is not None
        )
        if not (isinstance(scalar_op, Add) or negated):
            return None
        if not Elemwise(scalar_op)(*inputs).dtype.startswith(("int", "uint")):
            return None
    if isinstance(scalar_op, Pow):
        if base is not inputs[0]:
            return None
        try:
            power = pt.get_underlying_scalar_constant_value(others[0]).item()
        except NotScalarConstantError:
            return None
        transform = PowerTransform(power)
    elif isinstance(scalar_op, Add):
        others = (variadic_add(*others),)
        transform = LocTransform(lambda *args: args[-1])
    elif isinstance(scalar_op, Mul):
        others = (variadic_mul(*others),)
        transform = ScaleTransform(lambda *args: args[-1])
    elif isinstance(scalar_op, ps.Neg):
        others = (-1,)
        transform = ScaleTransform(lambda *args: args[-1])
    else:
        transform_types: dict[type, type[Transform]] = {
            Exp: ExpTransform,
            Log: LogTransform,
            Abs: AbsTransform,
            Sinh: SinhTransform,
            Cosh: CoshTransform,
            Tanh: TanhTransform,
            ArcSin: ArcsinTransform,
            ArcCos: ArccosTransform,
            ArcTan: ArctanTransform,
            ArcSinh: ArcsinhTransform,
            ArcCosh: ArccoshTransform,
            ArcTanh: ArctanhTransform,
            Erf: ErfTransform,
            Erfc: ErfcTransform,
            Erfcx: ErfcxTransform,
            Erfinv: ErfinvTransform,
            Erfcinv: ErfcinvTransform,
        }
        transform_type = transform_types.get(type(scalar_op))
        if transform_type is None:
            return None
        transform = transform_type()
    return base, transform, others


MONOTONICALLY_INCREASING_OPS = (
    Exp,
    Log,
    Add,
    Sinh,
    Tanh,
    ArcSin,
    ArcTan,
    ArcSinh,
    ArcCosh,
    ArcTanh,
    Erf,
    Erfinv,
    Sigmoid,
)
MONOTONICALLY_DECREASING_OPS = (ArcCos, Erfc, Erfcx, Erfcinv)


@_logcdf.register(ps.ScalarOp)
def measurable_transform_logcdf(op, value, *inputs):
    """Compute the log-CDF graph for a `MeasurabeTransform`."""
    selection = select_transform(op, inputs)
    if selection is None:
        raise NotImplementedError(f"LogCDF method not implemented for {type(op).__name__}")
    measurable_input, transform, other_inputs = selection
    backward_value = transform.backward(value, *other_inputs)

    # Fail if transformation is not injective
    # A TensorVariable is returned in 1-to-1 inversions, and a tuple in 1-to-many
    if isinstance(backward_value, tuple):
        raise NotImplementedError

    is_discrete = measurable_input.type.dtype.startswith("int")

    logcdf = _logcdf_helper(measurable_input, backward_value)
    if is_discrete:
        # For discrete distributions, P(X >= t) = P(X > t-1)
        logccdf = _logccdf_helper(measurable_input, backward_value - 1)
    else:
        logccdf = _logccdf_helper(measurable_input, backward_value)

    if isinstance(op, MONOTONICALLY_INCREASING_OPS):
        pass
    elif isinstance(op, MONOTONICALLY_DECREASING_OPS):
        logcdf = logccdf
    # mul is monotonically increasing for scale > 0, and monotonically decreasing otherwise
    elif isinstance(op, Mul):
        [scale] = other_inputs
        logcdf = pt.switch(pt.ge(scale, 0), logcdf, logccdf)
    # pow is increasing if pow > 0, and decreasing otherwise (even powers are rejected above)!
    # Care must be taken to handle negative values (https://math.stackexchange.com/a/442362/783483)
    elif isinstance(transform, PowerTransform):
        if transform.power < 0:
            logcdf_zero = _logcdf_helper(measurable_input, 0)
            logcdf = pt.switch(
                pt.lt(backward_value, 0),
                logdiffexp(logcdf_zero, logcdf),
                pt.logaddexp(logccdf, logcdf_zero),
            )
    else:
        # We don't know if this Op is monotonically increasing/decreasing
        raise NotImplementedError

    if is_discrete:
        return logcdf

    # The jacobian is used to ensure a value in the supported domain was provided
    jacobian = transform.log_jac_det(value, *other_inputs)
    return pt.switch(pt.isnan(jacobian), -np.inf, logcdf)


@_icdf.register(ps.ScalarOp)
def measurable_transform_icdf(op, value, *inputs):
    """Compute the inverse CDF graph for a `MeasurabeTransform`."""
    selection = select_transform(op, inputs)
    if selection is None:
        raise NotImplementedError(f"Inverse CDF method not implemented for {type(op).__name__}")
    measurable_input, transform, other_inputs = selection

    # Do not apply rewrite to discrete variables
    if measurable_input.type.dtype.startswith("int"):
        raise NotImplementedError("icdf of transformed discrete variables not implemented")

    if isinstance(op, MONOTONICALLY_INCREASING_OPS):
        pass
    elif isinstance(op, MONOTONICALLY_DECREASING_OPS):
        value = 1 - value
    elif isinstance(op, Mul):
        [scale] = other_inputs
        value = pt.switch(pt.lt(scale, 0), 1 - value, value)
    elif isinstance(transform, PowerTransform):
        if transform.power < 0:
            # Note: Negative even powers will be rejected below when inverting the transform
            # For the remaining negative powers the function is decreasing with a jump around 0
            # We adjust the value with the mass below zero.
            # For non-negative RVs with cdf(0)=0, it simplifies to 1 - value
            cdf_zero = pt.exp(_logcdf_helper(measurable_input, 0))
            # Use nan to not mask invalid values accidentally
            value = pt.switch((value >= 0) & (value <= 1), value, np.nan)
            value = pt.switch(
                (cdf_zero > 0) & (value < cdf_zero),
                cdf_zero - value,
                1 + cdf_zero - value,
            )
    else:
        raise NotImplementedError

    input_icdf = _icdf_helper(measurable_input, value)
    icdf = transform.forward(input_icdf, *other_inputs)

    # Fail if transformation is not injective
    # A TensorVariable is returned in 1-to-1 inversions, and a tuple in 1-to-many
    if isinstance(transform.backward(icdf, *other_inputs), tuple):
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
    if contains_random(base):
        return None

    base = CheckParameterValue("base >= 0")(base, pt.all(pt.ge(base, 0.0)))

    return [pt.exp(pt.log(base) * inp_exponent)]


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


class ArcsinTransform(Transform):
    name = "arcsin"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arcsin(value)

    def backward(self, value, *inputs):
        return pt.sin(value)


class ArccosTransform(Transform):
    name = "arccos"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arccos(value)

    def backward(self, value, *inputs):
        return pt.cos(value)


class ArctanTransform(Transform):
    name = "arctan"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.arctan(value)

    def backward(self, value, *inputs):
        return pt.tan(value)


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


class ErfinvTransform(Transform):
    name = "erfinv"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erfinv(value)

    def backward(self, value, *inputs):
        return pt.erf(value)


class ErfcinvTransform(Transform):
    name = "erfcinv"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.erfcinv(value)

    def backward(self, value, *inputs):
        return pt.erfc(value)


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

        result = scan(
            fn=calc_delta_x,
            outputs_info=pt.ones_like(x),
            non_sequences=value,
            n_steps=10,
            return_updates=False,
        )
        return result[-1]


class LocTransform(Transform):
    name = "loc"
    ndim_supp = 0

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
    ndim_supp = 0

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
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.log(value)

    def backward(self, value, *inputs):
        return pt.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


class ExpTransform(Transform):
    name = "exp"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.exp(value)

    def backward(self, value, *inputs):
        return pt.log(value)

    def log_jac_det(self, value, *inputs):
        return -pt.log(value)


class AbsTransform(Transform):
    name = "abs"
    ndim_supp = 0

    def forward(self, value, *inputs):
        return pt.abs(value)

    def backward(self, value, *inputs):
        value = pt.switch(value >= 0, value, np.nan)
        return -value, value

    def log_jac_det(self, value, *inputs):
        return pt.switch(value >= 0, 0, np.nan)


class PowerTransform(Transform):
    name = "power"
    ndim_supp = 0

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
    ndim_supp = 0

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
    ndim_supp = 0

    def backward(self, value, *inputs):
        return pt.expit(value)

    def forward(self, value, *inputs):
        return pt.log(value / (1 - value))

    def log_jac_det(self, value, *inputs):
        sigmoid_value = pt.sigmoid(value)
        return pt.log(sigmoid_value) + pt.log1p(-sigmoid_value)


class SimplexTransform(Transform):
    name = "simplex"
    ndim_supp = 1

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
    ndim_supp = 0

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
        ndims_supp = [transform.ndim_supp for transform in transform_list]
        self.ndim_supp = max(ndims_supp) if None not in ndims_supp else None

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


@infer_support_axes.register(Elemwise)
def measure_elemwise(op, var):
    candidates = [inp for inp in var.owner.inputs if contains_random(inp)]
    if not candidates:
        return ()
    metas = []
    for inp in candidates:
        meta = support_axes(inp)
        if meta is None:
            raise UnsupportedObservation(
                "Elementwise transform of grouped events is not implemented"
            )
        axes = tuple(axis + var.ndim - inp.ndim for axis in meta)
        metas.append(axes)
    if len(set(metas)) != 1:
        raise UnsupportedObservation("The measurable input's support axes are ambiguous")
    return metas[0]


@rewrite_logprob_query.register(Elemwise)
def rewrite_elemwise_logprob(op, fgraph, query, **kwargs):
    return rewrite_logprob_query(op.scalar_op, fgraph, query, **kwargs)


@rewrite_logprob_query.register(ps.ScalarOp)
def rewrite_transform_logprob(op, fgraph, query, **kwargs):
    rv, value = query_parts(query)
    selection = select_transform(op, rv.owner.inputs)
    if selection is None:
        return None
    base, transform, other_inputs = selection
    backward = transform.backward(value, *other_inputs)
    axes = support_axes(rv)
    if isinstance(backward, tuple):
        # Event densities need all combinations of entrywise inverses, not just whole-array ones.
        if axes:
            return None
        # Alternative values copy the producer, so its other outputs cannot stay in a joint query.
        source_outputs = [out for source in density_sources(base) for out in source.owner.outputs]
        if other_query_uses(fgraph, source_outputs, {query}):
            return None
        term = pt.logaddexp(*(query_at_value(base, val) for val in backward))
    else:
        term = logprob_query(bind_value(fgraph, base, backward))
    jacobian = transform.log_jac_det(value, *other_inputs)
    if axes:
        jacobian = pt.broadcast_to(jacobian, value.shape).sum(axis=axes)
    return [pt.switch(pt.isnan(jacobian), -np.inf, term + jacobian)]
