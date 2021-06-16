import abc
from functools import singledispatch
from typing import Dict, List, Optional

import aesara.tensor as at
from aesara.gradient import jacobian
from aesara.graph.basic import Node, Variable
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import local_optimizer
from aesara.graph.utils import MetaType
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

from aeppl.logprob import _logprob
from aeppl.utils import logsumexp


@singledispatch
def _transformed_rv(
    op: Op,
    node: Node,
) -> Optional[TensorVariable]:
    """Create a graph for a transformed log-probability of a ``RandomVariable``.

    This function dispatches on the type of ``op``, which should be a subclass
    of ``RandomVariable``.  If you want to implement new transforms for a
    ``RandomVariable``, register a function on this dispatcher.

    """
    return None


class DistributionMeta(MetaType):
    def __new__(cls, name, bases, clsdict):
        cls_res = super().__new__(cls, name, bases, clsdict)
        base_op = clsdict.get("base_op", None)

        if base_op is not None:
            # Create dispatch functions

            @_transformed_rv.register(type(base_op))
            def class_transformed_rv(op, node):
                new_op = cls_res()
                res = new_op.make_node(*node.inputs)
                res.outputs[1].name = node.outputs[1].name
                return res

        return cls_res


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
        # return at.log(at.abs_(jac))
        phi_inv = self.backward(value, *inputs)
        return at.log(at.nlinalg.det(at.atleast_2d(jacobian(phi_inv, [value]))))


class TransformedRV(RandomVariable, metaclass=DistributionMeta):
    r"""A base class for transformed `RandomVariable`\s."""


@_logprob.register(TransformedRV)
def trans_logprob(op, value, *inputs, name=None, **kwargs):

    logprob = _logprob(op.base_op, value, *inputs, name=name, **kwargs)

    jacobian = op.transform.log_jac_det(value, *inputs)

    if name:
        logprob.name = f"{name}_logprob"
        jacobian.name = f"{name}_logprob_jac"

    return logprob + jacobian


@local_optimizer(tracks=None)
def transform_logprob(fgraph: FunctionGraph, node: Node) -> Optional[List[Node]]:

    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return

    if not isinstance(node.op, RandomVariable):
        return

    trans_node = _transformed_rv(node.op, node)

    if trans_node is not None:
        # Get the old value variable and remove it from our value variables map
        rv_var = node.outputs[1]
        old_value_var = rv_map_feature.rv_values.pop(rv_var)

        transform = trans_node.op.transform

        # We now assume that the old value variable represents the *transformed space*.
        # This means that we need to replace all instance of the old value variable
        # with "inversely/un-" transformed versions of itself.
        new_value_var = transform.backward(old_value_var, *trans_node.inputs)
        rv_map_feature.rv_values[rv_var] = new_value_var

        new_rv_var = trans_node.outputs[1]

        if old_value_var.name and getattr(transform, "name", None):
            new_value_var.name = f"{old_value_var.name}_{transform.name}"

        rv_map_feature.rv_values[new_rv_var] = new_value_var

        return trans_node.outputs


class LogTransform(RVTransform):
    name = "log"

    def forward(self, value, *inputs):
        return at.log(value)

    def backward(self, value, *inputs):
        return at.exp(value)

    def log_jac_det(self, value, *inputs):
        return value


class IntervalTransform(RVTransform):
    name = "interval"

    def __init__(self, args_fn):
        self.args_fn = args_fn

    def forward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            return at.log(value - a) - at.log(b - value)
        elif a is not None:
            return at.log(value - a)
        elif b is not None:
            return at.log(b - value)

    def backward(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            sigmoid_x = at.sigmoid(value)
            return sigmoid_x * b + (1 - sigmoid_x) * a
        elif a is not None:
            return at.exp(value) + a
        elif b is not None:
            return b - at.exp(value)

    def log_jac_det(self, value, *inputs):
        a, b = self.args_fn(*inputs)

        if a is not None and b is not None:
            s = at.softplus(-value)
            return at.log(b - a) - 2 * s - value
        else:
            return value


class LogOddsTransform(RVTransform):
    name = "logodds"

    def backward(self, value, *inputs):
        return at.expit(value)

    def forward(self, value, *inputs):
        return at.log(value / (1 - value))


class StickBreaking(RVTransform):
    name = "stickbreaking"

    def forward(self, value, *inputs):
        log_value = at.log(value)
        shift = at.sum(log_value, -1, keepdims=True) / value.shape[-1]
        return log_value[..., :-1] - shift

    def backward(self, value, *inputs):
        value = at.concatenate([value, -at.sum(value, -1, keepdims=True)])
        exp_value_max = at.exp(value - at.max(value, -1, keepdims=True))
        return exp_value_max / at.sum(exp_value_max, -1, keepdims=True)

    def log_jac_det(self, value, *inputs):
        N = value.shape[-1] + 1
        sum_value = at.sum(value, -1, keepdims=True)
        value_sum_expanded = value + sum_value
        value_sum_expanded = at.concatenate(
            [value_sum_expanded, at.zeros(sum_value.shape)], -1
        )
        logsumexp_value_expanded = logsumexp(value_sum_expanded, -1, keepdims=True)
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


def create_transformed_rv_op(
    rv_op: Op, transform: RVTransform, cls_dict_extra: Optional[Dict] = None
) -> TransformedRV:

    trans_name = getattr(transform, "name", "")
    rv_type_name = type(rv_op).__name__
    cls_dict = type(rv_op).__dict__.copy()
    rv_name = cls_dict.get("name", "")
    if rv_name:
        cls_dict["name"] = (f"{rv_name}_{trans_name}",)
    cls_dict["base_op"] = rv_op

    if cls_dict_extra is not None:
        cls_dict.update(cls_dict_extra)

    new_op_type = type(f"Transformed{rv_type_name}", (TransformedRV,), cls_dict)
    new_op_type.transform = transform

    return new_op_type


TransformedUniformRV = create_transformed_rv_op(
    at.random.uniform, IntervalTransform(lambda *inputs: inputs[3:])
)
TransformedParetoRV = create_transformed_rv_op(
    at.random.pareto, IntervalTransform(lambda *inputs: (inputs[3], None))
)
TransformedTriangularRV = create_transformed_rv_op(
    at.random.triangular, IntervalTransform(lambda *inputs: (inputs[3], inputs[5]))
)
TransformedHalfNormalRV = create_transformed_rv_op(at.random.halfnormal, LogTransform())
TransformedWaldRV = create_transformed_rv_op(at.random.wald, LogTransform())
TransformedExponentialRV = create_transformed_rv_op(
    at.random.exponential, LogTransform()
)
TransformedLognormalRV = create_transformed_rv_op(at.random.lognormal, LogTransform())
TransformedHalfCauchyRV = create_transformed_rv_op(at.random.halfcauchy, LogTransform())
TransformedGammaRV = create_transformed_rv_op(at.random.gamma, LogTransform())
TransformedInvGammaRV = create_transformed_rv_op(at.random.invgamma, LogTransform())
TransformedChiSquareRV = create_transformed_rv_op(at.random.chisquare, LogTransform())
TransformedWeibullRV = create_transformed_rv_op(at.random.weibull, LogTransform())
TransformedBetaRV = create_transformed_rv_op(at.random.beta, LogOddsTransform())
TransformedVonMisesRV = create_transformed_rv_op(
    at.random.vonmises, CircularTransform()
)
TransformedDirichletRV = create_transformed_rv_op(at.random.dirichlet, StickBreaking())
