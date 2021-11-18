import abc
from copy import copy
from functools import partial, singledispatch
from typing import Callable, Dict, List, Optional, Tuple, Union

import aesara.tensor as at
from aesara.gradient import DisconnectedType, jacobian
from aesara.graph.basic import Apply, Node, Variable
from aesara.graph.features import AlreadyThere, Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import GlobalOptimizer, in2out, local_optimizer
from aesara.tensor.basic_opt import (
    register_specialize,
    register_stabilize,
    register_useless,
)
from aesara.tensor.var import TensorVariable

from aeppl.abstract import MeasurableVariable
from aeppl.logprob import _logprob


@singledispatch
def _default_transformed_rv(
    op: Op,
    node: Node,
) -> Optional[Apply]:
    """Create a node for a transformed log-probability of a `MeasurableVariable`.

    This function dispatches on the type of `op`.  If you want to implement
    new transforms for a `MeasurableVariable`, register a function on this
    dispatcher.

    """
    return None


class TransformedVariable(Op):
    """A no-op that identifies a transform and its un-transformed input."""

    view_map = {0: [0]}

    def make_node(self, tran_value: TensorVariable, value: TensorVariable):
        return Apply(self, [tran_value, value], [tran_value.type()])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError(
            "These `Op`s should be removed from graphs used for computation."
        )

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
@local_optimizer([TransformedVariable])
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
        # return at.log(at.abs_(jac))
        phi_inv = self.backward(value, *inputs)
        return at.log(at.nlinalg.det(at.atleast_2d(jacobian(phi_inv, [value]))))


class DefaultTransformSentinel:
    pass


DEFAULT_TRANSFORM = DefaultTransformSentinel()


@local_optimizer(tracks=None)
def transform_values(fgraph: FunctionGraph, node: Node) -> Optional[List[Node]]:
    """Apply transforms to value variables.

    It is assumed that the input value variables correspond to forward
    transformations, usually chosen in such a way that the values are
    unconstrained on the real line.

    For example, if ``Y = halfnormal(...)``, we assume the respective value
    variable is specified on the log scale and back-transform it to obtain
    ``Y`` on the natural scale.
    """

    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)
    values_to_transforms = getattr(fgraph, "values_to_transforms", None)

    if rv_map_feature is None or values_to_transforms is None:
        return None  # pragma: no cover

    try:
        rv_var = node.default_output()
        rv_var_out_idx = node.outputs.index(rv_var)
    except ValueError:
        return None

    value_var = rv_map_feature.rv_values.get(rv_var, None)
    if value_var is None:
        return None

    transform = values_to_transforms.get(value_var, None)

    if transform is None:
        return None
    elif transform is DEFAULT_TRANSFORM:
        trans_node = _default_transformed_rv(node.op, node)
        if trans_node is None:
            return None
        transform = trans_node.op.transform
    else:
        new_op = _create_transformed_rv_op(node.op, transform)
        # Create a new `Apply` node and outputs
        trans_node = node.clone()
        trans_node.op = new_op
        trans_node.outputs[rv_var_out_idx].name = node.outputs[rv_var_out_idx].name

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    new_value_var = transformed_variable(
        transform.backward(value_var, *trans_node.inputs), value_var
    )
    if value_var.name and getattr(transform, "name", None):
        new_value_var.name = f"{value_var.name}_{transform.name}"

    rv_map_feature.update_rv_maps(
        rv_var, new_value_var, trans_node.outputs[rv_var_out_idx]
    )

    return trans_node.outputs


class TransformValuesMapping(Feature):
    r"""A `Feature` that maintains a map between value variables and their transforms."""

    def __init__(self, values_to_transforms):
        self.values_to_transforms = values_to_transforms

    def on_attach(self, fgraph):
        if hasattr(fgraph, "values_to_transforms"):
            raise AlreadyThere()

        fgraph.values_to_transforms = self.values_to_transforms


class TransformValuesOpt(GlobalOptimizer):
    r"""Transforms value variables according to a map and/or per-`RandomVariable` defaults."""

    default_transform_opt = in2out(transform_values, ignore_newtrees=True)

    def __init__(
        self,
        values_to_transforms: Dict[
            TensorVariable, Union[RVTransform, DefaultTransformSentinel, None]
        ],
    ):
        """
        Parameters
        ==========
        values_to_transforms
            Mapping between value variables and their transformations.  Each
            value variable can be assigned one of `RVTransform`,
            ``DEFAULT_TRANSFORM``, or ``None``. If a transform is not specified
            for a specific value variable it will not be transformed.

        """

        self.values_to_transforms = values_to_transforms

    def add_requirements(self, fgraph):
        values_transforms_feature = TransformValuesMapping(self.values_to_transforms)
        fgraph.attach_feature(values_transforms_feature)

    def apply(self, fgraph: FunctionGraph):
        return self.default_transform_opt.optimize(fgraph)


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

    def __init__(
        self, args_fn: Callable[..., Tuple[Optional[Variable], Optional[Variable]]]
    ):
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


class Simplex(RVTransform):
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
        value_sum_expanded = at.concatenate(
            [value_sum_expanded, at.zeros(sum_value.shape)], -1
        )
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


def _create_transformed_rv_op(
    rv_op: Op,
    transform: RVTransform,
    *,
    default: bool = False,
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
    default
        If ``False`` do not make `transform` the default transform for `rv_op`.
    cls_dict_extra
        Additional class members to add to the constructed `TransformedRV`.

    """

    trans_name = getattr(transform, "name", "transformed")
    rv_op_type = type(rv_op)
    rv_type_name = rv_op_type.__name__
    cls_dict = rv_op_type.__dict__.copy()
    rv_name = cls_dict.get("name", "")
    if rv_name:
        cls_dict["name"] = f"{rv_name}_{trans_name}"
    cls_dict["transform"] = transform

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
        (value,) = values

        logprob = _logprob(rv_op, values, *inputs, **kwargs)

        if use_jacobian:
            assert isinstance(value.owner.op, TransformedVariable)
            original_forward_value = value.owner.inputs[1]
            jacobian = op.transform.log_jac_det(original_forward_value, *inputs)
            logprob += jacobian

        return logprob

    transform_op = rv_op_type if default else new_op_type

    @_default_transformed_rv.register(transform_op)
    def class_transformed_rv(op, node):
        new_op = new_op_type()
        res = new_op.make_node(*node.inputs)
        res.outputs[1].name = node.outputs[1].name
        return res

    new_op = copy(rv_op)
    new_op.__class__ = new_op_type

    return new_op


create_default_transformed_rv_op = partial(_create_transformed_rv_op, default=True)


TransformedUniformRV = create_default_transformed_rv_op(
    at.random.uniform,
    # inputs[3] = lower; inputs[4] = upper
    IntervalTransform(lambda *inputs: (inputs[3], inputs[4])),
)
TransformedParetoRV = create_default_transformed_rv_op(
    at.random.pareto,
    # inputs[3] = alpha
    IntervalTransform(lambda *inputs: (inputs[3], None)),
)
TransformedTriangularRV = create_default_transformed_rv_op(
    at.random.triangular,
    # inputs[3] = lower; inputs[5] = upper
    IntervalTransform(lambda *inputs: (inputs[3], inputs[5])),
)
TransformedHalfNormalRV = create_default_transformed_rv_op(
    at.random.halfnormal,
    # inputs[3] = loc
    IntervalTransform(lambda *inputs: (inputs[3], None)),
)
TransformedWaldRV = create_default_transformed_rv_op(
    at.random.wald,
    LogTransform(),
)
TransformedExponentialRV = create_default_transformed_rv_op(
    at.random.exponential,
    LogTransform(),
)
TransformedLognormalRV = create_default_transformed_rv_op(
    at.random.lognormal,
    LogTransform(),
)
TransformedHalfCauchyRV = create_default_transformed_rv_op(
    at.random.halfcauchy,
    LogTransform(),
)
TransformedGammaRV = create_default_transformed_rv_op(
    at.random.gamma,
    LogTransform(),
)
TransformedInvGammaRV = create_default_transformed_rv_op(
    at.random.invgamma,
    LogTransform(),
)
TransformedChiSquareRV = create_default_transformed_rv_op(
    at.random.chisquare,
    LogTransform(),
)
TransformedWeibullRV = create_default_transformed_rv_op(
    at.random.weibull,
    LogTransform(),
)
TransformedBetaRV = create_default_transformed_rv_op(
    at.random.beta,
    LogOddsTransform(),
)
TransformedVonMisesRV = create_default_transformed_rv_op(
    at.random.vonmises,
    CircularTransform(),
)
TransformedDirichletRV = create_default_transformed_rv_op(
    at.random.dirichlet,
    Simplex(),
)
