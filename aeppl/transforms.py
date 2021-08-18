import abc
from functools import partial, singledispatch
from typing import Dict, List, Optional, Type, Union

import aesara.tensor as at
from aesara.gradient import jacobian
from aesara.graph.basic import Node, Variable
from aesara.graph.features import AlreadyThere, Feature
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op
from aesara.graph.opt import GlobalOptimizer, in2out, local_optimizer
from aesara.graph.utils import MetaType
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

from aeppl.logprob import _logprob


@singledispatch
def _default_transformed_rv(
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

        if base_op is not None and clsdict.get("default", False):
            # Create dispatch functions
            @_default_transformed_rv.register(type(base_op))
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
def transformed_logprob(op, values, *inputs, name=None, **kwargs):
    """Compute the log-likelihood graph for a `TransformedRV`.

    We assume that the value variable was back-transformed to be on the natural
    support of the respective random variable.
    """
    (value,) = values

    logprob = _logprob(op.base_op, values, *inputs, name=name, **kwargs)

    original_forward_value = op.transform.forward(value, *inputs)
    jacobian = op.transform.log_jac_det(original_forward_value, *inputs)

    if name:
        logprob.name = f"{name}_logprob"
        jacobian.name = f"{name}_logprob_jac"

    return logprob + jacobian


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
        new_op = _create_transformed_rv_op(node.op, transform)()
        trans_node = new_op.make_node(*node.inputs)
        trans_node.outputs[1].name = node.outputs[1].name

    # We now assume that the old value variable represents the *transformed space*.
    # This means that we need to replace all instance of the old value variable
    # with "inversely/un-" transformed versions of itself.
    new_value_var = transform.backward(value_var, *trans_node.inputs)
    if value_var.name and getattr(transform, "name", None):
        new_value_var.name = f"{value_var.name}_{transform.name}"

    # Map TransformedRV to new value var and delete old mapping
    new_rv_var = trans_node.outputs[1]
    rv_map_feature.rv_values[new_rv_var] = new_value_var
    del rv_map_feature.rv_values[rv_var]

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

    def log_jac_det(self, value, *inputs):
        sigmoid_value = at.sigmoid(value)
        return at.log(sigmoid_value) + at.log1p(-sigmoid_value)


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
) -> Type[TransformedRV]:
    """Create a new `TransformedRV` given a base `RandomVariable` `Op`

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
    rv_type_name = type(rv_op).__name__
    cls_dict = type(rv_op).__dict__.copy()
    rv_name = cls_dict.get("name", "")
    if rv_name:
        cls_dict["name"] = f"{rv_name}_{trans_name}"
    cls_dict["base_op"] = rv_op
    cls_dict["transform"] = transform
    cls_dict["default"] = default

    if cls_dict_extra is not None:
        cls_dict.update(cls_dict_extra)

    new_op_type = type(f"Transformed{rv_type_name}", (TransformedRV,), cls_dict)

    return new_op_type


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
    StickBreaking(),
)
