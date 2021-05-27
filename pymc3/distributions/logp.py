#   Copyright 2020 The PyMC Developers
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

from collections.abc import Mapping
from functools import singledispatch
from typing import Dict, Optional, Union

import aesara.tensor as at
import numpy as np

from aesara import config
from aesara.gradient import disconnected_grad
from aesara.graph.basic import Constant, clone, graph_inputs, io_toposort
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op, compute_test_value
from aesara.graph.type import CType
from aesara.scalar.basic import Add, Mul
from aesara.tensor.elemwise import Elemwise
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.opt import local_subtensor_rv_lift
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from aesara.tensor.var import TensorVariable

from pymc3.aesaraf import (
    change_rv_size,
    extract_rv_and_value_vars,
    floatX,
    rvs_to_value_vars,
    walk_model,
)


@singledispatch
def logp_transform(op: Op):
    return None


def _get_scaling(total_size, shape, ndim):
    """
    Gets scaling constant for logp

    Parameters
    ----------
    total_size: int or list[int]
    shape: shape
        shape to scale
    ndim: int
        ndim hint

    Returns
    -------
    scalar
    """
    if total_size is None:
        coef = floatX(1)
    elif isinstance(total_size, int):
        if ndim >= 1:
            denom = shape[0]
        else:
            denom = 1
        coef = floatX(total_size) / floatX(denom)
    elif isinstance(total_size, (list, tuple)):
        if not all(isinstance(i, int) for i in total_size if (i is not Ellipsis and i is not None)):
            raise TypeError(
                "Unrecognized `total_size` type, expected "
                "int or list of ints, got %r" % total_size
            )
        if Ellipsis in total_size:
            sep = total_size.index(Ellipsis)
            begin = total_size[:sep]
            end = total_size[sep + 1 :]
            if Ellipsis in end:
                raise ValueError(
                    "Double Ellipsis in `total_size` is restricted, got %r" % total_size
                )
        else:
            begin = total_size
            end = []
        if (len(begin) + len(end)) > ndim:
            raise ValueError(
                "Length of `total_size` is too big, "
                "number of scalings is bigger that ndim, got %r" % total_size
            )
        elif (len(begin) + len(end)) == 0:
            return floatX(1)
        if len(end) > 0:
            shp_end = shape[-len(end) :]
        else:
            shp_end = np.asarray([])
        shp_begin = shape[: len(begin)]
        begin_coef = [floatX(t) / shp_begin[i] for i, t in enumerate(begin) if t is not None]
        end_coef = [floatX(t) / shp_end[i] for i, t in enumerate(end) if t is not None]
        coefs = begin_coef + end_coef
        coef = at.prod(coefs)
    else:
        raise TypeError(
            "Unrecognized `total_size` type, expected int or list of ints, got %r" % total_size
        )
    return at.as_tensor(floatX(coef))


def logpt(
    var: TensorVariable,
    rv_values: Optional[Union[TensorVariable, Dict[TensorVariable, TensorVariable]]] = None,
    *,
    jacobian: bool = True,
    scaling: bool = True,
    transformed: bool = True,
    cdf: bool = False,
    sum: bool = False,
    **kwargs,
) -> TensorVariable:
    """Create a measure-space (i.e. log-likelihood) graph for a random variable at a given point.

    The input `var` determines which log-likelihood graph is used and
    `rv_value` is that graph's input parameter.  For example, if `var` is
    the output of a ``NormalRV`` ``Op``, then the output is a graph of the
    density function for `var` set to the value `rv_value`.

    Parameters
    ==========
    var
        The `RandomVariable` output that determines the log-likelihood graph.
    rv_values
        A variable, or ``dict`` of variables, that represents the value of
        `var` in its log-likelihood.  If no `rv_value` is provided,
        ``var.tag.value_var`` will be checked and, when available, used.
    jacobian
        Whether or not to include the Jacobian term.
    scaling
        A scaling term to apply to the generated log-likelihood graph.
    transformed
        Apply transforms.
    cdf
        Return the log cumulative distribution.
    sum
        Sum the log-likelihood.

    """
    if not isinstance(rv_values, Mapping):
        rv_values = {var: rv_values} if rv_values is not None else {}

    rv_var, rv_value_var = extract_rv_and_value_vars(var)

    rv_value = rv_values.get(rv_var, rv_value_var)

    if rv_var is not None and rv_value is None:
        raise ValueError(f"No value variable specified or associated with {rv_var}")

    if rv_value is not None:
        rv_value = at.as_tensor(rv_value)

        if rv_var is not None:
            # Make sure that the value is compatible with the random variable
            rv_value = rv_var.type.filter_variable(rv_value.astype(rv_var.dtype))

        if rv_value_var is None:
            rv_value_var = rv_value

    if rv_var is None:
        if var.owner is not None:
            return _logp(
                var.owner.op,
                var,
                rv_values,
                *var.owner.inputs,
                jacobian=jacobian,
                scaling=scaling,
                transformed=transformed,
                cdf=cdf,
                sum=sum,
            )

        return at.zeros_like(var)

    rv_node = rv_var.owner

    rng, size, dtype, *dist_params = rv_node.inputs

    # Here, we plug the actual random variable into the log-likelihood graph,
    # because we want a log-likelihood graph that only contains
    # random variables.  This is important, because a random variable's
    # parameters can contain random variables themselves.
    # Ultimately, with a graph containing only random variables and
    # "deterministics", we can simply replace all the random variables with
    # their value variables and be done.
    tmp_rv_values = rv_values.copy()
    tmp_rv_values[rv_var] = rv_var

    if not cdf:
        logp_var = _logp(rv_node.op, rv_var, tmp_rv_values, *dist_params, **kwargs)
    else:
        logp_var = _logcdf(rv_node.op, rv_var, tmp_rv_values, *dist_params, **kwargs)

    transform = getattr(rv_value_var.tag, "transform", None) if rv_value_var else None

    if transform and transformed and not cdf and jacobian:
        transformed_jacobian = transform.jacobian_det(rv_var, rv_value)
        if transformed_jacobian:
            if logp_var.ndim > transformed_jacobian.ndim:
                logp_var = logp_var.sum(axis=-1)
            logp_var += transformed_jacobian

    # Replace random variables with their value variables
    replacements = rv_values.copy()
    replacements.update({rv_var: rv_value, rv_value_var: rv_value})

    (logp_var,), _ = rvs_to_value_vars(
        (logp_var,),
        apply_transforms=transformed and not cdf,
        initial_replacements=replacements,
    )

    if sum:
        logp_var = at.sum(logp_var)

    if scaling:
        logp_var *= _get_scaling(
            getattr(rv_var.tag, "total_size", None), rv_value.shape, rv_value.ndim
        )

    # Recompute test values for the changes introduced by the replacements
    # above.
    if config.compute_test_value != "off":
        for node in io_toposort(graph_inputs((logp_var,)), (logp_var,)):
            compute_test_value(node)

    if rv_var.name is not None:
        logp_var.name = "__logp_%s" % rv_var.name

    return logp_var


@singledispatch
def _logp(
    op: Op,
    var: TensorVariable,
    rvs_to_values: Dict[TensorVariable, TensorVariable],
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a log-likelihood graph.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.  If you want to implement new log-likelihood graphs
    for a `RandomVariable`, register a new function on this dispatcher.

    The default assumes that the log-likelihood of a term is a zero.

    """
    # value_var = rvs_to_values.get(var, var)
    # return at.zeros_like(value_var)
    raise NotImplementedError(f"Logp cannot be computed for op {op}")


@_logp.register(Elemwise)
def elemwise_logp(op, *args, **kwargs):
    return _logp(op.scalar_op, *args, **kwargs)


# TODO: Implement DimShuffle logp?
# @_logp.register(DimShuffle)
# def logp_dimshuffle(op, var, *args, **kwargs):
#     if var.owner and len(var.owner.inputs) == 1:
#         inp = var.owner.inputs[0]
#         if inp.owner and hasattr(inp.owner, 'op'):
#             return _logp(inp.owner.op, inp, *args, **kwargs)
#     raise NotImplementedError


@_logp.register(Add)
@_logp.register(Mul)
def linear_logp(
    op,
    var,
    rvs_to_values,
    *linear_inputs,
    transformed=True,
    sum=False,
    **kwargs,
):

    if len(linear_inputs) != 2:
        raise ValueError(f"Expected 2 inputs but got: {len(linear_inputs)}")

    # Find base_rv and constant inputs
    base_rv = []
    constant = []
    for inp in linear_inputs:
        res_ancestors = list(walk_model((inp,), walk_past_rvs=True))
        # unregistered variables do not contain a value_var tag
        res_unregistered_ancestors = [
            v
            for v in res_ancestors
            if v.owner
            and isinstance(v.owner.op, RandomVariable)
            and not getattr(v.tag, "value_var", False)
        ]
        if res_unregistered_ancestors:
            base_rv.append(inp)
        else:
            constant.append(inp)

    if len(base_rv) != 1:
        raise NotImplementedError(
            f"Logp of linear transform requires one branch with an unregistered RandomVariable but got {len(base_rv)}"
        )

    base_rv = base_rv[0]
    constant = constant[0]
    var_value = rvs_to_values.get(var, var)

    # Get logp of base_rv with transformed input
    if isinstance(op, Add):
        base_value = var_value - constant
    else:
        base_value = var_value / constant

    # Change base rv shape if needed
    if isinstance(base_rv.owner.op, RandomVariable):
        ndim_supp = base_rv.owner.op.ndim_supp
        if ndim_supp > 0:
            new_size = base_value.shape[:-ndim_supp]
        else:
            new_size = base_value.shape
        base_rv = change_rv_size(base_rv, new_size)

    var_logp = logpt(base_rv, {base_rv: base_value}, transformed=transformed, sum=False, **kwargs)

    # Apply product jacobian correction for continuous rvs
    if isinstance(op, Mul) and "float" in base_rv.dtype:
        var_logp -= at.log(at.abs_(constant))

    # Replace rvs in graph
    (var_logp,), _ = rvs_to_value_vars(
        (var_logp,),
        apply_transforms=transformed,
        initial_replacements=None,
    )

    if sum:
        var_logp = at.sum(var_logp)

    var_logp.name = f"__logp_{var.name}"
    return var_logp


def convert_indices(indices, entry):
    if indices and isinstance(entry, CType):
        rval = indices.pop(0)
        return rval
    elif isinstance(entry, slice):
        return slice(
            convert_indices(indices, entry.start),
            convert_indices(indices, entry.stop),
            convert_indices(indices, entry.step),
        )
    else:
        return entry


def indices_from_subtensor(idx_list, indices):
    """Compute a useable index tuple from the inputs of a ``*Subtensor**`` ``Op``."""
    return tuple(
        tuple(convert_indices(list(indices), idx) for idx in idx_list) if idx_list else indices
    )


@_logp.register(IncSubtensor)
@_logp.register(AdvancedIncSubtensor)
@_logp.register(AdvancedIncSubtensor1)
def incsubtensor_logp(op, var, rvs_to_values, indexed_rv_var, rv_values, *indices, **kwargs):

    index = indices_from_subtensor(getattr(op, "idx_list", None), indices)

    _, (new_rv_var,) = clone(
        tuple(v for v in graph_inputs((indexed_rv_var,)) if not isinstance(v, Constant)),
        (indexed_rv_var,),
        copy_inputs=False,
        copy_orphans=False,
    )
    new_values = at.set_subtensor(disconnected_grad(new_rv_var)[index], rv_values)
    logp_var = logpt(indexed_rv_var, new_values, **kwargs)

    return logp_var


@_logp.register(Subtensor)
@_logp.register(AdvancedSubtensor)
@_logp.register(AdvancedSubtensor1)
def subtensor_logp(op, var, rvs_to_values, indexed_rv_var, *indices, **kwargs):

    index = indices_from_subtensor(getattr(op, "idx_list", None), indices)

    rv_value = rvs_to_values.get(var, getattr(var.tag, "value_var", None))

    if indexed_rv_var.owner and isinstance(indexed_rv_var.owner.op, RandomVariable):

        # We need to lift the index operation through the random variable so
        # that we have a new random variable consisting of only the relevant
        # subset of variables per the index.
        var_copy = var.owner.clone().default_output()
        fgraph = FunctionGraph(
            outputs=[var_copy],
            clone=False,
        )

        (lifted_var,) = local_subtensor_rv_lift.transform(fgraph, fgraph.outputs[0].owner)

        new_rvs_to_values = rvs_to_values.copy()
        new_rvs_to_values[lifted_var] = rv_value

        logp_var = logpt(lifted_var, new_rvs_to_values, **kwargs)

        for idx_var in index:
            logp_var += logpt(idx_var, rvs_to_values, **kwargs)

    # TODO: We could add the constant case (i.e. `indexed_rv_var.owner is None`)
    else:
        raise NotImplementedError(
            f"`Subtensor` log-likelihood not implemented for {indexed_rv_var.owner}"
        )

    return logp_var


def logcdf(*args, **kwargs):
    """Create a log-CDF graph."""
    return logpt(*args, cdf=True, **kwargs)


@singledispatch
def _logcdf(op, values, *args, **kwargs):
    """Create a log-CDF graph.

    This function dispatches on the type of `op`, which should be a subclass
    of `RandomVariable`.  If you want to implement new log-CDF graphs
    for a `RandomVariable`, register a new function on this dispatcher.

    """
    raise NotImplementedError()


def logpt_sum(*args, **kwargs):
    """Return the sum of the logp values for the given observations.

    Subclasses can use this to improve the speed of logp evaluations
    if only the sum of the logp values is needed.
    """
    return logpt(*args, sum=True, **kwargs)
