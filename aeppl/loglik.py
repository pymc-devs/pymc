from collections.abc import Mapping
from functools import singledispatch
from typing import Dict, Optional, Union

import aesara.tensor as at
from aesara import config
from aesara.gradient import disconnected_grad
from aesara.graph.basic import Constant, clone, graph_inputs, io_toposort
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import Op, compute_test_value
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

from aeppl.logpdf import _logpdf
from aeppl.utils import (
    extract_rv_and_value_vars,
    indices_from_subtensor,
    rvs_to_value_vars,
)


def loglik(
    var: TensorVariable,
    rv_values: Optional[
        Union[TensorVariable, Dict[TensorVariable, TensorVariable]]
    ] = None,
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
            return _loglik(
                var.owner.op,
                var,
                rv_values,
                *var.owner.inputs,
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

    # tmp_rv_values = rv_values.copy()
    # tmp_rv_values[rv_var] = rv_var

    logpdf_var = _logpdf(rv_node.op, rv_value_var, *dist_params, **kwargs)

    # Replace random variables with their value variables
    replacements = rv_values.copy()
    replacements.update({rv_var: rv_value, rv_value_var: rv_value})

    (logpdf_var,), _ = rvs_to_value_vars(
        (logpdf_var,),
        initial_replacements=replacements,
    )

    if sum:
        logpdf_var = at.sum(logpdf_var)

    # Recompute test values for the changes introduced by the replacements
    # above.
    if config.compute_test_value != "off":
        for node in io_toposort(graph_inputs((logpdf_var,)), (logpdf_var,)):
            compute_test_value(node)

    if rv_var.name is not None:
        logpdf_var.name = "__logp_%s" % rv_var.name

    return logpdf_var


@singledispatch
def _loglik(
    op: Op,
    var: TensorVariable,
    rvs_to_values: Dict[TensorVariable, TensorVariable],
    *inputs: TensorVariable,
    **kwargs,
):
    """Create a graph for the log-likelihood of a ``Variable``.

    This function dispatches on the type of ``op``.  If you want to implement
    new graphs for an ``Op``, register a new function on this dispatcher.

    The default returns a graph producing only zeros.

    """
    value_var = rvs_to_values.get(var, var)
    return at.zeros_like(value_var)


@_loglik.register(IncSubtensor)
@_loglik.register(AdvancedIncSubtensor)
@_loglik.register(AdvancedIncSubtensor1)
def incsubtensor_loglik(
    op, var, rvs_to_values, indexed_rv_var, rv_values, *indices, **kwargs
):

    index = indices_from_subtensor(getattr(op, "idx_list", None), indices)

    _, (new_rv_var,) = clone(
        tuple(
            v for v in graph_inputs((indexed_rv_var,)) if not isinstance(v, Constant)
        ),
        (indexed_rv_var,),
        copy_inputs=False,
        copy_orphans=False,
    )
    new_values = at.set_subtensor(disconnected_grad(new_rv_var)[index], rv_values)
    logp_var = loglik(indexed_rv_var, new_values, **kwargs)

    return logp_var


@_loglik.register(Subtensor)
@_loglik.register(AdvancedSubtensor)
@_loglik.register(AdvancedSubtensor1)
def subtensor_loglik(op, var, rvs_to_values, indexed_rv_var, *indices, **kwargs):

    index = indices_from_subtensor(getattr(op, "idx_list", None), indices)

    rv_value = rvs_to_values.get(var, getattr(var.tag, "value_var", None))

    if indexed_rv_var.owner and isinstance(indexed_rv_var.owner.op, RandomVariable):

        # We need to lift the index operation through the random variable so
        # that we have a new random variable consisting of only the relevant
        # subset of variables per the index.
        var_copy = var.owner.clone().default_output()
        fgraph = FunctionGraph(
            [i for i in graph_inputs((indexed_rv_var,)) if not isinstance(i, Constant)],
            [var_copy],
            clone=False,
        )

        (lifted_var,) = local_subtensor_rv_lift.transform(
            fgraph, fgraph.outputs[0].owner
        )

        new_rvs_to_values = rvs_to_values.copy()
        new_rvs_to_values[lifted_var] = rv_value

        logp_var = loglik(lifted_var, new_rvs_to_values, **kwargs)

        for idx_var in index:
            logp_var += loglik(idx_var, rvs_to_values, **kwargs)

    # TODO: We could add the constant case (i.e. `indexed_rv_var.owner is None`)
    else:
        raise NotImplementedError(
            f"`Subtensor` log-likelihood not implemented for {indexed_rv_var.owner}"
        )

    return logp_var
