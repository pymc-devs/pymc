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

import warnings

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import pytensor.tensor as pt

from pytensor.graph.basic import (
    Constant,
    Variable,
    ancestors,
)
from pytensor.graph.rewriting.basic import GraphRewriter, NodeRewriter
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import (
    MeasurableOp,
    _icdf_helper,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import cleanup_ir, construct_ir_fgraph
from pymc.logprob.transform_value import TransformValuesRewrite
from pymc.logprob.transforms import Transform
from pymc.logprob.utils import get_related_valued_nodes, rvs_in_graph
from pymc.pytensorf import replace_vars_in_graphs

TensorLike: TypeAlias = Variable | float | np.ndarray


def _find_unallowed_rvs_in_graph(graph):
    from pymc.data import MinibatchIndexRV
    from pymc.distributions.simulator import SimulatorRV

    return {
        rv
        for rv in rvs_in_graph(graph)
        if not isinstance(rv.owner.op, SimulatorRV | MinibatchIndexRV)
    }


def _warn_rvs_in_inferred_graph(graph: TensorVariable | Sequence[TensorVariable]):
    """Issue warning if any RVs are found in graph.

    RVs are usually an (implicit) conditional input of the derived probability expression,
    and meant to be replaced by respective value variables before evaluation.
    However, when the IR graph is built, any non-input nodes (including RVs) are cloned,
    breaking the link with the original ones.
    This makes it impossible (or difficult) to replace it by the respective values afterward,
    so we instruct users to do it beforehand.
    """
    rvs_in_graph = _find_unallowed_rvs_in_graph(graph)
    if rvs_in_graph:
        warnings.warn(
            f"RandomVariables {rvs_in_graph} were found in the derived graph. "
            "These variables are a clone and do not match the original ones on identity.\n"
            "If you are deriving a quantity that depends on model RVs, use `model.replace_rvs_by_values` first. "
            "For example: `logp(model.replace_rvs_by_values([rv])[0], value)`",
            stacklevel=3,
        )


def _deprecate_warn_missing_rvs(warn_rvs, kwargs):
    if "warn_missing_rvs" in kwargs:
        warnings.warn(
            "Argument `warn_missing_rvs` was renamed to `warn_rvs` and will be removed in a future release",
            FutureWarning,
        )
        if warn_rvs is None:
            warn_rvs = kwargs.pop("warn_missing_rvs")
        else:
            raise ValueError("Can't set both warn_rvs and warn_missing_rvs")
    else:
        if warn_rvs is None:
            warn_rvs = True
    return warn_rvs, kwargs


def logp(rv: TensorVariable, value: TensorLike, warn_rvs=None, **kwargs) -> TensorVariable:
    """Create a graph for the log-probability of a random variable.

    Parameters
    ----------
    rv : TensorVariable
    value : tensor_like
        Should be the same type (shape and dtype) as the rv.
    warn_rvs : bool, default True
        Warn if RVs were found in the logp graph.
        This can happen when a variable has other other random variables as inputs.
        In that case, those random variables should be replaced by their respective values.
        `pymc.logprob.conditional_logp` can also be used as an alternative.

    Returns
    -------
    logp : TensorVariable

    Raises
    ------
    RuntimeError
        If the logp cannot be derived.

    Examples
    --------
    Create a compiled function that evaluates the logp of a variable

    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt

        mu = pt.scalar("mu")
        rv = pm.Normal.dist(mu, 1.0)

        value = pt.scalar("value")
        rv_logp = pm.logp(rv, value)

        # Use .eval() for debugging
        print(rv_logp.eval({value: 0.9, mu: 0.0}))  # -1.32393853

        # Compile a function for repeated evaluations
        rv_logp_fn = pm.compile_pymc([value, mu], rv_logp)
        print(rv_logp_fn(value=0.9, mu=0.0))  # -1.32393853


    Derive the graph for a transformation of a RandomVariable

    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt

        mu = pt.scalar("mu")
        rv = pm.Normal.dist(mu, 1.0)
        exp_rv = pt.exp(rv)

        value = pt.scalar("value")
        exp_rv_logp = pm.logp(exp_rv, value)

        # Use .eval() for debugging
        print(exp_rv_logp.eval({value: 0.9, mu: 0.0}))  # -0.81912844

        # Compile a function for repeated evaluations
        exp_rv_logp_fn = pm.compile_pymc([value, mu], exp_rv_logp)
        print(exp_rv_logp_fn(value=0.9, mu=0.0))  # -0.81912844


    Define a CustomDist logp

    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt


        def normal_logp(value, mu, sigma):
            return pm.logp(pm.Normal.dist(mu, sigma), value)


        with pm.Model() as model:
            mu = pm.Normal("mu")
            sigma = pm.HalfNormal("sigma")
            pm.CustomDist("x", mu, sigma, logp=normal_logp)

    """
    warn_rvs, kwargs = _deprecate_warn_missing_rvs(warn_rvs, kwargs)

    value = pt.as_tensor_variable(value, dtype=rv.dtype)
    try:
        return _logprob_helper(rv, value, **kwargs)
    except NotImplementedError:
        fgraph = construct_ir_fgraph({rv: value})
        [ir_valued_var] = fgraph.outputs
        [ir_rv, ir_value] = ir_valued_var.owner.inputs
        expr = _logprob_helper(ir_rv, ir_value, **kwargs)
        cleanup_ir([expr])
        if warn_rvs:
            _warn_rvs_in_inferred_graph(expr)
        return expr


def logcdf(rv: TensorVariable, value: TensorLike, warn_rvs=None, **kwargs) -> TensorVariable:
    """Create a graph for the log-CDF of a random variable.

    Parameters
    ----------
    rv : TensorVariable
    value : tensor_like
        Should be the same type (shape and dtype) as the rv.
    warn_rvs : bool, default True
        Warn if RVs were found in the logcdf graph.
        This can happen when a variable has other random variables as inputs.
        In that case, those random variables should be replaced by their respective values.

    Returns
    -------
    logp : TensorVariable

    Raises
    ------
    RuntimeError
        If the logcdf cannot be derived.

    Examples
    --------
    Create a compiled function that evaluates the logcdf of a variable

    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt

        mu = pt.scalar("mu")
        rv = pm.Normal.dist(mu, 1.0)

        value = pt.scalar("value")
        rv_logcdf = pm.logcdf(rv, value)

        # Use .eval() for debugging
        print(rv_logcdf.eval({value: 0.9, mu: 0.0}))  # -0.2034146

        # Compile a function for repeated evaluations
        rv_logcdf_fn = pm.compile_pymc([value, mu], rv_logcdf)
        print(rv_logcdf_fn(value=0.9, mu=0.0))  # -0.2034146


    Derive the graph for a transformation of a RandomVariable

    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt

        mu = pt.scalar("mu")
        rv = pm.Normal.dist(mu, 1.0)
        exp_rv = pt.exp(rv)

        value = pt.scalar("value")
        exp_rv_logcdf = pm.logcdf(exp_rv, value)

        # Use .eval() for debugging
        print(exp_rv_logcdf.eval({value: 0.9, mu: 0.0}))  # -0.78078813

        # Compile a function for repeated evaluations
        exp_rv_logcdf_fn = pm.compile_pymc([value, mu], exp_rv_logcdf)
        print(exp_rv_logcdf_fn(value=0.9, mu=0.0))  # -0.78078813


    Define a CustomDist logcdf

    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt


        def normal_logcdf(value, mu, sigma):
            return pm.logcdf(pm.Normal.dist(mu, sigma), value)


        with pm.Model() as model:
            mu = pm.Normal("mu")
            sigma = pm.HalfNormal("sigma")
            pm.CustomDist("x", mu, sigma, logcdf=normal_logcdf)

    """
    warn_rvs, kwargs = _deprecate_warn_missing_rvs(warn_rvs, kwargs)
    value = pt.as_tensor_variable(value, dtype=rv.dtype)
    try:
        return _logcdf_helper(rv, value, **kwargs)
    except NotImplementedError:
        # Try to rewrite rv
        fgraph = construct_ir_fgraph({rv: value})
        [ir_valued_rv] = fgraph.outputs
        [ir_rv, ir_value] = ir_valued_rv.owner.inputs
        expr = _logcdf_helper(ir_rv, ir_value, **kwargs)
        cleanup_ir([expr])
        if warn_rvs:
            _warn_rvs_in_inferred_graph(expr)
        return expr


def icdf(rv: TensorVariable, value: TensorLike, warn_rvs=None, **kwargs) -> TensorVariable:
    """Create a graph for the inverse CDF of a random variable.

    Parameters
    ----------
    rv : TensorVariable
    value : tensor_like
        Should be the same type (shape and dtype) as the rv.
    warn_rvs : bool, default True
        Warn if RVs were found in the icdf graph.
        This can happen when a variable has other random variables as inputs.
        In that case, those random variables should be replaced by their respective values.

    Returns
    -------
    icdf : TensorVariable

    Raises
    ------
    RuntimeError
        If the icdf cannot be derived.

    Examples
    --------
    Create a compiled function that evaluates the icdf of a variable

    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt

        mu = pt.scalar("mu")
        rv = pm.Normal.dist(mu, 1.0)

        value = pt.scalar("value")
        rv_icdf = pm.icdf(rv, value)

        # Use .eval() for debugging
        print(rv_icdf.eval({value: 0.9, mu: 0.0}))  # 1.28155157

        # Compile a function for repeated evaluations
        rv_icdf_fn = pm.compile_pymc([value, mu], rv_icdf)
        print(rv_icdf_fn(value=0.9, mu=0.0))  # 1.28155157


    Derive the graph for a transformation of a RandomVariable

    .. code-block:: python

        import pymc as pm
        import pytensor.tensor as pt

        mu = pt.scalar("mu")
        rv = pm.Normal.dist(mu, 1.0)
        exp_rv = pt.exp(rv)

        value = pt.scalar("value")
        exp_rv_icdf = pm.icdf(exp_rv, value)

        # Use .eval() for debugging
        print(exp_rv_icdf.eval({value: 0.9, mu: 0.0}))  # 3.60222448

        # Compile a function for repeated evaluations
        exp_rv_icdf_fn = pm.compile_pymc([value, mu], exp_rv_icdf)
        print(exp_rv_icdf_fn(value=0.9, mu=0.0))  # 3.60222448

    """
    warn_rvs, kwargs = _deprecate_warn_missing_rvs(warn_rvs, kwargs)
    value = pt.as_tensor_variable(value, dtype="floatX")
    try:
        return _icdf_helper(rv, value, **kwargs)
    except NotImplementedError:
        # Try to rewrite rv
        fgraph = construct_ir_fgraph({rv: value})
        [ir_valued_rv] = fgraph.outputs
        [ir_rv, ir_value] = ir_valued_rv.owner.inputs
        expr = _icdf_helper(ir_rv, ir_value, **kwargs)
        cleanup_ir([expr])
        if warn_rvs:
            _warn_rvs_in_inferred_graph(expr)
        return expr


RVS_IN_JOINT_LOGP_GRAPH_MSG = (
    "Random variables detected in the logp graph: %s.\n"
    "This can happen when DensityDist logp or Interval transform functions reference nonlocal variables,\n"
    "or when not all rvs have a corresponding value variable."
)


def conditional_logp(
    rv_values: dict[TensorVariable, TensorVariable],
    warn_rvs=None,
    ir_rewriter: GraphRewriter | None = None,
    extra_rewrites: GraphRewriter | NodeRewriter | None = None,
    **kwargs,
) -> dict[TensorVariable, TensorVariable]:
    r"""Create a map between variables and conditional logps such that the sum is their joint logp.

    The `rv_values` dictionary specifies a joint probability graph defined by
    pairs of random variables and respective measure-space input parameters

    For example, consider the following

    .. code-block:: python

        import pytensor.tensor as pt

        sigma2_rv = pt.random.invgamma(0.5, 0.5)
        Y_rv = pt.random.normal(0, pt.sqrt(sigma2_rv))

    This graph for ``Y_rv`` is equivalent to the following hierarchical model:

    .. math::

        \sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5) \\
        Y \sim& \operatorname{N}(0, \sigma^2)

    If we create a value variable for ``Y_rv``, i.e. ``y_vv = pt.scalar("y")``,
    the graph of ``conditional_logp({Y_rv: y_vv})`` is equivalent to the
    conditional log-probability :math:`\log p_{Y \mid \sigma^2}(y \mid s^2)`, with a stochastic
    ``sigma2_rv``.

    If we specify a value variable for ``sigma2_rv``, i.e.
    ``s2_vv = pt.scalar("s2")``, then ``conditional_logp({Y_rv: y_vv, sigma2_rv: s2_vv})``
    yields the conditional log-probabilities of the two variables.
    The sum of the two terms gives their joint log-probability.

    .. math::

        \log p_{Y, \sigma^2}(y, s^2) =
            \log p_{Y \mid \sigma^2}(y \mid s^2) + \log p_{\sigma^2}(s^2)


    Parameters
    ----------
    rv_values: dict
        A ``dict`` of variables that maps stochastic elements
        (e.g. `RandomVariable`\s) to symbolic `Variable`\s representing their
        values in a log-probability.
    warn_rvs : bool, default True
        When ``True``, issue a warning when a `RandomVariable` is found in
        the logp graph and doesn't have a corresponding value variable specified in
        `rv_values`.
    ir_rewriter
        Rewriter that produces the intermediate representation of Measurable Variables.
    extra_rewrites
        Extra rewrites to be applied (e.g. reparameterizations, transforms,
        etc.)

    Returns
    -------
    values_to_logps: dict
        A ``dict`` that maps each value variable to the conditional log-probability term derived
        from the respective `RandomVariable`.

    """
    warn_rvs, kwargs = _deprecate_warn_missing_rvs(warn_rvs, kwargs)

    fgraph = construct_ir_fgraph(rv_values, ir_rewriter=ir_rewriter)

    if extra_rewrites is not None:
        extra_rewrites.rewrite(fgraph)

    # Walk the graph from its inputs to its outputs and construct the
    # log-probability
    replacements = {}

    # To avoid cloning the value variables (or ancestors of value variables),
    # we map them to themselves in the `replacements` `dict`
    # (i.e. entries already existing in `replacements` aren't cloned)
    replacements.update(
        {v: v for v in ancestors(rv_values.values()) if not isinstance(v, Constant)}
    )

    # Walk the graph from its inputs to its outputs and construct the
    # log-probability
    values_to_logprobs = {}
    original_values = tuple(rv_values.values())

    # TODO: This seems too convoluted, can we just replace all RVs by their values,
    #  except for the fgraph outputs (for which we want to call _logprob on)?
    for node in fgraph.toposort():
        if not isinstance(node.op, MeasurableOp):
            continue

        valued_nodes = get_related_valued_nodes(fgraph, node)

        if not valued_nodes:
            continue

        node_rvs = [valued_var.inputs[0] for valued_var in valued_nodes]
        node_values = [valued_var.inputs[1] for valued_var in valued_nodes]
        node_output_idxs = [
            fgraph.outputs.index(valued_var.outputs[0]) for valued_var in valued_nodes
        ]

        # Replace `RandomVariable`s in the inputs with value variables.
        # Also, store the results in the `replacements` map for the nodes that follow.
        for node_rv, node_value in zip(node_rvs, node_values):
            replacements[node_rv] = node_value

        remapped_vars = replace_vars_in_graphs(
            graphs=node_values + list(node.inputs),
            replacements=replacements,
        )
        node_values = remapped_vars[: len(node_values)]
        node_inputs = remapped_vars[len(node_values) :]

        node_logprobs = _logprob(
            node.op,
            node_values,
            *node_inputs,
            **kwargs,
        )

        if not isinstance(node_logprobs, list | tuple):
            node_logprobs = [node_logprobs]

        for node_output_idx, node_value, node_logprob in zip(
            node_output_idxs, node_values, node_logprobs
        ):
            original_value = original_values[node_output_idx]

            if original_value.name:
                node_logprob.name = f"{original_value.name}_logprob"

            if original_value in values_to_logprobs:
                raise ValueError(
                    f"More than one logprob term was assigned to the value var {original_value}"
                )

            values_to_logprobs[original_value] = node_logprob

    missing_value_terms = set(original_values) - set(values_to_logprobs)
    if missing_value_terms:
        raise RuntimeError(
            f"The logprob terms of the following value variables could not be derived: {missing_value_terms}"
        )

    logprobs = list(values_to_logprobs.values())
    cleanup_ir(logprobs)

    if warn_rvs:
        rvs_in_logp_expressions = _find_unallowed_rvs_in_graph(logprobs)
        if rvs_in_logp_expressions:
            warnings.warn(RVS_IN_JOINT_LOGP_GRAPH_MSG % rvs_in_logp_expressions, UserWarning)

    return values_to_logprobs


def transformed_conditional_logp(
    rvs: Sequence[TensorVariable],
    *,
    rvs_to_values: dict[TensorVariable, TensorVariable],
    rvs_to_transforms: dict[TensorVariable, Transform],
    jacobian: bool = True,
    **kwargs,
) -> list[TensorVariable]:
    """Thin wrapper around conditional_logprob, which creates a value transform rewrite.

    This helper will only return the subset of logprob terms corresponding to `rvs`.
    All rvs_to_values and rvs_to_transforms mappings are required.
    """
    transform_rewrite = None
    values_to_transforms = {
        rvs_to_values[rv]: transform
        for rv, transform in rvs_to_transforms.items()
        if transform is not None
    }
    if values_to_transforms:
        # There seems to be an incorrect type hint in TransformValuesRewrite
        transform_rewrite = TransformValuesRewrite(values_to_transforms)  # type: ignore[arg-type]

    kwargs.setdefault("warn_rvs", False)
    temp_logp_terms = conditional_logp(
        rvs_to_values,
        extra_rewrites=transform_rewrite,
        use_jacobian=jacobian,
        **kwargs,
    )

    # The function returns the logp for every single value term we provided to it.
    # This includes the extra values we plugged in above, so we filter those we
    # actually wanted in the same order they were given in.
    logp_terms = {}
    for rv in rvs:
        value_var = rvs_to_values[rv]
        logp_terms[value_var] = temp_logp_terms[value_var]

    logp_terms_list = list(logp_terms.values())

    rvs_in_logp_expressions = _find_unallowed_rvs_in_graph(logp_terms_list)
    if rvs_in_logp_expressions:
        raise ValueError(RVS_IN_JOINT_LOGP_GRAPH_MSG % rvs_in_logp_expressions)

    return logp_terms_list


def factorized_joint_logprob(*args, **kwargs):
    warnings.warn(
        "`factorized_joint_logprob` was renamed to `conditional_logp`. "
        "The function will be removed in a future release",
        FutureWarning,
    )
    return conditional_logp(*args, **kwargs)


def joint_logp(*args, **kwargs):
    warnings.warn(
        "`joint_logp` was renamed to `transformed_conditional_logp`. "
        "The function will be removed in a future release",
        FutureWarning,
    )
    return transformed_conditional_logp(*args, **kwargs)
