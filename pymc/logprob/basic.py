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

import warnings

from collections import deque
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pytensor.tensor as pt

from pytensor import config
from pytensor.graph.basic import (
    Constant,
    Variable,
    ancestors,
    graph_inputs,
    io_toposort,
)
from pytensor.graph.op import compute_test_value
from pytensor.graph.rewriting.basic import GraphRewriter, NodeRewriter
from pytensor.tensor.var import TensorVariable
from typing_extensions import TypeAlias

from pymc.logprob.abstract import (
    MeasurableVariable,
    _icdf_helper,
    _logcdf_helper,
    _logprob,
    _logprob_helper,
)
from pymc.logprob.rewriting import cleanup_ir, construct_ir_fgraph
from pymc.logprob.transforms import RVTransform, TransformValuesRewrite
from pymc.logprob.utils import find_rvs_in_graph, rvs_to_value_vars

TensorLike: TypeAlias = Union[Variable, float, np.ndarray]


def _find_unallowed_rvs_in_graph(graph):
    from pymc.data import MinibatchIndexRV
    from pymc.distributions.simulator import SimulatorRV

    return {
        rv
        for rv in find_rvs_in_graph(graph)
        if not isinstance(rv.owner.op, (SimulatorRV, MinibatchIndexRV))
    }


def _warn_rvs_in_inferred_graph(graph: Union[TensorVariable, Sequence[TensorVariable]]):
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
            "If you are deriving a quantity that depends on model RVs, use `model.replace_rvs_by_values` first. For example: "
            "`logp(model.replace_rvs_by_values([rv])[0], value)`",
            stacklevel=3,
        )


def logp(
    rv: TensorVariable, value: TensorLike, warn_missing_rvs: bool = True, **kwargs
) -> TensorVariable:
    """Return the log-probability graph of a Random Variable"""

    value = pt.as_tensor_variable(value, dtype=rv.dtype)
    try:
        return _logprob_helper(rv, value, **kwargs)
    except NotImplementedError:
        fgraph, _, _ = construct_ir_fgraph({rv: value})
        [(ir_rv, ir_value)] = fgraph.preserve_rv_mappings.rv_values.items()
        expr = _logprob_helper(ir_rv, ir_value, **kwargs)
        cleanup_ir([expr])
        if warn_missing_rvs:
            _warn_rvs_in_inferred_graph(expr)
        return expr


def logcdf(
    rv: TensorVariable, value: TensorLike, warn_missing_rvs: bool = True, **kwargs
) -> TensorVariable:
    """Create a graph for the log-CDF of a Random Variable."""
    value = pt.as_tensor_variable(value, dtype=rv.dtype)
    try:
        return _logcdf_helper(rv, value, **kwargs)
    except NotImplementedError:
        # Try to rewrite rv
        fgraph, rv_values, _ = construct_ir_fgraph({rv: value})
        [ir_rv] = fgraph.outputs
        expr = _logcdf_helper(ir_rv, value, **kwargs)
        cleanup_ir([expr])
        if warn_missing_rvs:
            _warn_rvs_in_inferred_graph(expr)
        return expr


def icdf(
    rv: TensorVariable, value: TensorLike, warn_missing_rvs: bool = True, **kwargs
) -> TensorVariable:
    """Create a graph for the inverse CDF of a  Random Variable."""
    value = pt.as_tensor_variable(value, dtype="floatX")
    try:
        return _icdf_helper(rv, value, **kwargs)
    except NotImplementedError:
        # Try to rewrite rv
        fgraph, rv_values, _ = construct_ir_fgraph({rv: value})
        [ir_rv] = fgraph.outputs
        expr = _icdf_helper(ir_rv, value, **kwargs)
        cleanup_ir([expr])
        if warn_missing_rvs:
            _warn_rvs_in_inferred_graph(expr)
        return expr


RVS_IN_JOINT_LOGP_GRAPH_MSG = (
    "Random variables detected in the logp graph: %s.\n"
    "This can happen when DensityDist logp or Interval transform functions reference nonlocal variables,\n"
    "or when not all rvs have a corresponding value variable."
)


def conditional_logp(
    rv_values: Dict[TensorVariable, TensorVariable],
    warn_missing_rvs: bool = True,
    ir_rewriter: Optional[GraphRewriter] = None,
    extra_rewrites: Optional[Union[GraphRewriter, NodeRewriter]] = None,
    **kwargs,
) -> Dict[TensorVariable, TensorVariable]:
    r"""Create a map between variables and conditional log-probabilities
    such that the sum is their joint log-probability.

    The `rv_values` dictionary specifies a joint probability graph defined by
    pairs of random variables and respective measure-space input parameters

    For example, consider the following

    .. code-block:: python

        import pytensor.tensor as pt

        sigma2_rv = pt.random.invgamma(0.5, 0.5)
        Y_rv = pt.random.normal(0, pt.sqrt(sigma2_rv))

    This graph for ``Y_rv`` is equivalent to the following hierarchical model:

    .. math::

        \Sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5) \\
        Y \sim& \operatorname{N}(0, \Sigma)

    If we create a value variable for ``Y_rv``, i.e. ``y_vv = pt.scalar("y")``,
    the graph of ``conditional_logp({Y_rv: y_vv})`` is equivalent to the
    conditional log-probability :math:`\log p(Y = y \mid \Sigma^2)`, with a stochastic
    ``sigma2_rv``. If we specify a value variable for ``sigma2_rv``, i.e.
    ``s_vv = pt.scalar("s2")``, then ``conditional_logp({Y_rv: y_vv, sigma2_rv: s_vv})``
    yields the conditional log-probabilities of the two variables.
    The sum of the two terms gives their joint log-probability.

    .. math::

        \log p(Y = y, \Sigma^2 = \sigma^2) =
            \log p(Y = y \mid \Sigma^2 = \sigma^2) + \log p(\Sigma^2 = \sigma^2)


    Parameters
    ----------
    rv_values
        A ``dict`` of variables that maps stochastic elements
        (e.g. `RandomVariable`\s) to symbolic `Variable`\s representing their
        values in a log-probability.
    warn_missing_rvs
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
    A ``dict`` that maps each value variable to the conditional log-probability term derived
    from the respective `RandomVariable`.

    """
    fgraph, rv_values, _ = construct_ir_fgraph(rv_values, ir_rewriter=ir_rewriter)

    if extra_rewrites is not None:
        extra_rewrites.rewrite(fgraph)

    rv_remapper = fgraph.preserve_rv_mappings

    # This is the updated random-to-value-vars map with the lifted/rewritten
    # variables.  The rewrites are supposed to produce new
    # `MeasurableVariable`s that are amenable to `_logprob`.
    updated_rv_values = rv_remapper.rv_values

    # Some rewrites also transform the original value variables. This is the
    # updated map from the new value variables to the original ones, which
    # we want to use as the keys in the final dictionary output
    original_values = rv_remapper.original_values

    # When a `_logprob` has been produced for a `MeasurableVariable` node, all
    # other references to it need to be replaced with its value-variable all
    # throughout the `_logprob`-produced graphs.  The following `dict`
    # cumulatively maintains remappings for all the variables/nodes that needed
    # to be recreated after replacing `MeasurableVariable`s with their
    # value-variables.  Since these replacements work in topological order, all
    # the necessary value-variable replacements should be present for each
    # node.
    replacements = updated_rv_values.copy()

    # To avoid cloning the value variables (or ancestors of value variables),
    # we map them to themselves in the `replacements` `dict`
    # (i.e. entries already existing in `replacements` aren't cloned)
    replacements.update(
        {
            v: v
            for v in ancestors(rv_values.values())
            if (not isinstance(v, Constant) and v not in replacements)
        }
    )

    # Walk the graph from its inputs to its outputs and construct the
    # log-probability
    q = deque(fgraph.toposort())
    logprob_vars = {}

    while q:
        node = q.popleft()

        if not isinstance(node.op, MeasurableVariable):
            continue

        q_values = [replacements[q_rv] for q_rv in node.outputs if q_rv in updated_rv_values]

        if not q_values:
            continue

        # Replace `RandomVariable`s in the inputs with value variables.
        # Also, store the results in the `replacements` map for the nodes
        # that follow.
        remapped_vars, _ = rvs_to_value_vars(
            q_values + list(node.inputs),
            initial_replacements=replacements,
        )
        q_values = remapped_vars[: len(q_values)]
        q_rv_inputs = remapped_vars[len(q_values) :]

        q_logprob_vars = _logprob(
            node.op,
            q_values,
            *q_rv_inputs,
            **kwargs,
        )

        if not isinstance(q_logprob_vars, (list, tuple)):
            q_logprob_vars = [q_logprob_vars]

        for q_value_var, q_logprob_var in zip(q_values, q_logprob_vars):
            q_value_var = original_values[q_value_var]

            if q_value_var.name:
                q_logprob_var.name = f"{q_value_var.name}_logprob"

            if q_value_var in logprob_vars:
                raise ValueError(
                    f"More than one logprob term was assigned to the value var {q_value_var}"
                )

            logprob_vars[q_value_var] = q_logprob_var

        # Recompute test values for the changes introduced by the
        # replacements above.
        if config.compute_test_value != "off":
            for node in io_toposort(graph_inputs(q_logprob_vars), q_logprob_vars):
                compute_test_value(node)

    missing_value_terms = set(original_values.values()) - set(logprob_vars.keys())
    if missing_value_terms:
        raise RuntimeError(
            f"The logprob terms of the following value variables could not be derived: {missing_value_terms}"
        )

    logprob_expressions = list(logprob_vars.values())
    cleanup_ir(logprob_expressions)

    if warn_missing_rvs:
        rvs_in_logp_expressions = _find_unallowed_rvs_in_graph(logprob_expressions)
        if rvs_in_logp_expressions:
            warnings.warn(RVS_IN_JOINT_LOGP_GRAPH_MSG % rvs_in_logp_expressions, UserWarning)

    return logprob_vars


def transformed_conditional_logp(
    rvs: Sequence[TensorVariable],
    *,
    rvs_to_values: Dict[TensorVariable, TensorVariable],
    rvs_to_transforms: Dict[TensorVariable, RVTransform],
    jacobian: bool = True,
    **kwargs,
) -> List[TensorVariable]:
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
        transform_rewrite = TransformValuesRewrite(values_to_transforms)  # type: ignore

    kwargs.setdefault("warn_missing_rvs", False)
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
