import warnings
from collections import deque
from typing import Dict, Optional, Union

import aesara.tensor as at
from aesara import config
from aesara.graph.basic import graph_inputs, io_toposort
from aesara.graph.op import compute_test_value
from aesara.graph.rewriting.basic import GraphRewriter, NodeRewriter
from aesara.tensor.var import TensorVariable

from aeppl.abstract import get_measurable_outputs
from aeppl.logprob import _logprob
from aeppl.rewriting import construct_ir_fgraph
from aeppl.utils import rvs_to_value_vars


def factorized_joint_logprob(
    rv_values: Dict[TensorVariable, TensorVariable],
    warn_missing_rvs: bool = True,
    ir_rewriter: Optional[GraphRewriter] = None,
    extra_rewrites: Optional[Union[GraphRewriter, NodeRewriter]] = None,
    **kwargs,
) -> Dict[TensorVariable, TensorVariable]:
    r"""Create a map between variables and their log-probabilities such that the
    sum is their joint log-probability.

    The `rv_values` dictionary specifies a joint probability graph defined by
    pairs of random variables and respective measure-space input parameters

    For example, consider the following

    .. code-block:: python

        import aesara.tensor as at

        sigma2_rv = at.random.invgamma(0.5, 0.5)
        Y_rv = at.random.normal(0, at.sqrt(sigma2_rv))

    This graph for ``Y_rv`` is equivalent to the following hierarchical model:

    .. math::

        \sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5) \\
        Y \sim& \operatorname{N}(0, \sigma^2)

    If we create a value variable for ``Y_rv``, i.e. ``y_vv = at.scalar("y")``,
    the graph of ``factorized_joint_logprob({Y_rv: y_vv})`` is equivalent to the
    conditional probability :math:`\log p(Y = y \mid \sigma^2)`, with a stochastic
    ``sigma2_rv``. If we specify a value variable for ``sigma2_rv``, i.e.
    ``s_vv = at.scalar("s2")``, then ``factorized_joint_logprob({Y_rv: y_vv, sigma2_rv: s_vv})``
    yields the joint log-probability of the two variables.

    .. math::

        \log p(Y = y, \sigma^2 = s) =
            \log p(Y = y \mid \sigma^2 = s) + \log p(\sigma^2 = s)


    Parameters
    ==========
    rv_values
        A ``dict`` of variables that maps stochastic elements
        (e.g. `RandomVariable`\s) to symbolic `Variable`\s representing their
        values in a log-probability.
    warn_missing_rvs
        When ``True``, issue a warning when a `RandomVariable` is found in
        the graph and doesn't have a corresponding value variable specified in
        `rv_values`.
    ir_rewriter
        Rewriter that produces the intermediate representation of Measurable Variables.
    extra_rewrites
        Extra rewrites to be applied (e.g. reparameterizations, transforms,
        etc.)

    Returns
    =======
    A ``dict`` that maps each value variable to the log-probability factor derived
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

    # To avoid cloning the value variables, we map them to themselves in the
    # `replacements` `dict` (i.e. entries already existing in `replacements`
    # aren't cloned)
    replacements.update({v: v for v in rv_values.values()})

    # Walk the graph from its inputs to its outputs and construct the
    # log-probability
    q = deque(fgraph.toposort())

    logprob_vars = {}

    while q:
        node = q.popleft()

        outputs = get_measurable_outputs(node.op, node)

        if not outputs:
            continue

        if any(o not in updated_rv_values for o in outputs):
            if warn_missing_rvs:
                warnings.warn(
                    "Found a random variable that was neither among the observations "
                    f"nor the conditioned variables: {node.outputs}"
                )
            continue

        q_value_vars = [replacements[q_rv_var] for q_rv_var in outputs]

        if not q_value_vars:
            continue

        # Replace `RandomVariable`s in the inputs with value variables.
        # Also, store the results in the `replacements` map for the nodes
        # that follow.
        remapped_vars, _ = rvs_to_value_vars(
            q_value_vars + list(node.inputs),
            initial_replacements=replacements,
        )
        q_value_vars = remapped_vars[: len(q_value_vars)]
        q_rv_inputs = remapped_vars[len(q_value_vars) :]

        q_logprob_vars = _logprob(
            node.op,
            q_value_vars,
            *q_rv_inputs,
            **kwargs,
        )

        if not isinstance(q_logprob_vars, (list, tuple)):
            q_logprob_vars = [q_logprob_vars]

        for q_value_var, q_logprob_var in zip(q_value_vars, q_logprob_vars):

            q_value_var = original_values[q_value_var]

            if q_value_var.name:
                q_logprob_var.name = f"{q_value_var.name}_logprob"

            if q_value_var in logprob_vars:
                raise ValueError(
                    f"More than one logprob factor was assigned to the value var {q_value_var}"
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

    return logprob_vars


def joint_logprob(*args, sum: bool = True, **kwargs) -> Optional[TensorVariable]:
    """Create a graph representing the joint log-probability/measure of a graph.

    This function calls `factorized_joint_logprob` and returns the combined
    log-probability factors as a single graph.

    Parameters
    ----------
    sum: bool
        If ``True`` each factor is collapsed to a scalar via ``sum`` before
        being joined with the remaining factors. This may be necessary to
        avoid incorrect broadcasting among independent factors.

    """
    logprob = factorized_joint_logprob(*args, **kwargs)
    if not logprob:
        return None
    elif len(logprob) == 1:
        logprob = tuple(logprob.values())[0]
        if sum:
            return at.sum(logprob)
        else:
            return logprob
    else:
        if sum:
            return at.sum([at.sum(factor) for factor in logprob.values()])
        else:
            return at.add(*logprob.values())
