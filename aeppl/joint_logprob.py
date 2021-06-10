import warnings
from collections import deque
from typing import Dict, Optional

from aesara import config
from aesara.graph.basic import graph_inputs, io_toposort
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import compute_test_value
from aesara.graph.opt_utils import optimize_graph
from aesara.tensor.basic_opt import ShapeFeature
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.var import TensorVariable

from aeppl.logprob import _logprob
from aeppl.opt import PreserveRVMappings, RVSinker
from aeppl.utils import rvs_to_value_vars


def joint_logprob(
    var: TensorVariable,
    rv_values: Optional[Dict[TensorVariable, TensorVariable]] = None,
    warn_missing_rvs=True,
    **kwargs,
) -> TensorVariable:
    r"""Create a graph representing the joint log-probability/measure of a graph.

    The input `var` determines which graph is used and `rv_values` specifies
    the resulting measure-space graph's input parameters.

    For example, consider the following

    .. code-block:: python

        import aesara.tensor as at

        Y_rv = at.random.normal(0, at.sqrt(sigma2_rv))
        sigma2_rv = at.random.invgamma(0.5, 0.5)

    This graph for ``Y_rv`` is equivalent to the following hierarchical model:

    .. math::

        Y \sim& \operatorname{N}(0, \sigma^2) \\
        \sigma^2 \sim& \operatorname{InvGamma}(0.5, 0.5)

    If we create a value variable for ``Y_rv``, i.e. ``y = at.scalar("y")``,
    the graph of ``joint_logprob(Y_rv, {Y_rv: y})`` is equivalent to the
    conditional probability :math:`\log p(Y = y \mid \sigma^2)`.  If we specify
    a value variable for ``sigma2_rv``, i.e. ``s = at.scalar("s2")``, then
    ``joint_logprob(Y_rv, {Y_rv: y, sigma2_rv: s})`` yields the joint
    log-probability

    .. math::

        \log p(Y = y, \sigma^2 = s) =
            \log p(Y = y \mid \sigma^2 = s) + \log p(\sigma^2 = s)


    Parameters
    ==========
    var
        The graph containing the stochastic/`RandomVariable` elements for
        which we want to compute a joint log-probability.  This graph
        effectively represents a statistical model.
    rv_values
        A ``dict`` of variables that maps stochastic elements
        (e.g. `RandomVariable`\s) to symbolic `Variable`\s representing their
        values in a log-probability.
    warn_missing_rvs
        When ``True``, issue a warning when a `RandomVariable` is found in
        the graph and doesn't have a corresponding value variable specified in
        `rv_values`.

    """
    # Since we're going to clone the entire graph, we need to keep a map from
    # the old nodes to the new ones; otherwise, we won't be able to use
    # `rv_values`.
    # We start the `dict` with mappings from the value variables to themselves,
    # to prevent them from being cloned.
    memo = {v: v for v in rv_values.values()}

    # We add `ShapeFeature` because it will get rid of references to the old
    # `RandomVariable`s that have been lifted; otherwise, it will be difficult
    # to give good warnings when an unaccounted for `RandomVairiable` is
    # encountered
    fgraph = FunctionGraph(
        outputs=[var],
        clone=True,
        memo=memo,
        copy_orphans=False,
        features=[ShapeFeature()],
    )

    # Update `rv_values` so that it uses the new cloned variables
    rv_values = {memo[k]: v for k, v in rv_values.items()}

    # This `Feature` preserves the relationships between the original
    # random variables (i.e. keys in `rv_values`) and the new ones
    # produced when `Op`s are lifted through them.
    rv_remapper = PreserveRVMappings(rv_values)

    fgraph.attach_feature(rv_remapper)

    _ = optimize_graph(fgraph, custom_opt=RVSinker())

    # This is the updated random-to-value-vars map with the
    # lifted variables
    lifted_rv_values = rv_remapper.rv_values
    replacements = lifted_rv_values.copy()

    # Walk the graph from its inputs to its outputs and construct the
    # log-probability
    q = deque(fgraph.toposort())

    logprob_var = None

    while q:
        node = q.popleft()

        if not any(o in lifted_rv_values for o in node.outputs):
            if isinstance(node.op, RandomVariable) and warn_missing_rvs:
                warnings.warn(
                    "Found a random variable that was neither among the observations "
                    f"nor the conditioned variables: {node}"
                )
            continue

        if isinstance(node.op, RandomVariable):
            q_rv_var = node.outputs[1]
            q_rv_value_var = replacements[q_rv_var]

            # Replace `RandomVariable`s in the inputs with value variables.
            # Also, store the results in the `replacements` map so that we
            # don't need to redo these replacements.
            value_var_inputs, _ = rvs_to_value_vars(
                node.inputs,
                initial_replacements=replacements,
            )

            q_logprob_var = _logprob(
                node.op, q_rv_value_var, *value_var_inputs, **kwargs
            )

        else:
            raise NotImplementedError(
                f"A measure/probability could not be derived for {node}"
            )

        if logprob_var is None:
            logprob_var = q_logprob_var
        else:
            logprob_var += q_logprob_var

    # Recompute test values for the changes introduced by the replacements
    # above.
    if config.compute_test_value != "off":
        for node in io_toposort(graph_inputs((logprob_var,)), (logprob_var,)):
            compute_test_value(node)

    return logprob_var
