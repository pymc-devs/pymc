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
import pytensor
import pytensor.tensor as pt

from pytensor import config
from pytensor.graph.basic import graph_inputs, io_toposort
from pytensor.graph.op import compute_test_value
from pytensor.graph.rewriting.basic import GraphRewriter, NodeRewriter
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.var import TensorVariable

from pymc.logprob.abstract import _logprob, get_measurable_outputs
from pymc.logprob.abstract import logprob as logp_logprob
from pymc.logprob.rewriting import construct_ir_fgraph
from pymc.logprob.transforms import RVTransform, TransformValuesRewrite
from pymc.logprob.utils import rvs_to_value_vars
from pymc.pytensorf import floatX


def logp(rv: TensorVariable, value) -> TensorVariable:
    """Return the log-probability graph of a Random Variable"""

    value = pt.as_tensor_variable(value, dtype=rv.dtype)
    try:
        return logp_logprob(rv, value)
    except NotImplementedError:
        try:
            value = rv.type.filter_variable(value)
        except TypeError as exc:
            raise TypeError(
                "When RV is not a pure distribution, value variable must have the same type"
            ) from exc
        try:
            return factorized_joint_logprob({rv: value}, warn_missing_rvs=False)[value]
        except Exception as exc:
            raise NotImplementedError("PyMC could not infer logp of input variable.") from exc


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

        import pytensor.tensor as at

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
    ----------
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
    -------
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


TOTAL_SIZE = Union[int, Sequence[int], None]


def _get_scaling(total_size: TOTAL_SIZE, shape, ndim: int) -> TensorVariable:
    """
    Gets scaling constant for logp.

    Parameters
    ----------
    total_size: Optional[int|List[int]]
        size of a fully observed data without minibatching,
        `None` means data is fully observed
    shape: shape
        shape of an observed data
    ndim: int
        ndim hint

    Returns
    -------
    scalar
    """
    if total_size is None:
        coef = 1.0
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
            coef = 1.0
        if len(end) > 0:
            shp_end = shape[-len(end) :]
        else:
            shp_end = np.asarray([])
        shp_begin = shape[: len(begin)]
        begin_coef = [
            floatX(t) / floatX(shp_begin[i]) for i, t in enumerate(begin) if t is not None
        ]
        end_coef = [floatX(t) / floatX(shp_end[i]) for i, t in enumerate(end) if t is not None]
        coefs = begin_coef + end_coef
        coef = pt.prod(coefs)
    else:
        raise TypeError(
            "Unrecognized `total_size` type, expected int or list of ints, got %r" % total_size
        )
    return pt.as_tensor(coef, dtype=pytensor.config.floatX)


def _check_no_rvs(logp_terms: Sequence[TensorVariable]):
    # Raise if there are unexpected RandomVariables in the logp graph
    # Only SimulatorRVs MinibatchIndexRVs are allowed
    from pymc.data import MinibatchIndexRV
    from pymc.distributions.simulator import SimulatorRV

    unexpected_rv_nodes = [
        node
        for node in pytensor.graph.ancestors(logp_terms)
        if (
            node.owner
            and isinstance(node.owner.op, RandomVariable)
            and not isinstance(node.owner.op, (SimulatorRV, MinibatchIndexRV))
        )
    ]
    if unexpected_rv_nodes:
        raise ValueError(
            f"Random variables detected in the logp graph: {unexpected_rv_nodes}.\n"
            "This can happen when DensityDist logp or Interval transform functions "
            "reference nonlocal variables."
        )


def joint_logp(
    rvs: Sequence[TensorVariable],
    *,
    rvs_to_values: Dict[TensorVariable, TensorVariable],
    rvs_to_transforms: Dict[TensorVariable, RVTransform],
    jacobian: bool = True,
    rvs_to_total_sizes: Dict[TensorVariable, TOTAL_SIZE],
    **kwargs,
) -> List[TensorVariable]:
    """Thin wrapper around pymc.logprob.factorized_joint_logprob, extended with Model
    specific concerns such as transforms, jacobian, and scaling"""

    transform_rewrite = None
    values_to_transforms = {
        rvs_to_values[rv]: transform
        for rv, transform in rvs_to_transforms.items()
        if transform is not None
    }
    if values_to_transforms:
        # There seems to be an incorrect type hint in TransformValuesRewrite
        transform_rewrite = TransformValuesRewrite(values_to_transforms)  # type: ignore

    temp_logp_terms = factorized_joint_logprob(
        rvs_to_values,
        extra_rewrites=transform_rewrite,
        use_jacobian=jacobian,
        **kwargs,
    )

    # The function returns the logp for every single value term we provided to it. This
    # includes the extra values we plugged in above, so we filter those we actually
    # wanted in the same order they were given in.
    logp_terms = {}
    for rv in rvs:
        value_var = rvs_to_values[rv]
        logp_term = temp_logp_terms[value_var]
        total_size = rvs_to_total_sizes.get(rv, None)
        if total_size is not None:
            scaling = _get_scaling(total_size, value_var.shape, value_var.ndim)
            logp_term *= scaling
        logp_terms[value_var] = logp_term

    _check_no_rvs(list(logp_terms.values()))
    return list(logp_terms.values())
