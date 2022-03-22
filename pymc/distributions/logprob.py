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
from typing import Dict, List, Optional, Sequence, Union

import aesara
import aesara.tensor as at
import numpy as np

from aeppl import factorized_joint_logprob
from aeppl.logprob import logcdf as logcdf_aeppl
from aeppl.logprob import logprob as logp_aeppl
from aeppl.transforms import TransformValuesOpt
from aesara.graph.basic import graph_inputs, io_toposort
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.subtensor import (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)
from aesara.tensor.var import TensorVariable

from pymc.aesaraf import floatX


def _get_scaling(total_size: Optional[Union[int, Sequence[int]]], shape, ndim: int):
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
        coef = at.prod(coefs)
    else:
        raise TypeError(
            "Unrecognized `total_size` type, expected int or list of ints, got %r" % total_size
        )
    return at.as_tensor(coef, dtype=aesara.config.floatX)


subtensor_types = (
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    AdvancedSubtensor,
    AdvancedSubtensor1,
    IncSubtensor,
    Subtensor,
)


def joint_logpt(
    var: Union[TensorVariable, List[TensorVariable]],
    rv_values: Optional[Union[TensorVariable, Dict[TensorVariable, TensorVariable]]] = None,
    *,
    jacobian: bool = True,
    scaling: bool = True,
    transformed: bool = True,
    sum: bool = True,
    **kwargs,
) -> Union[TensorVariable, List[TensorVariable]]:
    """Create a measure-space (i.e. log-likelihood) graph for a random variable
    or a list of random variables at a given point.

    The input `var` determines which log-likelihood graph is used and
    `rv_value` is that graph's input parameter.  For example, if `var` is
    the output of a ``NormalRV`` ``Op``, then the output is a graph of the
    density function for `var` set to the value `rv_value`.

    Parameters
    ==========
    var
        The `RandomVariable` output that determines the log-likelihood graph.
        Can also be a list of variables. The final log-likelihood graph will
        be the sum total of all individual log-likelihood graphs of variables
        in the list.
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
    sum
        Sum the log-likelihood or return each term as a separate list item.

    """
    # TODO: In future when we drop support for tag.value_var most of the following
    # logic can be removed and logpt can just be a wrapper function that calls aeppl's
    # joint_logprob directly.

    # If var is not a list make it one.
    if not isinstance(var, (list, tuple)):
        var = [var]

    # If logpt isn't provided values it is assumed that the tagged value var or
    # observation is the value variable for that particular RV.
    if rv_values is None:
        rv_values = {}
        for rv in var:
            value_var = getattr(rv.tag, "observations", getattr(rv.tag, "value_var", None))
            if value_var is None:
                raise ValueError(f"No value variable found for var {rv}")
            rv_values[rv] = value_var
    # Else we assume we were given a single rv and respective value
    elif not isinstance(rv_values, Mapping):
        if len(var) == 1:
            rv_values = {var[0]: at.as_tensor_variable(rv_values).astype(var[0].type)}
        else:
            raise ValueError("rv_values must be a dict if more than one var is requested")

    if scaling:
        rv_scalings = {}
        for rv, value_var in rv_values.items():
            rv_scalings[value_var] = _get_scaling(
                getattr(rv.tag, "total_size", None), value_var.shape, value_var.ndim
            )

    # Aeppl needs all rv-values pairs, not just that of the requested var.
    # Hence we iterate through the graph to collect them.
    tmp_rvs_to_values = rv_values.copy()
    for node in io_toposort(graph_inputs(var), var):
        try:
            curr_vars = [node.default_output()]
        except ValueError:
            curr_vars = node.outputs
        for curr_var in curr_vars:
            if curr_var in tmp_rvs_to_values:
                continue
            # Check if variable has a value variable
            value_var = getattr(
                curr_var.tag, "observations", getattr(curr_var.tag, "value_var", None)
            )
            if value_var is not None:
                tmp_rvs_to_values[curr_var] = value_var

    # After collecting all necessary rvs and values, we check for any value transforms
    transform_map = {}
    if transformed:
        for rv, value_var in tmp_rvs_to_values.items():
            if hasattr(value_var.tag, "transform"):
                transform_map[value_var] = value_var.tag.transform
            # If the provided value_variable does not have transform information, we
            # check if the original `rv.tag.value_var` does.
            # TODO: This logic should be replaced by an explicit dict of
            #  `{value_var: transform}` similar to `rv_values`.
            else:
                original_value_var = getattr(rv.tag, "value_var", None)
                if original_value_var is not None and hasattr(original_value_var.tag, "transform"):
                    transform_map[value_var] = original_value_var.tag.transform

    transform_opt = TransformValuesOpt(transform_map)
    temp_logp_var_dict = factorized_joint_logprob(
        tmp_rvs_to_values, extra_rewrites=transform_opt, use_jacobian=jacobian, **kwargs
    )

    # Raise if there are unexpected RandomVariables in the logp graph
    # Only SimulatorRVs are allowed
    from pymc.distributions.simulator import SimulatorRV

    unexpected_rv_nodes = [
        node
        for node in aesara.graph.ancestors(list(temp_logp_var_dict.values()))
        if (
            node.owner
            and isinstance(node.owner.op, RandomVariable)
            and not isinstance(node.owner.op, SimulatorRV)
        )
    ]
    if unexpected_rv_nodes:
        raise ValueError(
            f"Random variables detected in the logp graph: {unexpected_rv_nodes}.\n"
            "This can happen when DensityDist logp or Interval transform functions "
            "reference nonlocal variables."
        )

    # aeppl returns the logpt for every single value term we provided to it. This includes
    # the extra values we plugged in above, so we filter those we actually wanted in the
    # same order they were given in.
    logp_var_dict = {}
    for value_var in rv_values.values():
        logp_var_dict[value_var] = temp_logp_var_dict[value_var]

    if scaling:
        for value_var in logp_var_dict.keys():
            if value_var in rv_scalings:
                logp_var_dict[value_var] *= rv_scalings[value_var]

    if sum:
        logp_var = at.sum([at.sum(factor) for factor in logp_var_dict.values()])
    else:
        logp_var = list(logp_var_dict.values())

    return logp_var


def logp(rv, value):
    """Return the log-probability graph of a Random Variable"""

    value = at.as_tensor_variable(value, dtype=rv.dtype)
    return logp_aeppl(rv, value)


def logcdf(rv, value):
    """Return the log-cdf graph of a Random Variable"""

    value = at.as_tensor_variable(value, dtype=rv.dtype)
    return logcdf_aeppl(rv, value)
