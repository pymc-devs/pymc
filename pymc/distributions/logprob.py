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


from typing import Dict, List, Sequence, Union

import numpy as np
import pytensor

from pytensor import tensor as at
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.var import TensorVariable

from pymc.logprob.abstract import assign_custom_measurable_outputs
from pymc.logprob.abstract import logcdf as logcdf_logprob
from pymc.logprob.abstract import logprob as logp_logprob
from pymc.logprob.joint_logprob import factorized_joint_logprob
from pymc.logprob.transforms import RVTransform, TransformValuesRewrite
from pymc.pytensorf import floatX

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
        coef = at.prod(coefs)
    else:
        raise TypeError(
            "Unrecognized `total_size` type, expected int or list of ints, got %r" % total_size
        )
    return at.as_tensor(coef, dtype=pytensor.config.floatX)


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


def _joint_logp(
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


def logp(rv: TensorVariable, value) -> TensorVariable:
    """Return the log-probability graph of a Random Variable"""

    value = at.as_tensor_variable(value, dtype=rv.dtype)
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


def logcdf(rv: TensorVariable, value) -> TensorVariable:
    """Return the log-cdf graph of a Random Variable"""

    value = at.as_tensor_variable(value, dtype=rv.dtype)
    return logcdf_logprob(rv, value)


def ignore_logprob(rv: TensorVariable) -> TensorVariable:
    """Return a duplicated variable that is ignored when creating logprob graphs

    This is used in SymbolicDistributions that use other RVs as inputs but account
    for their logp terms explicitly.

    If the variable is already ignored, it is returned directly.
    """
    prefix = "Unmeasurable"
    node = rv.owner
    op_type = type(node.op)
    if op_type.__name__.startswith(prefix):
        return rv
    new_node = assign_custom_measurable_outputs(node, type_prefix=prefix)
    return new_node.outputs[node.outputs.index(rv)]
