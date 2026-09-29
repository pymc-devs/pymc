#   Copyright 2024 - present The PyMC Developers
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

import pytensor.tensor as pt

from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import GraphRewriter
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import valued_rv
from pymc.logprob.query import query_parts
from pymc.logprob.transforms import Transform


def add_value_jacobian(logp, value, transform, rv, use_jacobian=True):
    log_jac_det = transform.log_jac_det(value, *rv.owner.inputs).copy()
    if log_jac_det.ndim < logp.ndim:
        # An event transform combines formerly independent density entries.
        logp = logp.sum(axis=tuple(range(log_jac_det.ndim, logp.ndim)))
    elif log_jac_det.ndim > logp.ndim:
        raise NotImplementedError(
            f"Univariate transform {transform} cannot be applied to multivariate {rv.owner.op}"
        )
    if logp.type.broadcastable != log_jac_det.type.broadcastable:
        broadcastable_axes = [
            i
            for i, (left, right) in enumerate(
                zip(logp.type.broadcastable, log_jac_det.type.broadcastable, strict=True)
            )
            if left or right
        ]
        try:
            logp = pt.specify_broadcastable(logp, *broadcastable_axes)
            log_jac_det = pt.specify_broadcastable(log_jac_det, *broadcastable_axes)
        except ValueError as err:
            raise ValueError(
                f"The logp of {rv.owner.op} and log_jac_det of {transform} are not allowed to "
                "broadcast together. There is a bug in the implementation of either one."
            ) from err
    if not use_jacobian:
        return logp
    if value.name:
        log_jac_det.name = f"{value.name}_jacobian"
    return logp + log_jac_det


class TransformValuesRewrite(GraphRewriter):
    """Express sampling transforms as explicit query values and tensor Jacobians.

    Applied once to the initial density queries, before inference. Transform
    parameters remain in the same graph as those queries, so inferred values
    condition both inverse transforms and Jacobians.
    """

    def __init__(
        self,
        values_to_transforms: Mapping[TensorVariable, Transform | None],
        use_jacobian: bool = True,
    ):
        self.values_to_transforms = values_to_transforms
        self.use_jacobian = use_jacobian

    def apply(self, fgraph: FunctionGraph):
        original_outputs = list(fgraph.outputs)
        for term in original_outputs:
            query = term.owner
            rv, value = query_parts(query)
            transform = self.values_to_transforms.get(value)
            if transform is not None:
                natural_value = transform.backward(value, *rv.owner.inputs)
                fgraph.replace(
                    query.inputs[0],
                    valued_rv(rv, natural_value),
                    reason="transform query value",
                    import_missing=True,
                )
                term = add_value_jacobian(term, value, transform, rv, self.use_jacobian)
            # Import corrections now, so subsequent bindings also update their parameters.
            fgraph.add_output(term, reason="transform density", import_missing=True)
        for _ in original_outputs:
            fgraph.remove_output(0, reason="transform density")
