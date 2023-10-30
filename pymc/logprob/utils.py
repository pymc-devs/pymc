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

from typing import Container, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pytensor

from pytensor import Variable
from pytensor import tensor as pt
from pytensor.graph import Apply, Op, node_rewriter
from pytensor.graph.basic import walk
from pytensor.graph.op import HasInnerGraph
from pytensor.link.c.type import CType
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable

from pymc.distributions.transforms import Transform
from pymc.logprob.abstract import MeasurableVariable, _logprob
from pymc.pytensorf import replace_vars_in_graphs
from pymc.util import makeiter


def replace_rvs_by_values(
    graphs: Sequence[TensorVariable],
    *,
    rvs_to_values: Dict[TensorVariable, TensorVariable],
    rvs_to_transforms: Optional[Dict[TensorVariable, "Transform"]] = None,
) -> List[TensorVariable]:
    """Clone and replace random variables in graphs with their value variables.

    Parameters
    ----------
    graphs
        The graphs in which to perform the replacements.
    rvs_to_values
        Mapping between the original graph RVs and respective value variables
    rvs_to_transforms, optional
        Mapping between the original graph RVs and respective value transforms
    """

    replacements = {}

    def populate_replacements(var):
        # Populate replacements dict with {rv: value} pairs indicating which graph
        # RVs should be replaced by what value variables.
        if not var.owner:
            return []

        next_vars = []
        value = rvs_to_values.get(var, None)
        if value is not None:
            rv = var

            if rvs_to_transforms is not None:
                transform = rvs_to_transforms.get(rv, None)
                if transform is not None:
                    # We want to replace uses of the RV by the back-transformation of its value
                    value = transform.backward(value, *rv.owner.inputs)
                    # The value may have a less precise type than the rv. In this case
                    # filter_variable will add a SpecifyShape to ensure they are consistent
                    value = rv.type.filter_variable(value, allow_convert=True)
                    value.name = rv.name

            replacements[rv] = value
            # Also walk the graph of the value variable to make any additional
            # replacements if that is not a simple input variable
            next_vars.append(value)

        next_vars.extend(reversed(var.owner.inputs))
        return next_vars

    # Iterate over the generator to populate the replacements
    for _ in walk(graphs, populate_replacements, bfs=False):
        pass

    return replace_vars_in_graphs(graphs, replacements)


def rvs_in_graph(vars: Union[Variable, Sequence[Variable]]) -> Set[Variable]:
    """Assert that there are no `MeasurableVariable` nodes in a graph."""

    def expand(r):
        owner = r.owner
        if owner:
            inputs = list(reversed(owner.inputs))

            if isinstance(owner.op, HasInnerGraph):
                inputs += owner.op.inner_outputs

            return inputs

    return {
        node
        for node in walk(makeiter(vars), expand, False)
        if node.owner and isinstance(node.owner.op, (RandomVariable, MeasurableVariable))
    }


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


def check_potential_measurability(
    inputs: Tuple[TensorVariable], valued_rvs: Container[TensorVariable]
) -> bool:
    valued_rvs = set(valued_rvs)

    def expand_fn(var):
        # expand_fn does not go beyond valued_rvs or any MeasurableVariable
        if var.owner and not isinstance(var.owner.op, MeasurableVariable) and var not in valued_rvs:
            return reversed(var.owner.inputs)
        else:
            return []

    if any(
        ancestor_var
        for ancestor_var in walk(inputs, expand=expand_fn, bfs=False)
        if (
            ancestor_var.owner
            and isinstance(ancestor_var.owner.op, MeasurableVariable)
            and ancestor_var not in valued_rvs
        )
    ):
        return True
    return False


class ParameterValueError(ValueError):
    """Exception for invalid parameters values in logprob graphs"""


class CheckParameterValue(CheckAndRaise):
    """Implements a parameter value check in a logprob graph.

    Raises `ParameterValueError` if the check is not True.
    """

    __props__ = ("msg", "exc_type", "can_be_replaced_by_ninf")

    def __init__(self, msg: str = "", can_be_replaced_by_ninf: bool = False):
        super().__init__(ParameterValueError, msg)
        self.can_be_replaced_by_ninf = can_be_replaced_by_ninf

    def __str__(self):
        return f"Check{{{self.msg}}}"


@node_rewriter(tracks=[CheckParameterValue])
def local_remove_check_parameter(fgraph, node):
    """Rewrite that removes CheckParameterValue

    This is used when compile_rv_inplace
    """
    if isinstance(node.op, CheckParameterValue):
        return [node.inputs[0]]


@node_rewriter(tracks=[CheckParameterValue])
def local_check_parameter_to_ninf_switch(fgraph, node):
    if not node.op.can_be_replaced_by_ninf:
        return None

    logp_expr, *logp_conds = node.inputs
    if len(logp_conds) > 1:
        logp_cond = pt.all(logp_conds)
    else:
        (logp_cond,) = logp_conds
    out = pt.switch(logp_cond, logp_expr, -np.inf)
    out.name = node.op.msg

    if out.dtype != node.outputs[0].dtype:
        out = pt.cast(out, node.outputs[0].dtype)

    return [out]


pytensor.compile.optdb["canonicalize"].register(
    "local_remove_check_parameter",
    local_remove_check_parameter,
    use_db_name_as_tag=False,
)

pytensor.compile.optdb["canonicalize"].register(
    "local_check_parameter_to_ninf_switch",
    local_check_parameter_to_ninf_switch,
    use_db_name_as_tag=False,
)


class DiracDelta(Op):
    """An `Op` that represents a Dirac-delta distribution."""

    __props__ = ("rtol", "atol")

    def __init__(self, rtol=1e-5, atol=1e-8):
        self.rtol = rtol
        self.atol = atol

    def make_node(self, x):
        x = pt.as_tensor(x)
        return Apply(self, [x], [x.type()])

    def do_constant_folding(self, fgraph, node):
        # Without this, the `Op` would be removed from the graph during
        # canonicalization
        return False

    def perform(self, node, inp, out):
        (x,) = inp
        (z,) = out
        warnings.warn("DiracDelta is a dummy Op that shouldn't be used in a compiled graph")
        z[0] = x

    def infer_shape(self, fgraph, node, input_shapes):
        return input_shapes


MeasurableVariable.register(DiracDelta)


dirac_delta = DiracDelta()


@_logprob.register(DiracDelta)
def diracdelta_logprob(op, values, *inputs, **kwargs):
    (values,) = values
    (const_value,) = inputs
    values, const_value = pt.broadcast_arrays(values, const_value)
    return pt.switch(pt.isclose(values, const_value, rtol=op.rtol, atol=op.atol), 0.0, -np.inf)
