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

from typing import (
    Callable,
    Container,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

from pytensor import Variable
from pytensor import tensor as pt
from pytensor.graph import Apply, Op
from pytensor.graph.basic import Constant, clone_get_equiv, graph_inputs, walk
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.op import HasInnerGraph
from pytensor.link.c.type import CType
from pytensor.raise_op import CheckAndRaise
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import MeasurableVariable, _logprob
from pymc.util import makeiter


def walk_model(
    graphs: Iterable[TensorVariable],
    walk_past_rvs: bool = False,
    stop_at_vars: Optional[Set[TensorVariable]] = None,
    expand_fn: Callable[[TensorVariable], List[TensorVariable]] = lambda var: [],
) -> Generator[TensorVariable, None, None]:
    """Walk model graphs and yield their nodes.

    By default, these walks will not go past ``MeasurableVariable`` nodes.

    Parameters
    ----------
    graphs
        The graphs to walk.
    walk_past_rvs
        If ``True``, the walk will not terminate at ``MeasurableVariable``s.
    stop_at_vars
        A list of variables at which the walk will terminate.
    expand_fn
        A function that returns the next variable(s) to be traversed.
    """
    if stop_at_vars is None:
        stop_at_vars = set()

    def expand(var: TensorVariable, stop_at_vars=stop_at_vars) -> List[TensorVariable]:
        new_vars = expand_fn(var)

        if (
            var.owner
            and (walk_past_rvs or not isinstance(var.owner.op, MeasurableVariable))
            and (var not in stop_at_vars)
        ):
            new_vars.extend(reversed(var.owner.inputs))

        return new_vars

    yield from walk(graphs, expand, False)


def replace_rvs_in_graphs(
    graphs: Iterable[TensorVariable],
    replacement_fn: Callable[
        [TensorVariable, Dict[TensorVariable, TensorVariable]],
        Dict[TensorVariable, TensorVariable],
    ],
    initial_replacements: Optional[Dict[TensorVariable, TensorVariable]] = None,
    **kwargs,
) -> Tuple[TensorVariable, Dict[TensorVariable, TensorVariable]]:
    """Replace random variables in graphs.

    This will *not* recompute test values.

    Parameters
    ----------
    graphs
        The graphs in which random variables are to be replaced.

    Returns
    -------
    A ``tuple`` containing the transformed graphs and a ``dict`` of the
    replacements that were made.
    """
    replacements = {}
    if initial_replacements:
        replacements.update(initial_replacements)

    def expand_replace(var: TensorVariable) -> List[TensorVariable]:
        new_nodes: List[TensorVariable] = []
        if var.owner and isinstance(var.owner.op, MeasurableVariable):
            new_nodes.extend(replacement_fn(var, replacements))
        return new_nodes

    for var in walk_model(graphs, expand_fn=expand_replace, **kwargs):
        pass

    if replacements:
        inputs = [i for i in graph_inputs(graphs) if not isinstance(i, Constant)]
        equiv = {k: k for k in replacements.keys()}
        equiv = clone_get_equiv(inputs, graphs, False, False, equiv)

        fg = FunctionGraph(
            [equiv[i] for i in inputs],
            [equiv[o] for o in graphs],
            clone=False,
        )

        fg.replace_all(replacements.items(), import_missing=True)

        graphs = list(fg.outputs)

    return graphs, replacements


def rvs_to_value_vars(
    graphs: Iterable[TensorVariable],
    initial_replacements: Optional[Dict[TensorVariable, TensorVariable]] = None,
    **kwargs,
) -> Tuple[TensorVariable, Dict[TensorVariable, TensorVariable]]:
    """Replace random variables in graphs with their value variables.

    This will *not* recompute test values in the resulting graphs.

    Parameters
    ----------
    graphs
        The graphs in which to perform the replacements.
    initial_replacements
        A ``dict`` containing the initial replacements to be made.

    """

    def replace_fn(var, replacements):
        rv_value_var = replacements.get(var, None)
        if rv_value_var is not None:
            replacements[var] = rv_value_var
            # In case the value variable is itself a graph, we walk it for
            # potential replacements
            return [rv_value_var]
        return []

    return replace_rvs_in_graphs(graphs, replace_fn, initial_replacements, **kwargs)


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
    if any(
        ancestor_var
        for ancestor_var in walk_model(
            inputs,
            walk_past_rvs=False,
            stop_at_vars=set(valued_rvs),
        )
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


def find_rvs_in_graph(vars: Union[Variable, Sequence[Variable]]) -> Set[Variable]:
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
