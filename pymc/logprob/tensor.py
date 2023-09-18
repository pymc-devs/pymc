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

from typing import List, Optional, Union

from pytensor import tensor as pt
from pytensor.graph import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.basic import Join, MakeVector
from pytensor.tensor.elemwise import DimShuffle
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.random.rewriting import local_dimshuffle_rv_lift

from pymc.logprob.abstract import (
    MeasurableVariable,
    _logprob,
    _logprob_helper,
    promised_valued_rv,
)
from pymc.logprob.rewriting import (
    assume_valued_outputs,
    early_measurable_ir_rewrites_db,
    measurable_ir_rewrites_db,
    remove_valued_rvs,
)
from pymc.logprob.utils import (
    check_potential_measurability,
    filter_measurable_variables,
    replace_rvs_by_values,
)
from pymc.pytensorf import constant_fold, toposort_replace


class MeasurableMakeVector(MakeVector):
    """A placeholder used to specify a log-likelihood for a cumsum sub-graph."""


MeasurableVariable.register(MeasurableMakeVector)


@_logprob.register(MeasurableMakeVector)
def logprob_make_vector(op, values, *base_rvs, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableMakeVector`."""
    # TODO: Sort out this circular dependency issue

    (value,) = values

    temp_fgraph = FunctionGraph(outputs=base_rvs, clone=False)
    remove_valued_rvs.apply(temp_fgraph)
    base_rvs = temp_fgraph.outputs

    base_rvs_to_values = {base_rv: value[i] for i, base_rv in enumerate(base_rvs)}
    for i, (base_rv, value) in enumerate(base_rvs_to_values.items()):
        base_rv.name = f"base_rv[{i}]"
        value.name = f"value[{i}]"

    logps = [_logprob_helper(base_rv, value) for base_rv, value in base_rvs_to_values.items()]

    # If the stacked variables depend on each other, we have to replace them by the respective values
    logps = replace_rvs_by_values(logps, rvs_to_values=base_rvs_to_values)

    return pt.stack(logps)


class MeasurableJoin(Join):
    """A placeholder used to specify a log-likelihood for a join sub-graph."""


MeasurableVariable.register(MeasurableJoin)


@_logprob.register(MeasurableJoin)
def logprob_join(op, values, axis, *base_rvs, **kwargs):
    """Compute the log-likelihood graph for a `Join`."""
    (value,) = values

    temp_fgraph = FunctionGraph(outputs=base_rvs, clone=False)
    remove_valued_rvs.apply(temp_fgraph)
    base_rvs = temp_fgraph.outputs

    base_rv_shapes = [base_var.shape[axis] for base_var in base_rvs]

    # We don't need the graph to be constant, just to have RandomVariables removed
    base_rv_shapes = constant_fold(base_rv_shapes, raise_not_constant=False)

    split_values = pt.split(
        value,
        splits_size=base_rv_shapes,
        n_splits=len(base_rvs),
        axis=axis,
    )

    base_rvs_to_split_values = {base_rv: value for base_rv, value in zip(base_rvs, split_values)}
    logps = [
        _logprob_helper(base_var, split_value)
        for base_var, split_value in base_rvs_to_split_values.items()
    ]

    if len({logp.ndim for logp in logps}) != 1:
        raise ValueError(
            "Joined logps have different number of dimensions, this can happen when "
            "joining univariate and multivariate distributions",
        )

    # If the stacked variables depend on each other, we have to replace them by the respective values
    logps = replace_rvs_by_values(logps, rvs_to_values=base_rvs_to_split_values)

    base_vars_ndim_supp = split_values[0].ndim - logps[0].ndim
    join_logprob = pt.concatenate(
        [pt.atleast_1d(logp) for logp in logps],
        axis=axis - base_vars_ndim_supp,
    )

    return join_logprob


@node_rewriter([MakeVector, Join])
def find_measurable_stacks(
    fgraph, node
) -> Optional[List[Union[MeasurableMakeVector, MeasurableJoin]]]:
    r"""Finds `Joins`\s and `MakeVector`\s for which a `logprob` can be computed.

    Because base variables in the Join and MakeVector may be interdependent,
    the IR graph of these Ops is almost like an inner IR, except it doesn't have "valued RVs".

    To circumvent this issue, we run this rewrite early on, and tag variables as being
    "promised_valued_rvs". A perhaps more elegant alternative is to wrap the sub-graph in an OpFromGraph?
    This might avoid the repeated boxing and unboxing done to the "base_vars" throughout the life of the IR
    """

    if isinstance(node.op, (MeasurableMakeVector, MeasurableJoin)):
        return None

    is_join = isinstance(node.op, Join)

    if is_join:
        axis, *base_vars = node.inputs
    else:
        base_vars = node.inputs

    if not all(check_potential_measurability([base_var]) for base_var in base_vars):
        return None

    base_vars = assume_valued_outputs(base_vars)
    if not all(var.owner and isinstance(var.owner.op, MeasurableVariable) for var in base_vars):
        return None

    # Each base var will be "valued" by the logprob method, so other rewrites shouldn't mess with it
    # and potentially break interdependencies. For this reason, this rewrite should be applied early in
    # the IR construction
    replacements = [(base_var, promised_valued_rv(base_var)) for base_var in base_vars]
    temp_fgraph = FunctionGraph(outputs=base_vars, clone=False)
    toposort_replace(temp_fgraph, replacements)
    new_base_vars = temp_fgraph.outputs

    if is_join:
        measurable_stack = MeasurableJoin()(axis, *new_base_vars)
    else:
        measurable_stack = MeasurableMakeVector(node.op.dtype)(*new_base_vars)

    return [measurable_stack]


class MeasurableDimShuffle(DimShuffle):
    """A placeholder used to specify a log-likelihood for a dimshuffle sub-graph."""

    # Need to get the absolute path of `c_func_file`, otherwise it tries to
    # find it locally and fails when a new `Op` is initialized
    c_func_file = DimShuffle.get_path(DimShuffle.c_func_file)


MeasurableVariable.register(MeasurableDimShuffle)


@_logprob.register(MeasurableDimShuffle)
def logprob_dimshuffle(op, values, base_var, **kwargs):
    """Compute the log-likelihood graph for a `MeasurableDimShuffle`."""
    (value,) = values

    # Reverse the effects of dimshuffle on the value variable
    # First, drop any augmented dimensions and reinsert any dropped dimensions
    undo_ds: List[Union[int, str]] = [i for i, o in enumerate(op.new_order) if o != "x"]
    dropped_dims = tuple(sorted(set(op.transposition) - set(op.shuffle)))
    for dropped_dim in dropped_dims:
        undo_ds.insert(dropped_dim, "x")
    value = value.dimshuffle(undo_ds)

    # Then, unshuffle remaining dims
    original_shuffle = list(op.shuffle)
    for dropped_dim in dropped_dims:
        original_shuffle.insert(dropped_dim, dropped_dim)
    undo_ds = [original_shuffle.index(i) for i in range(len(original_shuffle))]
    value = value.dimshuffle(undo_ds)

    raw_logp = _logprob_helper(base_var, value)

    # Re-apply original dimshuffle, ignoring any support dimensions consumed by
    # the logprob function. This assumes that support dimensions are always in
    # the rightmost positions, and all we need to do is to discard the highest
    # indexes in the original dimshuffle order. Otherwise, there is no way of
    # knowing which dimensions were consumed by the logprob function.
    redo_ds = [o for o in op.new_order if o == "x" or o < raw_logp.ndim]
    return raw_logp.dimshuffle(redo_ds)


@node_rewriter([DimShuffle])
def find_measurable_dimshuffles(fgraph, node) -> Optional[List[MeasurableDimShuffle]]:
    r"""Finds `Dimshuffle`\s for which a `logprob` can be computed."""

    if isinstance(node.op, MeasurableDimShuffle):
        return None

    if not filter_measurable_variables(node.inputs):
        return None

    base_var = node.inputs[0]

    # We can only apply this rewrite directly to `RandomVariable`s, as those are
    # the only `Op`s for which we always know the support axis. Other measurable
    # variables can have arbitrary support axes (e.g., if they contain separate
    # `MeasurableDimShuffle`s). Most measurable variables with `DimShuffle`s
    # should still be supported as long as the `DimShuffle`s can be merged/
    # lifted towards the base RandomVariable.
    # TODO: If we include the support axis as meta information in each
    # intermediate MeasurableVariable, we can lift this restriction.
    if not isinstance(base_var.owner.op, RandomVariable):
        return None  # pragma: no cover

    measurable_dimshuffle = MeasurableDimShuffle(node.op.input_broadcastable, node.op.new_order)(
        base_var
    )

    return [measurable_dimshuffle]


measurable_ir_rewrites_db.register("dimshuffle_lift", local_dimshuffle_rv_lift, "basic", "tensor")


# We register this later than `dimshuffle_lift` so that it is only applied as a fallback
measurable_ir_rewrites_db.register(
    "find_measurable_dimshuffles", find_measurable_dimshuffles, "basic", "tensor"
)

early_measurable_ir_rewrites_db.register(
    "find_measurable_stacks",
    find_measurable_stacks,
    "basic",
    "tensor",
)
