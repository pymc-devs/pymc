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

from collections.abc import Callable, Iterable
from copy import copy
from typing import cast

import numpy as np
import pytensor.tensor as pt

from pytensor.scan.op import Scan
from pytensor.scan.rewriting import scan_eqopt1, scan_eqopt2
from pytensor.scan.utils import ScanArgs
from pytensor.tensor.basic import AllocEmpty
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.subtensor import IncSubtensor
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.basic import conditional_logp
from pymc.logprob.query import (
    contains_random,
    infer_support_axes,
    other_query_uses,
    output_queries,
    query_parts,
    rewrite_logprob_query,
    support_axes,
)
from pymc.logprob.rewriting import (
    logprob_rewrites_db,
)
from pymc.logprob.utils import replace_rvs_by_values


def convert_outer_out_to_in(
    input_scan_args: ScanArgs,
    outer_out_vars: Iterable[TensorVariable],
    new_outer_input_vars: dict[TensorVariable, TensorVariable],
    inner_out_fn: Callable[[dict[TensorVariable, TensorVariable]], Iterable[TensorVariable]],
) -> ScanArgs:
    r"""Convert outer-graph outputs into outer-graph inputs.

    Parameters
    ----------
    input_scan_args:
        The source `Scan` arguments.
    outer_out_vars:
        The outer-graph output variables that are to be converted into an
        outer-graph input.
    new_outer_input_vars:
        The variables used for the new outer-graph input computed for
        `outer_out_vars`.
    inner_out_fn:
        A function that takes the remapped outer-out variables and produces new
        inner-graph outputs.  This can be used to transform the
        `outer_out_vars`\s' corresponding inner-graph outputs into something
        else entirely, like log-probabilities.

    Outputs
    =======
    A `ScanArgs` object for a `Scan` in which `outer_out_vars` has been converted to an
    outer-graph input.
    """
    output_scan_args = copy(input_scan_args)
    inner_outs_to_new_inner_ins = {}

    # Map inner-outputs to outer-outputs
    old_inner_outs_to_outer_outs = {}

    for oo_var in outer_out_vars:
        var_info = output_scan_args.find_among_fields(
            oo_var, field_filter=lambda x: x.startswith("outer_out")
        )

        assert var_info is not None
        assert oo_var in new_outer_input_vars

        io_var = output_scan_args.get_alt_field(var_info, "inner_out")
        old_inner_outs_to_outer_outs[io_var] = oo_var

    # In this loop, we gather information about the new inner-inputs that have
    # been created and what their corresponding inner-outputs were, and we
    # update the outer and inner-inputs to reflect the addition of new
    # inner-inputs.
    for old_inner_out_var, oo_var in old_inner_outs_to_outer_outs.items():
        # Couldn't one do the same with `var_info`?
        inner_out_info = output_scan_args.find_among_fields(
            old_inner_out_var, field_filter=lambda x: x.startswith("inner_out")
        )

        output_scan_args.remove_from_fields(old_inner_out_var, rm_dependents=False)

        # Remove the old outer-output variable.
        # Not sure if this really matters, since we don't use the outer-outputs
        # when building a new `Scan`, but doing it keeps the `ScanArgs` object
        # consistent.
        output_scan_args.remove_from_fields(oo_var, rm_dependents=False)

        # Use the index for the specific inner-graph sub-collection to which this
        # variable belongs (e.g. index `1` among the inner-graph sit-sot terms)
        var_idx = inner_out_info.index

        # The old inner-output variable becomes the a new inner-input
        new_inner_in_var = old_inner_out_var.clone()
        if new_inner_in_var.name:
            new_inner_in_var.name = f"{new_inner_in_var.name}_vv"

        inner_outs_to_new_inner_ins[old_inner_out_var] = new_inner_in_var

        # We want to remove elements from both lists and tuples, because the
        # members of `ScanArgs` could switch from being `list`s to `tuple`s
        # soon
        def remove(x, i):
            return x[:i] + x[i + 1 :]

        # If we're replacing a [m|s]it-sot, then we need to add a new nit-sot
        add_nit_sot = False
        if inner_out_info.name.endswith("mit_sot"):
            inner_in_mit_sot_var = cast(
                tuple[int, ...], tuple(output_scan_args.inner_in_mit_sot[var_idx])
            )
            new_inner_in_seqs = (*inner_in_mit_sot_var, new_inner_in_var)
            new_inner_in_mit_sot = remove(output_scan_args.inner_in_mit_sot, var_idx)
            new_outer_in_mit_sot = remove(output_scan_args.outer_in_mit_sot, var_idx)
            new_inner_in_sit_sot = tuple(output_scan_args.inner_in_sit_sot)
            new_outer_in_sit_sot = tuple(output_scan_args.outer_in_sit_sot)
            add_nit_sot = True
        elif inner_out_info.name.endswith("sit_sot"):
            new_inner_in_seqs = (output_scan_args.inner_in_sit_sot[var_idx], new_inner_in_var)
            new_inner_in_sit_sot = remove(output_scan_args.inner_in_sit_sot, var_idx)
            new_outer_in_sit_sot = remove(output_scan_args.outer_in_sit_sot, var_idx)
            new_inner_in_mit_sot = tuple(output_scan_args.inner_in_mit_sot)
            new_outer_in_mit_sot = tuple(output_scan_args.outer_in_mit_sot)
            add_nit_sot = True
        else:
            new_inner_in_seqs = (new_inner_in_var,)
            new_inner_in_mit_sot = tuple(output_scan_args.inner_in_mit_sot)
            new_outer_in_mit_sot = tuple(output_scan_args.outer_in_mit_sot)
            new_inner_in_sit_sot = tuple(output_scan_args.inner_in_sit_sot)
            new_outer_in_sit_sot = tuple(output_scan_args.outer_in_sit_sot)

        output_scan_args.inner_in_mit_sot = list(new_inner_in_mit_sot)
        output_scan_args.inner_in_sit_sot = list(new_inner_in_sit_sot)
        output_scan_args.outer_in_mit_sot = list(new_outer_in_mit_sot)
        output_scan_args.outer_in_sit_sot = list(new_outer_in_sit_sot)

        if inner_out_info.name.endswith("mit_sot"):
            mit_sot_var_taps = cast(
                tuple[int, ...], tuple(output_scan_args.mit_sot_in_slices[var_idx])
            )
            taps = (*mit_sot_var_taps, 0)
            new_mit_sot_in_slices = remove(output_scan_args.mit_sot_in_slices, var_idx)
        elif inner_out_info.name.endswith("sit_sot"):
            taps = (-1, 0)
            new_mit_sot_in_slices = tuple(output_scan_args.mit_sot_in_slices)
        else:
            taps = (0,)
            new_mit_sot_in_slices = tuple(output_scan_args.mit_sot_in_slices)

        output_scan_args.mit_sot_in_slices = list(new_mit_sot_in_slices)

        taps, new_inner_in_seqs = zip(*sorted(zip(taps, new_inner_in_seqs), key=lambda x: x[0]))

        new_inner_in_seqs = tuple(output_scan_args.inner_in_seqs) + tuple(
            reversed(new_inner_in_seqs)
        )

        output_scan_args.inner_in_seqs = list(new_inner_in_seqs)

        slice_seqs = zip(-np.asarray(taps), [n if n < 0 else None for n in reversed(taps)])

        # XXX: If the caller passes the variables output by `pytensor.scan`, it's
        # likely that this will fail, because those variables can sometimes be
        # slices of the actual outer-inputs (e.g. `out[1:]` instead of `out`
        # when `taps=[-1]`).
        var_slices = [new_outer_input_vars[oo_var][b:e] for b, e in slice_seqs]
        n_steps = pt.min([pt.shape(n)[0] for n in var_slices])

        output_scan_args.n_steps = n_steps

        new_outer_in_seqs = tuple(output_scan_args.outer_in_seqs) + tuple(
            v[:n_steps] for v in var_slices
        )

        output_scan_args.outer_in_seqs = list(new_outer_in_seqs)

        if add_nit_sot:
            new_outer_in_nit_sot = (*output_scan_args.outer_in_nit_sot, n_steps)
        else:
            new_outer_in_nit_sot = tuple(output_scan_args.outer_in_nit_sot)

        output_scan_args.outer_in_nit_sot = list(new_outer_in_nit_sot)

    # Now, we can add new inner-outputs for the custom calculations.
    # We don't need to create corresponding outer-outputs, because `Scan` will
    # do that when we call `Scan.make_node`.  All we need is a consistent
    # outer-inputs and inner-graph spec., which we should have in
    # `output_scan_args`.
    remapped_io_to_ii = inner_outs_to_new_inner_ins
    new_inner_out_nit_sot = tuple(output_scan_args.inner_out_nit_sot) + tuple(
        inner_out_fn(remapped_io_to_ii)
    )
    output_scan_args.inner_out_nit_sot = list(new_inner_out_nit_sot)

    # Finally, we need to replace any lingering references to the new
    # internal variables that could be in the recurrent states needed
    # to compute the new nit_sots
    traced_outs = (
        output_scan_args.inner_out_mit_sot
        + output_scan_args.inner_out_sit_sot
        + output_scan_args.inner_out_nit_sot
    )
    traced_outs = replace_rvs_by_values(traced_outs, rvs_to_values=remapped_io_to_ii)
    # Update output mappings
    n_mit_sot = len(output_scan_args.inner_out_mit_sot)
    output_scan_args.inner_out_mit_sot = traced_outs[:n_mit_sot]
    offset = n_mit_sot
    n_sit_sot = len(output_scan_args.inner_out_sit_sot)
    output_scan_args.inner_out_sit_sot = traced_outs[offset : offset + n_sit_sot]
    offset += n_sit_sot
    n_nit_sot = len(output_scan_args.inner_out_nit_sot)
    output_scan_args.inner_out_nit_sot = traced_outs[offset : offset + n_nit_sot]

    return output_scan_args


def get_random_outer_outputs(
    scan_args: ScanArgs,
) -> list[tuple[int, TensorVariable, TensorVariable]]:
    """Return (outer index, outer output, inner output) for stochastic Scan outputs."""
    rv_vars = []
    for n, oo_var in enumerate(
        [o for o in scan_args.outer_outputs if not isinstance(o.type, RandomType)]
    ):
        oo_info = scan_args.find_among_fields(oo_var)
        io_type = oo_info.name[(oo_info.name.index("_", 6) + 1) :]
        inner_out_type = f"inner_out_{io_type}"
        io_var = getattr(scan_args, inner_out_type)[oo_info.index]
        if contains_random(io_var):
            rv_vars.append((n, oo_var, io_var))
    return rv_vars


def construct_scan(scan_args: ScanArgs, **kwargs) -> tuple[list[TensorVariable], dict]:
    scan_op = Scan(scan_args.inner_inputs, scan_args.inner_outputs, scan_args.info, **kwargs)
    node = scan_op.make_node(*scan_args.outer_inputs)
    updates = dict(zip(scan_args.outer_in_shared, scan_args.outer_out_shared))
    return node.outputs, updates


def get_initval_from_scan_tap_input(inp) -> TensorVariable:
    """Get initval from the buffer allocated to tap (recurring) inputs.

    Raises ValueError, if input does not correspond to expected graph.
    """
    if not isinstance(inp.owner.op, IncSubtensor) and inp.owner.op.set_instead_of_inc:
        raise ValueError

    idx_list = inp.owner.op.idx_list
    if not len(idx_list) == 1:
        raise ValueError

    [idx_slice] = idx_list
    if not (
        isinstance(idx_slice, slice)
        and idx_slice.start is None
        and idx_slice.stop is not None
        and idx_slice.step is None
    ):
        raise ValueError

    empty, initval, _ = inp.owner.inputs
    if not isinstance(empty.owner.op, AllocEmpty):
        raise ValueError

    return initval


def logprob_scan(op, values, *inputs, name=None, output_indices=None, **kwargs):
    new_node = op.make_node(*inputs)
    # clone=True thaws the frozen inner graph into a mutable copy
    scan_args = ScanArgs.from_node(new_node, clone=True)
    rv_outer_outs = get_random_outer_outputs(scan_args)
    if output_indices is not None:
        rv_outer_outs = [item for item in rv_outer_outs if item[0] in output_indices]

    # values = (pt.zeros(11)[1:].set(values[0]),)
    # For random variable sequences with taps, we need to place the value variable in the
    # input tensor that contains the initial state and the empty buffer for the output
    values = list(values)
    var_indices, outer_rvs, inner_rvs = zip(*rv_outer_outs)
    for inp, out in zip(
        scan_args.outer_in_sit_sot + scan_args.outer_in_mit_sot,
        scan_args.outer_out_sit_sot + scan_args.outer_out_mit_sot,
    ):
        if out not in outer_rvs:
            continue

        # Tap inputs should be a SetSubtensor(empty()[:start], initial_value)
        # We will replace it by Join(axis=0, initial_value, value)
        initval = get_initval_from_scan_tap_input(inp)
        idx = outer_rvs.index(out)
        values[idx] = pt.join(0, initval, values[idx])

    value_map = dict(zip(outer_rvs, values))

    def create_inner_out_logp(value_map: dict[TensorVariable, TensorVariable]) -> TensorVariable:
        """Create a log-likelihood inner-output for a `Scan`."""
        logp_parts = conditional_logp(value_map, warn_rvs=False)
        return logp_parts.values()

    logp_scan_args = convert_outer_out_to_in(
        scan_args,
        outer_rvs,
        value_map,
        inner_out_fn=create_inner_out_logp,
    )

    # Unrequested mapping outputs do not carry state between steps. Keep only
    # the density outputs; auxiliary stochastic outputs must not retain sampling.
    logp_scan_args.inner_out_nit_sot = logp_scan_args.inner_out_nit_sot[-len(values) :]
    logp_scan_args.outer_in_nit_sot = logp_scan_args.outer_in_nit_sot[-len(values) :]
    logp_scan_args.outer_out_nit_sot = []

    # Remove the shared variables corresponding to replaced terms.

    # TODO FIXME: This is a really dirty approach, because it effectively
    # assumes that all sampling is being removed, and, thus, all shared updates
    # relating to `RandomType`s.  Instead, we should be more precise and only
    # remove the `RandomType`s associated with `values`.
    logp_scan_args.outer_in_shared = [
        i for i in logp_scan_args.outer_in_shared if not isinstance(i.type, RandomType)
    ]
    logp_scan_args.inner_in_shared = [
        i for i in logp_scan_args.inner_in_shared if not isinstance(i.type, RandomType)
    ]
    logp_scan_args.inner_out_shared = [
        i for i in logp_scan_args.inner_out_shared if not isinstance(i.type, RandomType)
    ]
    # XXX TODO: Remove this properly
    # logp_scan_args.outer_out_shared = []

    logp_scan_out, _updates = construct_scan(logp_scan_args, mode=op.mode)

    # Return only the logp outputs, not any potentially carried states
    logp_outputs = logp_scan_out[-len(values) :]

    if len(logp_outputs) == 1:
        return logp_outputs[0]
    return logp_outputs


# Add scan canonicalizations that aren't in the canonicalization DB
logprob_rewrites_db.register("scan_eqopt1", scan_eqopt1, "basic", "scan")
logprob_rewrites_db.register("scan_eqopt2", scan_eqopt2, "basic", "scan")


@infer_support_axes.register(Scan)
def measure_scan(op, var):
    mapping = op.get_oinp_iinp_iout_oout_mappings()["inner_out_from_outer_out"]
    inner = op.inner_outputs[mapping[var.index][-1]]
    axes = support_axes(inner)
    return tuple(axis + var.ndim - inner.ndim for axis in axes)


@rewrite_logprob_query.register(Scan)
def rewrite_scan_logprob(op, fgraph, query, **kwargs):
    rv, _ = query_parts(query)
    producer = rv.owner
    if op.info.as_while or op.info.n_mit_mot > 0:
        return None
    queries = output_queries(fgraph, producer)
    outputs = [out for out in producer.outputs if out in queries]
    missing = [
        out
        for out in producer.outputs
        if out not in queries and not isinstance(out.type, RandomType)
    ]
    if other_query_uses(fgraph, missing, set(queries.values())):
        return None
    values = [query_parts(queries[out])[1] for out in outputs]
    terms = logprob_scan(
        op, values, *producer.inputs, output_indices=[out.index for out in outputs], **kwargs
    )
    terms = terms if isinstance(terms, list | tuple) else [terms]
    return {queries[out].outputs[0]: term for out, term in zip(outputs, terms, strict=True)}


def scan_slice_source(node):
    """Recognize the slice that removes Scan's initial tap values."""
    from pytensor.tensor.exceptions import NotScalarConstantError
    from pytensor.tensor.subtensor import unflatten_index_variables

    source = node.inputs[0]
    args = ScanArgs.from_node(source.owner)
    outputs = args.outer_out_sit_sot + args.outer_out_mit_sot
    if source not in outputs:
        return None
    taps = [[-1]] * len(args.outer_out_sit_sot) + list(args.mit_sot_in_slices)
    start = abs(min(taps[outputs.index(source)]))
    indices = unflatten_index_variables(node.inputs[1:], node.op.idx_list)
    if len(indices) != 1 or not isinstance(indices[0], slice):
        return None
    idx = indices[0]
    if idx.stop is not None or idx.step is not None:
        return None
    try:
        if pt.get_underlying_scalar_constant_value(idx.start) != start:
            return None
    except NotScalarConstantError:
        return None
    return source
