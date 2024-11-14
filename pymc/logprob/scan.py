#   Copyright 2024 The PyMC Developers
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

from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import scan_eqopt1, scan_eqopt2
from pytensor.scan.utils import ScanArgs
from pytensor.tensor.basic import AllocEmpty
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.subtensor import IncSubtensor, Subtensor
from pytensor.tensor.variable import TensorVariable
from pytensor.updates import OrderedUpdates

from pymc.logprob.abstract import MeasurableOp, _logprob
from pymc.logprob.basic import conditional_logp
from pymc.logprob.rewriting import (
    construct_ir_fgraph,
    logprob_rewrites_db,
    measurable_ir_rewrites_db,
    remove_valued_rvs,
)
from pymc.logprob.utils import get_related_valued_nodes, replace_rvs_by_values
from pymc.pytensorf import toposort_replace


class MeasurableScan(MeasurableOp, Scan):
    """A placeholder used to specify a log-likelihood for a scan sub-graph."""

    def __str__(self):
        """Return a string representation of the object."""
        return f"Measurable{super().__str__()}"


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
    """Get the measurable outputs of a `Scan` (well, its `ScanArgs`).

    Returns
    -------
    A tuple of tuples containing the index of each outer-output variable, the
    outer-output variable itself, and the inner-output variable that
    is an instance of `MeasurableOp` variable.
    """
    rv_vars = []
    for n, oo_var in enumerate(
        [o for o in scan_args.outer_outputs if not isinstance(o.type, RandomType)]
    ):
        oo_info = scan_args.find_among_fields(oo_var)
        io_type = oo_info.name[(oo_info.name.index("_", 6) + 1) :]
        inner_out_type = f"inner_out_{io_type}"
        io_var = getattr(scan_args, inner_out_type)[oo_info.index]
        if io_var.owner and isinstance(io_var.owner.op, MeasurableOp):
            rv_vars.append((n, oo_var, io_var))
    return rv_vars


def construct_scan(scan_args: ScanArgs, **kwargs) -> tuple[list[TensorVariable], OrderedUpdates]:
    scan_op = Scan(scan_args.inner_inputs, scan_args.inner_outputs, scan_args.info, **kwargs)
    node = scan_op.make_node(*scan_args.outer_inputs)
    updates = OrderedUpdates(zip(scan_args.outer_in_shared, scan_args.outer_out_shared))
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


@_logprob.register(MeasurableScan)
def logprob_scan(op, values, *inputs, name=None, **kwargs):
    new_node = op.make_node(*inputs)
    scan_args = ScanArgs.from_node(new_node)
    rv_outer_outs = get_random_outer_outputs(scan_args)

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

    logp_scan_out, updates = construct_scan(logp_scan_args, mode=op.mode)

    # Automatically pick up updates so that we don't have to pass them around
    for key, value in updates.items():
        key.default_update = value

    # Return only the logp outputs, not any potentially carried states
    logp_outputs = logp_scan_out[-len(values) :]

    if len(logp_outputs) == 1:
        return logp_outputs[0]
    return logp_outputs


@node_rewriter([Scan, Subtensor])
def find_measurable_scans(fgraph, node):
    r"""Find `Scan`\s for which a `logprob` can be computed."""
    if isinstance(node.op, Subtensor):
        node = node.inputs[0].owner
        if not (node and isinstance(node.op, Scan)):
            return None

    if isinstance(node.op, MeasurableScan):
        return None

    if node.op.info.as_while:  # May work but we haven't tested it
        return None

    if node.op.info.n_mit_mot > 0:
        return None

    scan_args = ScanArgs.from_node(node)

    # TODO: Check what outputs are actually needed for ValuedRVs more than one node deep

    # To make the inner graph measurable, we need to know which inner outputs we are conditioning on from the outside
    # If there is only one output, we could always try to make it measurable, but with more outputs it would be ambiguous.
    # For example, if we have out1 = normal() and out2 = out1 + const, it's valid to condition on either (but not both).

    # Find outputs of scan that are directly valued.
    # These must be mapping outputs, such as `outputs_info = [None]` (i.e, no recurrence nit_sot outputs)
    direct_valued_outputs = [
        valued_node.inputs[0] for valued_node in get_related_valued_nodes(fgraph, node)
    ]
    if not all(valued_out in scan_args.outer_out_nit_sot for valued_out in direct_valued_outputs):
        return None

    # Find indirect (sliced) outputs of scan that are valued.
    # These must be recurring outputs, such as `outputs_info = [{"initial": x0, "taps": [-1]}]` (i.e, recurring sit-sot or mit-sot outputs)
    # For these outputs, the scan helper returns `out[abs(min(taps)):]` (out[:abs(min(taps))] includes the initial values)
    # This means that it's a Subtensor output, not a direct Scan output, that the user requests the logp of.
    sliced_valued_outputs = [
        client.outputs[0]
        for out in node.outputs
        for client, _ in fgraph.clients[out]
        if (isinstance(client.op, Subtensor) and get_related_valued_nodes(fgraph, client))
    ]
    indirect_valued_outputs = [out.owner.inputs[0] for out in sliced_valued_outputs]
    if not all(
        (valued_out in scan_args.outer_out_sit_sot or valued_out in scan_args.outer_out_mit_sot)
        for valued_out in indirect_valued_outputs
    ):
        return None

    valued_outputs = direct_valued_outputs + indirect_valued_outputs

    if not valued_outputs:
        return None

    valued_output_idxs = [node.outputs.index(out) for out in valued_outputs]

    # Make inner graph measurable
    mapping = node.op.get_oinp_iinp_iout_oout_mappings()["inner_out_from_outer_out"]
    inner_rvs = [node.op.inner_outputs[mapping[idx][-1]] for idx in valued_output_idxs]
    inner_fgraph = construct_ir_fgraph({rv: rv.type() for rv in inner_rvs})
    remove_valued_rvs(inner_fgraph)
    inner_rvs = list(inner_fgraph.outputs)
    if not all(isinstance(new_out.owner.op, MeasurableOp) for new_out in inner_rvs):
        return None

    # Create MeasurableScan with new inner outs
    # We must also replace any lingering references to the old RVs by the new measurable RVS
    # For example if we had measurable out1 = exp(normal()) and out2 = out1 - x
    # We need to replace references of original out1 by the new MeasurableExp(normal())
    clone_fgraph = node.op.fgraph.clone()
    inner_inps = clone_fgraph.inputs
    inner_outs = clone_fgraph.outputs
    inner_rvs_replacements = []
    for idx, new_inner_rv in zip(valued_output_idxs, inner_rvs, strict=True):
        old_inner_rv = inner_outs[idx]
        inner_outs[idx] = new_inner_rv
        inner_rvs_replacements.append((old_inner_rv, new_inner_rv))
    temp_fgraph = FunctionGraph(
        outputs=inner_outs + [a for a, _ in inner_rvs_replacements],
        clone=False,
    )
    toposort_replace(temp_fgraph, inner_rvs_replacements)
    op = MeasurableScan(inner_inps, inner_outs, node.op.info, mode=copy(node.op.mode))
    new_outs = op.make_node(*node.inputs).outputs

    old_outs = node.outputs
    replacements = {}
    for old_out, new_out in zip(old_outs, new_outs):
        if old_out in indirect_valued_outputs:
            # We sidestep the Subtensor operation, which is not relevant for the logp
            sliced_idx = indirect_valued_outputs.index(old_out)
            old_out = sliced_valued_outputs[sliced_idx]
            replacements[old_out] = new_out
        else:
            replacements[old_out] = new_out

    return replacements


measurable_ir_rewrites_db.register(
    "find_measurable_scans",
    find_measurable_scans,
    "basic",
    "scan",
)

# Add scan canonicalizations that aren't in the canonicalization DB
logprob_rewrites_db.register("scan_eqopt1", scan_eqopt1, "basic", "scan")
logprob_rewrites_db.register("scan_eqopt2", scan_eqopt2, "basic", "scan")
