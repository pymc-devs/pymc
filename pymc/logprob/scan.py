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

from copy import copy
from typing import Callable, Dict, Iterable, List, Tuple, cast

import numpy as np
import pytensor.tensor as pt

from pytensor.graph import FunctionGraph
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.scan.op import Scan
from pytensor.scan.rewriting import scan_eqopt1, scan_eqopt2
from pytensor.scan.utils import ScanArgs
from pytensor.tensor.random.type import RandomType
from pytensor.tensor.subtensor import Subtensor, indices_from_subtensor
from pytensor.tensor.variable import TensorVariable
from pytensor.updates import OrderedUpdates

from pymc.logprob.abstract import MeasurableVariable, _logprob, valued_rv
from pymc.logprob.basic import conditional_logp
from pymc.logprob.rewriting import (
    construct_ir_fgraph,
    inc_subtensor_ops,
    logprob_rewrites_db,
    measurable_ir_rewrites_db,
    remove_valued_rvs,
)
from pymc.logprob.utils import get_related_valued_nodes
from pymc.logprob.utils import replace_rvs_by_values


class MeasurableScan(Scan):
    """A placeholder used to specify a log-likelihood for a scan sub-graph."""

    def __str__(self):
        return f"Measurable({super().__str__()})"


MeasurableVariable.register(MeasurableScan)


def convert_outer_out_to_in(
    input_scan_args: ScanArgs,
    outer_out_vars: Iterable[TensorVariable],
    new_outer_input_vars: Dict[TensorVariable, TensorVariable],
    inner_out_fn: Callable[[Dict[TensorVariable, TensorVariable]], Iterable[TensorVariable]],
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
                Tuple[int, ...], tuple(output_scan_args.inner_in_mit_sot[var_idx])
            )
            new_inner_in_seqs = inner_in_mit_sot_var + (new_inner_in_var,)
            new_inner_in_mit_sot = remove(output_scan_args.inner_in_mit_sot, var_idx)
            new_outer_in_mit_sot = remove(output_scan_args.outer_in_mit_sot, var_idx)
            new_inner_in_sit_sot = tuple(output_scan_args.inner_in_sit_sot)
            new_outer_in_sit_sot = tuple(output_scan_args.outer_in_sit_sot)
            add_nit_sot = True
        elif inner_out_info.name.endswith("sit_sot"):
            new_inner_in_seqs = (output_scan_args.inner_in_sit_sot[var_idx],) + (new_inner_in_var,)
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
                Tuple[int, ...], tuple(output_scan_args.mit_sot_in_slices[var_idx])
            )
            taps = mit_sot_var_taps + (0,)
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
            new_outer_in_nit_sot = tuple(output_scan_args.outer_in_nit_sot) + (n_steps,)
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
) -> List[Tuple[int, TensorVariable, TensorVariable]]:
    """Get the `MeasurableVariable` outputs of a `Scan` (well, its `ScanArgs`).

    Returns
    -------
    A tuple of tuples containing the index of each outer-output variable, the
    outer-output variable itself, and the inner-output variable that
    is an instance of `MeasurableVariable`.
    """
    rv_vars = []
    for n, oo_var in enumerate(
        [o for o in scan_args.outer_outputs if not isinstance(o.type, RandomType)]
    ):
        oo_info = scan_args.find_among_fields(oo_var)
        io_type = oo_info.name[(oo_info.name.index("_", 6) + 1) :]
        inner_out_type = f"inner_out_{io_type}"
        io_var = getattr(scan_args, inner_out_type)[oo_info.index]
        if io_var.owner and isinstance(io_var.owner.op, MeasurableVariable):
            rv_vars.append((n, oo_var, io_var))
    return rv_vars


def construct_scan(scan_args: ScanArgs, **kwargs) -> Tuple[List[TensorVariable], OrderedUpdates]:
    scan_op = Scan(scan_args.inner_inputs, scan_args.inner_outputs, scan_args.info, **kwargs)
    node = scan_op.make_node(*scan_args.outer_inputs)
    updates = OrderedUpdates(zip(scan_args.outer_in_shared, scan_args.outer_out_shared))
    return node.outputs, updates


@_logprob.register(MeasurableScan)
def logprob_ScanRV(op, values, *inputs, name=None, **kwargs):
    new_node = op.make_node(*inputs)
    scan_args = ScanArgs.from_node(new_node)
    rv_outer_outs = get_random_outer_outputs(scan_args)

    var_indices, rv_vars, io_vars = zip(*rv_outer_outs)
    value_map = {_rv: _val for _rv, _val in zip(rv_vars, values)}

    def create_inner_out_logp(value_map: Dict[TensorVariable, TensorVariable]) -> TensorVariable:
        """Create a log-likelihood inner-output for a `Scan`."""
        logp_parts = conditional_logp(value_map, warn_rvs=False)
        return logp_parts.values()

    logp_scan_args = convert_outer_out_to_in(
        scan_args,
        rv_vars,
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
    r"""Find `Scan`\s for which a `logprob` can be computed.

    This will convert said `Scan`\s into `MeasurableScan`\s.  It also updates
    random variable and value variable mappings that have been specified for
    parts of a `Scan`\s outputs (e.g. everything except the initial values).
    """
    from pymc.pytensorf import toposort_replace

    if not hasattr(fgraph, "shape_feature"):
        return None  # pragma: no cover

    if isinstance(node.op, Subtensor):
        node = node.inputs[0].owner
        if not (node and isinstance(node.op, Scan)):
            return None

    if isinstance(node.op, MeasurableScan):
        return None

    curr_scanargs = ScanArgs.from_node(node)

    # TODO: Check what outputs are actually needed for ValuedRVs more than one node deep
    # Find outputs of scan that are directly valued (mapping)
    direct_valued_output_idxs = [
        node.outputs.index(valued_node.inputs[0])
        for valued_node in get_related_valued_nodes(node, fgraph)
    ]

    # Find outputs of scan, whose slices are valued (recurring outputs)
    indirect_valued_output_idxs = []
    if not get_related_valued_nodes(node, fgraph):
        # For example, the output that the user got was something like
        # `out[1:]` for `outputs_info = [{"initial": x0, "taps": [-1]}]`, so
        # they likely passed `{out[1:]: x_1T_vv}` to `joint_logprob`.
        # Since `out[1:]` isn't really the output of a `Scan`, but a
        # `Subtensor` of the output `out` of a `Scan`, we need to account for that.

        # Get any `Subtensor` outputs that have been applied to outputs of this
        # `Scan` (and get the corresponding indices of the outputs from this `Scan`)
        indirect_valued_outputs = [
            (idx, client)
            for idx, out in enumerate(node.outputs)
            for client, _ in fgraph.clients[out]
            if (isinstance(client.op, Subtensor) and get_related_valued_nodes(client, fgraph))
        ]
        indirect_valued_output_idxs = [idx for idx, _ in indirect_valued_outputs]
        indirect_valued_output_nodes = [
            get_related_valued_nodes(client, fgraph)[0] for _, client in indirect_valued_outputs
        ]

        # We're going to replace the user's random variable/value variable mappings
        # with ones that map directly to outputs of this `Scan`.
        old_indirect_scan_valued_nodes = []
        new_indirect_scan_values = []
        for out_idx, valued_node in zip(indirect_valued_output_idxs, indirect_valued_output_nodes):
            [scan_rv, scan_value] = valued_node.inputs
            # The full/un-`Subtensor`ed `Scan` output that we need to use
            full_out = node.outputs[out_idx]

            assert scan_rv.owner.inputs[0] == full_out

            # A new value variable that spans the full output.
            # We don't want the old graph to appear in the new log-probability
            # graph, so we use the shape feature to (hopefully) get the shape
            # without the entire `Scan` itself.
            full_out_shape = tuple(
                fgraph.shape_feature.get_shape(full_out, i) for i in range(full_out.ndim)
            )
            new_scan_value = pt.empty(full_out_shape, dtype=full_out.dtype)

            # Set the parts of this new value variable that applied to the
            # user-specified value variable to the user's value variable
            subtensor_indices = indices_from_subtensor(
                scan_rv.owner.inputs[1:],
                scan_rv.owner.op.idx_list,
            )
            # E.g. for a single `-1` TAPS, `s_0T[1:] = s_1T` where `s_0T` is
            # `new_val_var` and `s_1T` is the user-specified value variable
            # that only spans times `t=1` to `t=T`.
            new_scan_value = pt.set_subtensor(new_scan_value[subtensor_indices], scan_value)

            # This is the outer-input that sets `s_0T[i] = taps[i]` where `i`
            # is a TAP index (e.g. a TAP of `-1` maps to index `0` in a vector
            # of the entire series).
            var_info = curr_scanargs.find_among_fields(full_out)
            alt_type = var_info.name[(var_info.name.index("_", 6) + 1) :]
            outer_input_var = getattr(curr_scanargs, f"outer_in_{alt_type}")[var_info.index]

            # These outer-inputs are using by `pytensor.scan.utils.expand_empty`, and
            # are expected to consist of only a single `set_subtensor` call.
            # That's why we can simply replace the first argument of the node.
            assert isinstance(outer_input_var.owner.op, inc_subtensor_ops)

            # We're going to set those values on our `new_val_var` so that it can
            # serve as a complete replacement for the old input `outer_input_var`.
            new_scan_value = outer_input_var.owner.clone_with_new_inputs(
                [new_scan_value] + outer_input_var.owner.inputs[1:]
            ).default_output()

            # Replace the mapping
            old_indirect_scan_valued_nodes.append(valued_node)
            new_indirect_scan_values.append(new_scan_value)

    if direct_valued_output_idxs and indirect_valued_output_idxs:
        # TODO: Allow measuring mixed direct and indirect output types
        return None
    elif direct_valued_output_idxs:
        valued_output_idxs = direct_valued_output_idxs
    elif indirect_valued_output_idxs:
        valued_output_idxs = indirect_valued_output_idxs
    else:
        return None

    # Make inner graph measurable
    inner_rv_values = {
        out: out.type() for i, out in enumerate(node.op.inner_outputs) if i in valued_output_idxs
    }
    ir_rewriter = logprob_rewrites_db.query(RewriteDatabaseQuery(include=["basic"]))
    inner_fgraph, *_ = construct_ir_fgraph(inner_rv_values, ir_rewriter=ir_rewriter)
    remove_valued_rvs(inner_fgraph)
    new_rvs = list(inner_fgraph.outputs)
    if not all(isinstance(new_out.owner.op, MeasurableVariable) for new_out in new_rvs):
        return None

    # Get new inner outs (replace measured vars in graph of non-valued outputs)
    inner_outs = node.op.inner_outputs.copy()
    old_to_new_inner_rvs = []
    for idx, rv in zip(valued_output_idxs, new_rvs):
        old_rv = inner_outs[idx]
        inner_outs[idx] = rv
        old_to_new_inner_rvs.append((old_rv, rv))
    temp_fgraph = FunctionGraph(
        outputs=inner_outs + [a for a, _ in old_to_new_inner_rvs], clone=False
    )
    toposort_replace(temp_fgraph, old_to_new_inner_rvs)
    inner_outs = temp_fgraph.outputs[: len(inner_outs)]

    op = MeasurableScan(node.op.inner_inputs, inner_outs, node.op.info, mode=copy(node.op.mode))
    new_rvs = op.make_node(*node.inputs).outputs

    replacements = {}
    if indirect_valued_output_idxs:
        # Create new ValuedRVs that sidestep the Subtensor
        for new_rv, new_value, old_valued_node in zip(
            new_rvs, new_indirect_scan_values, indirect_valued_output_nodes
        ):
            new_valued_rv = valued_rv(new_rv, new_value)
            replacements[old_valued_node.outputs[0]] = new_valued_rv
    else:
        replacements.update(dict(zip(node.outputs, new_rvs)))
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
