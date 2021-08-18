from typing import List, Optional

import aesara.tensor as at
import numpy as np
from aesara.compile.builders import OpFromGraph
from aesara.graph.basic import Apply
from aesara.graph.fg import FunctionGraph
from aesara.graph.opt import local_optimizer, pre_greedy_local_optimizer
from aesara.ifelse import ifelse
from aesara.tensor.basic import Join, MakeVector
from aesara.tensor.random.op import RandomVariable
from aesara.tensor.random.opt import local_dimshuffle_rv_lift, local_subtensor_rv_lift
from aesara.tensor.shape import shape_tuple
from aesara.tensor.var import TensorVariable

from aeppl.abstract import MeasurableVariable
from aeppl.logprob import _logprob, logprob
from aeppl.opt import naive_bcast_rv_lift, rv_sinking_db, subtensor_ops
from aeppl.utils import get_constant_value, indices_from_subtensor


def rv_pull_down(x: TensorVariable, dont_touch_vars=None) -> TensorVariable:
    """Pull a ``RandomVariable`` ``Op`` down through a graph, when possible."""
    fgraph = FunctionGraph(outputs=dont_touch_vars or [], clone=False)

    return pre_greedy_local_optimizer(
        fgraph,
        [
            local_dimshuffle_rv_lift,
            local_subtensor_rv_lift,
            naive_bcast_rv_lift,
        ],
        x,
    )


class MixtureRV(OpFromGraph):
    """A placeholder used to specify a log-likelihood for a mixture sub-graph."""

    @classmethod
    def create_node(cls, node, indices, mixture_rvs):
        out_var = node.default_output()

        inputs = list(indices) + list(mixture_rvs)

        mixture_op = cls(
            inputs,
            [out_var],
            inline=True,
            on_unused_input="ignore",
        )

        mixture_op.name = f"{out_var.name if out_var.name else ''}-mixture"

        # new_node = mixture_op.make_node(None, None, None, *inputs)
        new_node = mixture_op(*inputs)

        return new_node.owner

    def get_non_shared_inputs(self, inputs):
        return inputs[: len(self.shared_inputs)]


MeasurableVariable.register(MixtureRV)


def get_stack_mixture_vars(
    node: Apply,
) -> Optional[List[TensorVariable]]:
    r"""Extract the mixture terms from a `*Subtensor*` applied to stacked `RandomVariable`\s."""
    if not isinstance(node.op, subtensor_ops):
        return None  # pragma: no cover

    joined_rvs = node.inputs[0]

    # First, make sure that it's some sort of concatenation
    if not (joined_rvs.owner and isinstance(joined_rvs.owner.op, (MakeVector, Join))):
        # Node is not a compatible join `Op`
        return None  # pragma: no cover

    if isinstance(joined_rvs.owner.op, MakeVector):
        mixture_rvs = joined_rvs.owner.inputs

    elif isinstance(joined_rvs.owner.op, Join):
        mixture_rvs = joined_rvs.owner.inputs[1:]
        join_axis = joined_rvs.owner.inputs[0]
        try:
            join_axis = int(get_constant_value(join_axis))
        except ValueError:
            # TODO: Support symbolic join axes
            return None

        if join_axis != 0:
            # TODO: Support other join axes
            return None

    if not all(
        rv.owner and isinstance(rv.owner.op, RandomVariable) for rv in mixture_rvs
    ):
        # Currently, all mixture components must be `RandomVariable` outputs
        # TODO: Allow constants and make them Dirac-deltas
        return None

    return mixture_rvs


@local_optimizer(subtensor_ops)
def mixture_replace(fgraph, node):
    r"""Identify mixture sub-graphs and replace them with a place-holder `Op`.

    The basic idea is to find ``stack(mixture_comps)[I_rv]``, where
    ``mixture_comps`` is a ``list`` of `RandomVariable`\s and ``I_rv`` is a
    `RandomVariable` with a discrete and finite support.
    From these terms, new terms ``Z_rv[i] = mixture_comps[i][i == I_rv]`` are
    created for each ``i`` in ``enumerate(mixture_comps)``.
    """

    rv_map_feature = getattr(fgraph, "preserve_rv_mappings", None)

    if rv_map_feature is None:
        return None  # pragma: no cover

    out_var = node.default_output()

    if out_var not in rv_map_feature.rv_values:
        return None  # pragma: no cover

    mixture_res = get_stack_mixture_vars(node)

    if mixture_res is None:
        return None  # pragma: no cover

    mixture_rvs = mixture_res

    mixture_value_var = rv_map_feature.rv_values.pop(out_var, None)

    # We loop through mixture components and collect all the array elements
    # that belong to each one (by way of their indices).
    for i, component_rv in enumerate(mixture_rvs):
        if component_rv in rv_map_feature.rv_values:
            raise ValueError("A value variable was specified for a mixture component")
        component_rv.tag.ignore_logprob = True

    # Replace this sub-graph with a `MixtureRV`
    new_node = MixtureRV.create_node(node, node.inputs[1:], mixture_rvs)

    new_mixture_rv = new_node.default_output()
    new_mixture_rv.name = "mixture"
    rv_map_feature.rv_values[new_mixture_rv] = mixture_value_var

    # FIXME: This is pretty hackish
    fgraph.import_node(new_node, import_missing=True, reason="mixture_rv")

    return [new_mixture_rv]


@_logprob.register(MixtureRV)
def logprob_MixtureRV(op, values, *inputs, name=None, **kwargs):
    (value,) = values
    inputs = op.get_non_shared_inputs(inputs)

    subtensor_node = op.outputs[0].owner
    num_indices = len(subtensor_node.inputs) - 1
    indices = inputs[:num_indices]
    indices = indices_from_subtensor(
        getattr(subtensor_node.op, "idx_list", None), indices
    )
    comp_rvs = inputs[num_indices:]

    if value.ndim > 0:
        # TODO: Make the join axis to the left-most dimension (or transpose the
        # problem)
        join_axis = 0  # op.join_axis

        value_shape = shape_tuple(value)
        logp_val = at.full(value_shape, -np.inf, dtype=value.dtype)

        for i, comp_rv in enumerate(comp_rvs):
            I_0 = indices[join_axis]
            join_indices = at.nonzero(at.eq(I_0, i))
            #
            # pre_index = (
            #     tuple(at.ogrid[tuple(slice(None, s) for s in at.shape(join_indices))])
            #     if I_0 is not None
            #     else (slice(None),)
            # )
            #
            # non_join_indices = pre_index + indices[1:]
            #
            # obs_i = value[join_indices][non_join_indices]
            obs_i = value[join_indices]

            comp_shape = shape_tuple(comp_rv)
            bcast_shape = at.broadcast_shape(
                value_shape, comp_shape, arrays_are_shapes=True
            )
            bcasted_comp_rv = at.broadcast_to(comp_rv, bcast_shape)
            zz = bcasted_comp_rv[join_indices]
            indexed_comp_rv = rv_pull_down(zz, inputs)
            # indexed_comp_rv = rv_pull_down(indexed_comp_rv[non_join_indices], inputs)

            logp_val = at.set_subtensor(
                # logp_val[join_indices][non_join_indices],
                logp_val[join_indices],
                logprob(indexed_comp_rv, obs_i),
            )

    else:
        logp_val = 0.0
        for i, comp_rv in enumerate(comp_rvs):
            comp_logp = logprob(comp_rv, value)
            logp_val += ifelse(
                at.eq(indices[0], i),
                comp_logp,
                at.as_tensor(0.0, dtype=comp_logp.dtype),
            )

    return logp_val


rv_sinking_db.register("mixture_replace", mixture_replace, -5, "basic")
