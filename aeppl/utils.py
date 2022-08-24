from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import aesara.tensor as at
import numpy as np
from aesara import config
from aesara.graph.basic import Constant, Variable, clone_get_equiv, graph_inputs, walk
from aesara.graph.fg import FunctionGraph
from aesara.graph.op import compute_test_value
from aesara.graph.rewriting.utils import rewrite_graph
from aesara.link.c.type import CType
from aesara.tensor.rewriting.basic import topo_constant_folding
from aesara.tensor.rewriting.shape import ShapeFeature
from aesara.tensor.sharedvar import SharedVariable
from aesara.tensor.subtensor import AdvancedIncSubtensor, AdvancedIncSubtensor1
from aesara.tensor.var import TensorVariable

from aeppl.abstract import MeasurableVariable

PotentialShapeType = Union[
    int,
    np.ndarray,
    Tuple[Union[int, Variable], ...],
    List[Union[int, Variable]],
    Variable,
]


def change_rv_size(
    rv_var: TensorVariable,
    new_size: PotentialShapeType,
    expand: Optional[bool] = False,
) -> TensorVariable:
    """Change or expand the size of a `RandomVariable`.

    Parameters
    ==========
    rv_var
        The `RandomVariable` output.
    new_size
        The new size.
    expand:
        Expand the existing size by `new_size`.

    """
    rv_node = rv_var.owner
    rng, size, dtype, *dist_params = rv_node.inputs
    name = rv_var.name
    tag = rv_var.tag

    if expand:
        if rv_node.op.ndim_supp == 0 and at.get_vector_length(size) == 0:
            size = rv_node.op._infer_shape(size, dist_params)
        new_size = tuple(at.atleast_1d(new_size)) + tuple(size)

    # Make sure the new size is a tensor. This helps to not unnecessarily pick
    # up a `Cast` in some cases
    new_size = at.as_tensor(new_size, ndim=1, dtype="int64")

    new_rv_node = rv_node.op.make_node(rng, new_size, dtype, *dist_params)
    rv_var = new_rv_node.outputs[-1]
    rv_var.name = name
    for k, v in tag.__dict__.items():
        rv_var.tag.__dict__.setdefault(k, v)

    if config.compute_test_value != "off":
        compute_test_value(new_rv_node)

    return rv_var


def extract_obs_data(x: TensorVariable) -> np.ndarray:
    """Extract data from observed symbolic variables.

    Raises
    ------
    TypeError

    """
    if isinstance(x, Constant):
        return x.data
    if isinstance(x, SharedVariable):
        return x.get_value()
    if x.owner and isinstance(
        x.owner.op, (AdvancedIncSubtensor, AdvancedIncSubtensor1)
    ):
        array_data = extract_obs_data(x.owner.inputs[0])
        mask_idx = tuple(extract_obs_data(i) for i in x.owner.inputs[2:])
        mask = np.zeros_like(array_data)
        mask[mask_idx] = 1
        return np.ma.MaskedArray(array_data, mask)

    raise TypeError(f"Data cannot be extracted from {x}")


def walk_model(
    graphs: Iterable[TensorVariable],
    walk_past_rvs: bool = False,
    stop_at_vars: Optional[Set[TensorVariable]] = None,
    expand_fn: Callable[[TensorVariable], List[TensorVariable]] = lambda var: [],
) -> Generator[TensorVariable, None, None]:
    """Walk model graphs and yield their nodes.

    By default, these walks will not go past ``MeasurableVariable`` nodes.

    Parameters
    ==========
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
    ==========
    graphs
        The graphs in which random variables are to be replaced.

    Returns
    =======
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
    ==========
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
        tuple(convert_indices(list(indices), idx) for idx in idx_list)
        if idx_list
        else indices
    )


def get_constant_value(x):
    """Use constant folding to get a constant value."""

    axis_fg = FunctionGraph(outputs=[x], features=[ShapeFeature()], clone=True)

    (folded_axis,) = rewrite_graph(
        axis_fg, custom_rewrite=topo_constant_folding
    ).outputs

    if not isinstance(folded_axis, Constant):
        raise ValueError("Could not obtain a constant from constant folding")

    return folded_axis.data
