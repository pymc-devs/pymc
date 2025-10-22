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
import functools
import warnings

from collections.abc import Callable, Sequence

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor import graph_replace
from pytensor.compile.ops import TypeCastingOp
from pytensor.graph.basic import Apply, Variable
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.db import RewriteDatabaseQuery, SequenceDB
from pytensor.graph.traversal import ancestors, walk
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.transforms import Transform
from pymc.pytensorf import (
    SeedSequenceSeed,
    compile,
    find_rng_nodes,
    replace_rng_nodes,
    reseed_rngs,
    toposort_replace,
)
from pymc.util import get_transformed_name, get_untransformed_name, is_transformed_name

StartDict = dict[Variable | str, np.ndarray | Variable | str]
PointType = dict[str, np.ndarray]
initial_point_rewrites_db = SequenceDB()
initial_point_basic_query = RewriteDatabaseQuery(include=["basic"])


def convert_str_to_rv_dict(
    model, start: StartDict
) -> dict[TensorVariable, np.ndarray | Variable | str | None]:
    """Convert a user-provided start dict to an untransformed RV start dict.

    Converts a dict of str keys of (transformed) variable names
    to a dict mapping the RV tensors to untransformed initvals.

    TODO: Deprecate this functionality and only accept TensorVariables as keys.
    """
    initvals = {}
    for key, initval in start.items():
        if isinstance(key, str):
            if is_transformed_name(key):
                rv = model[get_untransformed_name(key)]
                initvals[rv] = model.rvs_to_transforms[rv].backward(initval, *rv.owner.inputs)
            else:
                initvals[model[key]] = initval
        else:
            initvals[key] = initval
    return initvals


def make_initial_point_fns_per_chain(
    *,
    model,
    overrides: StartDict | Sequence[StartDict | None] | None,
    jitter_rvs: set[TensorVariable] | None = None,
    chains: int,
) -> list[Callable[[SeedSequenceSeed], PointType]]:
    """Create an initial point function for each chain, as defined by initvals.

    If a single initval dictionary is passed, the function is replicated for each
    chain, otherwise a unique function is compiled for each entry in the dictionary.

    Parameters
    ----------
    overrides : optional, list or dict
        Initial value strategy overrides that should take precedence over the defaults from the model.
        A sequence of None or dicts will be treated as chain-wise strategies and must have the same length as `seeds`.
    jitter_rvs : set, optional
        Random variable tensors for which U(-1, 1) jitter shall be applied.
        (To the transformed space if applicable.)

    Returns
    -------
    ipfns : list[Callable[[SeedSequenceSeed], dict[str, np.ndarray]]]
        list of functions that return initial points for each chain.

    Raises
    ------
    ValueError
        If the number of entries in initvals is different than the number of chains

    """
    if isinstance(overrides, dict) or overrides is None:
        # One strategy for all chains
        # Only one function compilation is needed.
        ipfns = [
            make_initial_point_fn(
                model=model,
                overrides=overrides,
                jitter_rvs=jitter_rvs,
                return_transformed=True,
            )
        ] * chains
    elif len(overrides) == chains:
        ipfns = [
            make_initial_point_fn(
                model=model,
                jitter_rvs=jitter_rvs,
                overrides=chain_overrides,
                return_transformed=True,
            )
            for chain_overrides in overrides
        ]
    else:
        raise ValueError(
            f"Number of initval dicts ({len(overrides)}) does not match the number of chains ({chains})."
        )

    return ipfns


def make_initial_point_fn(
    *,
    model,
    overrides: StartDict | None = None,
    jitter_rvs: set[TensorVariable] | None = None,
    default_strategy: str = "support_point",
    return_transformed: bool = True,
) -> Callable[[SeedSequenceSeed], PointType]:
    """Create seeded function that computes initial values for all free model variables.

    Parameters
    ----------
    jitter_rvs : set
        The set (or list or tuple) of random variables for which a U(-1, +1) jitter should be
        added to the initial value. Only available for variables that have a transform or real-valued support.
    default_strategy : str
        Which of { "support_point", "prior" } to prefer if the initval setting for an RV is None.
    overrides : dict
        Initial value (strategies) to use instead of what's specified in `Model.initial_values`.
    return_transformed : bool
        If `True` the returned variables will correspond to transformed initial values.

    Returns
    -------
    initial_point_fn : Callable[[SeedSequenceSeed], dict[str, np.ndarray]]
    """
    sdict_overrides = convert_str_to_rv_dict(model, overrides or {})
    initval_strats = {
        **model.rvs_to_initial_values,
        **sdict_overrides,
    }

    initial_values = make_initial_point_expression(
        free_rvs=model.free_RVs,
        rvs_to_transforms=model.rvs_to_transforms,
        initval_strategies=initval_strats,
        jitter_rvs=jitter_rvs,
        default_strategy=default_strategy,
        return_transformed=return_transformed,
    )

    # Replace original rng shared variables so that we don't mess with them
    # when calling the final seeded function
    initial_values = replace_rng_nodes(initial_values)
    func = compile(inputs=[], outputs=initial_values, mode=pytensor.compile.mode.FAST_COMPILE)

    varnames = []
    for var in model.free_RVs:
        transform = model.rvs_to_transforms[var]
        if transform is not None and return_transformed:
            name = get_transformed_name(var.name, transform)
        else:
            name = var.name
        varnames.append(name)

    def make_seeded_function(func):
        rngs = find_rng_nodes(func.maker.fgraph.outputs)

        @functools.wraps(func)
        def inner(seed, *args, **kwargs):
            reseed_rngs(rngs, seed)
            values = func(*args, **kwargs)
            return dict(zip(varnames, values))

        return inner

    return make_seeded_function(func)


class InitialPoint(TypeCastingOp):
    def make_node(self, var):
        return Apply(self, [var], [var.type()])


def non_support_point_ancestors(value):
    def expand(r: Variable):
        node = r.owner
        if node is not None and not isinstance(node.op, InitialPoint):
            # Stop graph traversal at InitialPoint ops
            return node.inputs
        return None

    yield from walk([value], expand, bfs=False)


initial_point_op = InitialPoint()


def make_initial_point_expression(
    *,
    free_rvs: Sequence[TensorVariable],
    rvs_to_transforms: dict[TensorVariable, Transform],
    initval_strategies: dict[TensorVariable, np.ndarray | Variable | str | None],
    jitter_rvs: set[TensorVariable] | None = None,
    default_strategy: str = "support_point",
    return_transformed: bool = False,
) -> list[TensorVariable]:
    """Create the tensor variables that need to be evaluated to obtain an initial point.

    Parameters
    ----------
    free_rvs : list
        Tensors of free random variables in the model.
    rvs_to_values : dict
        Mapping of free random variable tensors to value variable tensors.
    initval_strategies : dict
        Mapping of free random variable tensors to initial value strategies.
        For example the `Model.initial_values` dictionary.
    jitter_rvs : set
        The set (or list or tuple) of random variables for which a U(-1, +1) jitter should be
        added to the initial value. Only available for variables that have a transform or real-valued support.
    default_strategy : str
        Which of { "support_point", "prior" } to prefer if the initval strategy setting for an RV is None.
    return_transformed : bool
        Switches between returning the tensors for untransformed or transformed initial points.

    Returns
    -------
    initial_points : list of TensorVariable
        PyTensor expressions for initial values of the free random variables.
    """
    from pymc.distributions.distribution import support_point

    if jitter_rvs is None:
        jitter_rvs = set()

    # Clone free_rvs so we don't modify the original graph
    initial_point_fgraph = FunctionGraph(outputs=free_rvs, clone=True)
    # Wrap each rv in an initial_point Operation to avoid losing dependency between the RVs
    replacements = tuple((rv, initial_point_op(rv)) for rv in initial_point_fgraph.outputs)
    toposort_replace(initial_point_fgraph, replacements, reverse=True)

    # Apply any rewrites necessary to compute the initial points.
    initial_point_rewriter = initial_point_rewrites_db.query(initial_point_basic_query)
    if initial_point_rewriter:
        initial_point_rewriter.rewrite(initial_point_fgraph)

    ip_variables = initial_point_fgraph.outputs.copy()
    free_rvs_clone = [ip.owner.inputs[0] for ip in ip_variables]
    n_rvs = len(free_rvs_clone)

    initial_values = []
    initial_values_transformed = []
    for original_variable, variable in zip(free_rvs, free_rvs_clone):
        strategy = initval_strategies.get(original_variable)

        if strategy is None:
            strategy = default_strategy

        if isinstance(strategy, str):
            if strategy == "support_point":
                try:
                    value = support_point(variable)

                    # If a support point expression depends on other free_RVs that are not
                    # wrapped in InitialPoint, we need to replace them with their wrapped versions
                    # This can only happen for multi-output distributions, where the initial point
                    # of some outputs depends on the initial point of other outputs from the same node.
                    other_free_rvs = set(free_rvs_clone) - {variable}
                    support_point_replacements = {
                        ancestor: ip_variables[free_rvs_clone.index(ancestor)]
                        for ancestor in non_support_point_ancestors(value)
                        if ancestor in other_free_rvs
                    }
                    if support_point_replacements:
                        value = graph_replace(value, support_point_replacements)

                except NotImplementedError:
                    warnings.warn(
                        f"Support point not defined for variable {variable} of type "
                        f"{variable.owner.op.__class__.__name__}, defaulting to "
                        f"a draw from the prior. This can lead to difficulties "
                        f"during tuning. You can manually define an initval or "
                        f"implement a support_point dispatched function for this "
                        f"distribution.",
                        UserWarning,
                    )
                    value = variable
            elif strategy == "prior":
                value = variable
            else:
                raise ValueError(
                    f'Invalid string strategy: {strategy}. It must be one of ["support_point", "prior"]'
                )
        else:
            if isinstance(strategy, Variable) and (set(free_rvs) & set(ancestors([strategy]))):
                raise ValueError(
                    f"Initial value of {original_variable} depends on other random variables. This is not supported anymore."
                )
            value = pt.as_tensor(strategy, variable.dtype).astype(variable.dtype)

        transform = rvs_to_transforms.get(original_variable, None)

        if transform is not None:
            value = transform.forward(value, *variable.owner.inputs)

        if original_variable in jitter_rvs:
            jitter = pt.random.uniform(-1, 1, size=value.shape)
            # Hack to allow xtensor value to be added to tensor jitter
            jitter = value.type.filter_variable(jitter)
            jitter.name = f"{variable.name}_jitter"
            # Hack to allow xtensor value to be added to tensor jitter
            value = value + jitter

        value = value.astype(variable.dtype)
        initial_values_transformed.append(value)

        if transform is not None:
            value = transform.backward(value, *variable.owner.inputs)

        initial_values.append(value)

    for initial_value in initial_values:
        # FIXME: This is a hack so that interdependent replacements that can't
        # be sorted topologically from the initial point graph come out correctly.
        # This happens for multi-output nodes where the replacements depend on each other.
        # From the original graph perspective, their ordering is equivalent.
        initial_point_fgraph.add_output(initial_value, import_missing=True)

    # We now replace all rvs by the respective initial_point expressions
    # in the constrained (untransformed) space. We do this in reverse topological
    # order, so that later nodes do not reintroduce expressions with earlier
    # rvs that would need to once again be replaced by their initial_points
    toposort_replace(initial_point_fgraph, tuple(zip(ip_variables, initial_values)), reverse=True)

    if not return_transformed:
        return initial_point_fgraph.outputs[:n_rvs]

    # Because the unconstrained (transformed) expressions are a subgraph of the
    # constrained initial point they were also automatically updated inplace
    # when calling graph.replace_all above, so we don't need to do anything else
    return initial_values_transformed
