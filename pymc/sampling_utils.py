#   Copyright 2022 The PyMC Developers
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

"""Helper functions for MCMC, prior and posterior predictive sampling."""

import logging

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np

from aesara import tensor as at
from aesara.graph.basic import Apply, Constant, Variable, general_toposort, walk
from aesara.graph.fg import FunctionGraph
from aesara.tensor.random.var import (
    RandomGeneratorSharedVariable,
    RandomStateSharedVariable,
)
from aesara.tensor.sharedvar import SharedVariable
from typing_extensions import TypeAlias

from pymc.aesaraf import compile_pymc
from pymc.backends.base import BaseTrace, MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.initial_point import PointType
from pymc.vartypes import discrete_types

ArrayLike: TypeAlias = Union[np.ndarray, List[float]]
PointList: TypeAlias = List[PointType]
Backend: TypeAlias = Union[BaseTrace, MultiTrace, NDArray]

RandomSeed = Optional[Union[int, Sequence[int], np.ndarray]]
RandomState = Union[RandomSeed, np.random.RandomState, np.random.Generator]

_log = logging.getLogger("pymc")


__all__ = (
    "compile_forward_sampling_function",
    "draw",
)


def all_continuous(vars):
    """Check that vars not include discrete variables, excepting observed RVs."""

    vars_ = [var for var in vars if not hasattr(var.tag, "observations")]

    if any([(var.dtype in discrete_types) for var in vars_]):
        return False
    else:
        return True


def _get_seeds_per_chain(
    random_state: RandomState,
    chains: int,
) -> Union[Sequence[int], np.ndarray]:
    """Obtain or validate specified integer seeds per chain.

    This function process different possible sources of seeding and returns one integer
    seed per chain:
    1. If the input is an integer and a single chain is requested, the input is
        returned inside a tuple.
    2. If the input is a sequence or NumPy array with as many entries as chains,
        the input is returned.
    3. If the input is an integer and multiple chains are requested, new unique seeds
        are generated from NumPy default Generator seeded with that integer.
    4. If the input is None new unique seeds are generated from an unseeded NumPy default
        Generator.
    5. If a RandomState or Generator is provided, new unique seeds are generated from it.

    Raises
    ------
    ValueError
        If none of the conditions above are met
    """

    def _get_unique_seeds_per_chain(integers_fn):
        seeds = []
        while len(set(seeds)) != chains:
            seeds = [int(seed) for seed in integers_fn(2**30, dtype=np.int64, size=chains)]
        return seeds

    if random_state is None or isinstance(random_state, int):
        if chains == 1 and isinstance(random_state, int):
            return (random_state,)
        return _get_unique_seeds_per_chain(np.random.default_rng(random_state).integers)
    if isinstance(random_state, np.random.Generator):
        return _get_unique_seeds_per_chain(random_state.integers)
    if isinstance(random_state, np.random.RandomState):
        return _get_unique_seeds_per_chain(random_state.randint)

    if not isinstance(random_state, (list, tuple, np.ndarray)):
        raise ValueError(f"The `seeds` must be array-like. Got {type(random_state)} instead.")

    if len(random_state) != chains:
        raise ValueError(
            f"Number of seeds ({len(random_state)}) does not match the number of chains ({chains})."
        )

    return random_state


def get_vars_in_point_list(trace, model):
    """Get the list of Variable instances in the model that have values stored in the trace."""
    if not isinstance(trace, MultiTrace):
        names_in_trace = list(trace[0])
    else:
        names_in_trace = trace.varnames
    vars_in_trace = [model[v] for v in names_in_trace if v in model]
    return vars_in_trace


def compile_forward_sampling_function(
    outputs: List[Variable],
    vars_in_trace: List[Variable],
    basic_rvs: Optional[List[Variable]] = None,
    givens_dict: Optional[Dict[Variable, Any]] = None,
    constant_data: Optional[Dict[str, np.ndarray]] = None,
    constant_coords: Optional[Set[str]] = None,
    **kwargs,
) -> Tuple[Callable[..., Union[np.ndarray, List[np.ndarray]]], Set[Variable]]:
    """Compile a function to draw samples, conditioned on the values of some variables.

    The goal of this function is to walk the aesara computational graph from the list
    of output nodes down to the root nodes, and then compile a function that will produce
    values for these output nodes. The compiled function will take as inputs the subset of
    variables in the ``vars_in_trace`` that are deemed to not be **volatile**.

    Volatile variables are variables whose values could change between runs of the
    compiled function or after inference has been run. These variables are:

    - Variables in the outputs list
    - ``SharedVariable`` instances that are not ``RandomStateSharedVariable`` or ``RandomGeneratorSharedVariable``, and whose values changed with respect to what they were at inference time
    - Variables that are in the `basic_rvs` list but not in the ``vars_in_trace`` list
    - Variables that are keys in the ``givens_dict``
    - Variables that have volatile inputs

    Concretely, this function can be used to compile a function to sample from the
    posterior predictive distribution of a model that has variables that are conditioned
    on ``MutableData`` instances. The variables that depend on the mutable data that have changed
    will be considered volatile, and as such, they wont be included as inputs into the compiled
    function. This means that if they have values stored in the posterior, these values will be
    ignored and new values will be computed (in the case of deterministics and potentials) or
    sampled (in the case of random variables).

    This function also enables a way to impute values for any variable in the computational
    graph that produces the desired outputs: the ``givens_dict``. This dictionary can be used
    to set the ``givens`` argument of the aesara function compilation. This will essentially
    replace a node in the computational graph with any other expression that has the same
    type as the desired node. Passing variables in the givens_dict is considered an intervention
    that might lead to different variable values from those that could have been seen during
    inference, as such, **any variable that is passed in the ``givens_dict`` will be considered
    volatile**.

    Parameters
    ----------
    outputs : List[aesara.graph.basic.Variable]
        The list of variables that will be returned by the compiled function
    vars_in_trace : List[aesara.graph.basic.Variable]
        The list of variables that are assumed to have values stored in the trace
    basic_rvs : Optional[List[aesara.graph.basic.Variable]]
        A list of random variables that are defined in the model. This list (which could be the
        output of ``model.basic_RVs``) should have a reference to the variables that should
        be considered as random variable instances. This includes variables that have
        a ``RandomVariable`` owner op, but also unpure random variables like Mixtures, or
        Censored distributions.
    givens_dict : Optional[Dict[aesara.graph.basic.Variable, Any]]
        A dictionary that maps tensor variables to the values that should be used to replace them
        in the compiled function. The types of the key and value should match or an error will be
        raised during compilation.
    constant_data : Optional[Dict[str, numpy.ndarray]]
        A dictionary that maps the names of ``MutableData`` or ``ConstantData`` instances to their
        corresponding values at inference time. If a model was created with ``MutableData``, these
        are stored as ``SharedVariable`` with the name of the data variable and a value equal to
        the initial data. At inference time, this information is stored in ``InferenceData``
        objects under the ``constant_data`` group, which allows us to check whether a
        ``SharedVariable`` instance changed its values after inference or not. If the values have
        changed, then the ``SharedVariable`` is assumed to be volatile. If it has not changed, then
        the ``SharedVariable`` is assumed to not be volatile. If a ``SharedVariable`` is not found
        in either ``constant_data`` or ``constant_coords``, then it is assumed to be volatile.
        Setting ``constant_data`` to ``None`` is equivalent to passing an empty dictionary.
    constant_coords : Optional[Set[str]]
        A set with the names of the mutable coordinates that have not changed their shape after
        inference. If a model was created with mutable coordinates, these are stored as
        ``SharedVariable`` with the name of the coordinate and a value equal to the length of said
        coordinate. This set let's us check if a ``SharedVariable`` is a mutated coordinate, in
        which case, it is considered volatile. If a ``SharedVariable`` is not found
        in either ``constant_data`` or ``constant_coords``, then it is assumed to be volatile.
        Setting ``constant_coords`` to ``None`` is equivalent to passing an empty set.

    Returns
    -------
    function: Callable
        Compiled forward sampling Aesara function
    volatile_basic_rvs: Set of Variable
        Set of all basic_rvs that were considered volatile and will be resampled when
        the function is evaluated
    """
    if givens_dict is None:
        givens_dict = {}

    if basic_rvs is None:
        basic_rvs = []

    if constant_data is None:
        constant_data = {}
    if constant_coords is None:
        constant_coords = set()

    # We define a helper function to check if shared values match to an array
    def shared_value_matches(var):
        try:
            old_array_value = constant_data[var.name]
        except KeyError:
            return var.name in constant_coords
        current_shared_value = var.get_value(borrow=True)
        return np.array_equal(old_array_value, current_shared_value)

    # We need a function graph to walk the clients and propagate the volatile property
    fg = FunctionGraph(outputs=outputs, clone=False)

    # Walk the graph from inputs to outputs and tag the volatile variables
    nodes: List[Variable] = general_toposort(
        fg.outputs, deps=lambda x: x.owner.inputs if x.owner else []
    )
    volatile_nodes: Set[Any] = set()
    for node in nodes:
        if (
            node in fg.outputs
            or node in givens_dict
            or (  # SharedVariables, except RandomState/Generators
                isinstance(node, SharedVariable)
                and not isinstance(node, (RandomStateSharedVariable, RandomGeneratorSharedVariable))
                and not shared_value_matches(node)
            )
            or (  # Basic RVs that are not in the trace
                node in basic_rvs and node not in vars_in_trace
            )
            or (  # Variables that have any volatile input
                node.owner and any(inp in volatile_nodes for inp in node.owner.inputs)
            )
        ):
            volatile_nodes.add(node)

    # Collect the function inputs by walking the graph from the outputs. Inputs will be:
    # 1. Random variables that are not volatile
    # 2. Variables that have no owner and are not constant or shared
    inputs = []

    def expand(node):
        if (
            (
                node.owner is None and not isinstance(node, (Constant, SharedVariable))
            )  # Variables without owners that are not constant or shared
            or node in vars_in_trace  # Variables in the trace
        ) and node not in volatile_nodes:
            # This test will include variables without owners, and that are not constant
            # or shared, because these nodes will never be considered volatile
            inputs.append(node)
        if node.owner:
            return node.owner.inputs

    # walk produces a generator, so we have to actually exhaust the generator in a list to walk
    # the entire graph
    list(walk(fg.outputs, expand))

    # Populate the givens list
    givens = [
        (
            node,
            value
            if isinstance(value, (Variable, Apply))
            else at.constant(value, dtype=getattr(node, "dtype", None), name=node.name),
        )
        for node, value in givens_dict.items()
    ]

    return (
        compile_pymc(inputs, fg.outputs, givens=givens, on_unused_input="ignore", **kwargs),
        set(basic_rvs) & (volatile_nodes - set(givens_dict)),  # Basic RVs that will be resampled
    )


def draw(
    vars: Union[Variable, Sequence[Variable]],
    draws: int = 1,
    random_seed: RandomState = None,
    **kwargs,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Draw samples for one variable or a list of variables

    Parameters
    ----------
    vars : TensorVariable or iterable of TensorVariable
        A variable or a list of variables for which to draw samples.
    draws : int, default 1
        Number of samples needed to draw.
    random_seed : int, RandomState or numpy_Generator, optional
        Seed for the random number generator.
    **kwargs : dict, optional
        Keyword arguments for :func:`pymc.aesaraf.compile_pymc`.

    Returns
    -------
    list of ndarray
        A list of numpy arrays.

    Examples
    --------
        .. code-block:: python

            import pymc as pm

            # Draw samples for one variable
            with pm.Model():
                x = pm.Normal("x")
            x_draws = pm.draw(x, draws=100)
            print(x_draws.shape)

            # Draw 1000 samples for several variables
            with pm.Model():
                x = pm.Normal("x")
                y = pm.Normal("y", shape=10)
                z = pm.Uniform("z", shape=5)
            num_draws = 1000
            # Draw samples of a list variables
            draws = pm.draw([x, y, z], draws=num_draws)
            assert draws[0].shape == (num_draws,)
            assert draws[1].shape == (num_draws, 10)
            assert draws[2].shape == (num_draws, 5)
    """
    if random_seed is not None:
        (random_seed,) = _get_seeds_per_chain(random_seed, 1)

    draw_fn = compile_pymc(inputs=[], outputs=vars, random_seed=random_seed, **kwargs)

    if draws == 1:
        return draw_fn()

    # Single variable output
    if not isinstance(vars, (list, tuple)):
        cast(Callable[[], np.ndarray], draw_fn)
        return np.stack([draw_fn() for _ in range(draws)])

    # Multiple variable output
    cast(Callable[[], List[np.ndarray]], draw_fn)
    drawn_values = zip(*(draw_fn() for _ in range(draws)))
    return [np.stack(v) for v in drawn_values]
