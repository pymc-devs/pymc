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

from typing import List, Optional, Sequence, Union

import numpy as np

from typing_extensions import TypeAlias

from pymc.backends.base import BaseTrace, MultiTrace
from pymc.backends.ndarray import NDArray
from pymc.initial_point import PointType
from pymc.vartypes import discrete_types

ArrayLike: TypeAlias = Union[np.ndarray, List[float]]
PointList: TypeAlias = List[PointType]
Backend: TypeAlias = Union[BaseTrace, MultiTrace, NDArray]

RandomSeed = Optional[Union[int, Sequence[int], np.ndarray]]
RandomState = Union[RandomSeed, np.random.RandomState, np.random.Generator]


__all__ = ()


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
