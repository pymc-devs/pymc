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

from abc import abstractmethod
from collections.abc import Callable
from typing import cast

import numpy as np

from pymc.blocking import DictToArrayBijection, PointType, RaveledVars, StatsType
from pymc.model import modelcontext
from pymc.step_methods.compound import BlockedStep
from pymc.util import RandomGenerator, get_random_generator, get_var_name

__all__ = ["ArrayStep", "ArrayStepShared", "metrop_select"]


class ArrayStep(BlockedStep):
    """
    Blocked step method that is generalized to accept vectors of variables.

    Parameters
    ----------
    vars: list
        List of value variables for sampler.
    fs: list of logp PyTensor functions
    allvars: Boolean (default False)
    blocked: Boolean (default True)
    rng: RandomGenerator
        An object that can produce be used to produce the step method's
        :py:class:`~numpy.random.Generator` object. Refer to
        :py:func:`pymc.util.get_random_generator` for more information.
    """

    def __init__(self, vars, fs, allvars=False, blocked=True, rng: RandomGenerator = None):
        self.vars = vars
        self.fs = fs
        self.allvars = allvars
        self.blocked = blocked
        self.rng = get_random_generator(rng)

    def step(self, point: PointType) -> tuple[PointType, StatsType]:
        partial_funcs_and_point: list[Callable | PointType] = [
            DictToArrayBijection.mapf(x, start_point=point) for x in self.fs
        ]
        if self.allvars:
            partial_funcs_and_point.append(point)

        var_dict = {cast(str, v.name): point[cast(str, v.name)] for v in self.vars}
        apoint = DictToArrayBijection.map(var_dict)
        apoint_new, stats = self.astep(apoint, *partial_funcs_and_point)

        if not isinstance(apoint_new, RaveledVars):
            # We assume that the mapping has stayed the same
            apoint_new = RaveledVars(apoint_new, apoint.point_map_info)

        point_new = DictToArrayBijection.rmap(apoint_new, start_point=point)

        return point_new, stats

    @abstractmethod
    def astep(self, apoint: RaveledVars, *args) -> tuple[RaveledVars, StatsType]:
        """Perform a single sample step in a raveled and concatenated parameter space."""


class ArrayStepShared(BlockedStep):
    """Faster version of ArrayStep.

    It requires the substep method that does not wrap the functions the step
    method uses.

    Works by setting shared variables before using the step. This eliminates the mapping
    and unmapping overhead as well as moving fewer variables around.
    """

    def __init__(self, vars, shared, blocked=True, rng: RandomGenerator = None):
        """
        Create the ArrayStepShared object.

        Parameters
        ----------
        vars: list of sampling value variables
        shared: dict of PyTensor variable -> shared variable
        blocked: Boolean (default True)
        rng: RandomGenerator
            An object that can produce be used to produce the step method's
            :py:class:`~numpy.random.Generator` object. Refer to
            :py:func:`pymc.util.get_random_generator` for more information.
        """
        self.vars = vars
        self.shared = {get_var_name(var): shared for var, shared in shared.items()}
        self.blocked = blocked
        self.rng = get_random_generator(rng)

    def step(self, point: PointType) -> tuple[PointType, StatsType]:
        for name, shared_var in self.shared.items():
            shared_var.set_value(point[name])

        var_dict = {cast(str, v.name): point[cast(str, v.name)] for v in self.vars}
        q = DictToArrayBijection.map(var_dict)

        apoint, stats = self.astep(q)

        if not isinstance(apoint, RaveledVars):
            # We assume that the mapping has stayed the same
            apoint = RaveledVars(apoint, q.point_map_info)

        new_point = DictToArrayBijection.rmap(apoint, start_point=point)

        return new_point, stats

    @abstractmethod
    def astep(self, q0: RaveledVars) -> tuple[RaveledVars, StatsType]:
        """Perform a single sample step in a raveled and concatenated parameter space."""


class PopulationArrayStepShared(ArrayStepShared):
    """Version of ArrayStepShared that allows samplers to access the states of other chains in the population.

    Works by linking a list of Points that is updated as the chains are iterated.
    """

    def __init__(self, vars, shared, blocked=True, rng: RandomGenerator = None):
        """
        Create the PopulationArrayStepShared object.

        Parameters
        ----------
        vars: list of sampling value variables
        shared: dict of PyTensor variable -> shared variable
        blocked: Boolean (default True)
        rng: RandomGenerator
            An object that can produce be used to produce the step method's
            :py:class:`~numpy.random.Generator` object. Refer to
            :py:func:`pymc.util.get_random_generator` for more information.
        """
        self.population = None
        self.this_chain = None
        self.other_chains: list[int] | None = None
        return super().__init__(vars, shared, blocked, rng=rng)

    def link_population(self, population, chain_index):
        """Links the sampler to the population.

        Parameters
        ----------
        population: list of Points. (The elements of this list must be
            replaced with current chain states in every iteration.)
        chain_index: int of the index of this sampler in the population
        """
        self.population = population
        self.this_chain = chain_index
        self.other_chains = [c for c in range(len(population)) if c != chain_index]
        if not len(self.other_chains) > 1:
            raise ValueError(
                f"Population is just {self.this_chain} + {self.other_chains}. "
                "This is too small and the error should have been raised earlier."
            )
        return


class GradientSharedStep(ArrayStepShared):
    def __init__(
        self,
        vars,
        model=None,
        blocked=True,
        dtype=None,
        logp_dlogp_func=None,
        rng: RandomGenerator = None,
        **pytensor_kwargs,
    ):
        model = modelcontext(model)

        if logp_dlogp_func is None:
            func = model.logp_dlogp_function(vars, dtype=dtype, **pytensor_kwargs)
        else:
            func = logp_dlogp_func

        self._logp_dlogp_func = func

        super().__init__(vars, func._extra_vars_shared, blocked, rng=rng)

    def step(self, point) -> tuple[PointType, StatsType]:
        self._logp_dlogp_func._extra_are_set = True
        return super().step(point)


def metrop_select(
    mr: np.ndarray, q: np.ndarray, q0: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, bool]:
    """Perform rejection/acceptance step for Metropolis class samplers.

    Returns the new sample q if a uniform random number is less than the
    metropolis acceptance rate (`mr`), and the old sample otherwise, along
    with a boolean indicating whether the sample was accepted.

    Parameters
    ----------
    mr: float, Metropolis acceptance rate
    q: proposed sample
    q0: current sample
    rng: numpy.random.Generator
        A random number generator object

    Returns
    -------
    q or q0
    """
    # Compare acceptance ratio to uniform random number
    # TODO XXX: This `uniform` is not given a model-specific RNG state, which
    # means that sampler runs that use it will not be reproducible.
    if np.isfinite(mr) and np.log(rng.uniform()) < mr:
        return q, True
    else:
        return q0, False
