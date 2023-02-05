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

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import hagelkorn
import mcbackend as mcb
import numpy as np

from mcbackend.npproto.utils import ndarray_from_numpy
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Constant

from pymc.backends.base import IBaseTrace
from pymc.model import Model
from pymc.step_methods.compound import (
    BlockedStep,
    CompoundStep,
    StatsBijection,
    flatten_steps,
)


def find_data(pmodel: Model) -> List[mcb.DataVariable]:
    """Extracts data variables from a model."""
    observed_rvs = {pmodel.rvs_to_values[rv] for rv in pmodel.observed_RVs}
    dvars = []
    # All data containers are named vars!
    for name, var in pmodel.named_vars.items():
        dv = mcb.DataVariable(name)
        if isinstance(var, Constant):
            dv.value = ndarray_from_numpy(var.data)
        elif isinstance(var, SharedVariable):
            dv.value = ndarray_from_numpy(var.get_value())
        else:
            continue
        dv.dims = list(pmodel.named_vars_to_dims.get(name, []))
        dv.is_observed = var in observed_rvs
        dvars.append(dv)
    return dvars


def make_runmeta(
    *,
    var_dtypes: Dict[str, np.dtype],
    var_shapes: Dict[str, Sequence[int]],
    step: Union[CompoundStep, BlockedStep],
    model: Model,
) -> mcb.RunMeta:
    """Create an McBackend metadata description for the MCMC run.

    Parameters
    ----------
    var_dtypes : dict
        Variable names and corresponding NumPy dtypes.
    var_shapes : dict
        Variable names and corresponding shape tuples.
    step : CompoundStep or BlockedStep
        The step method that iterates the MCMC.
    model : pm.Model
        The current PyMC model.

    Returns
    -------
    rmeta : mcb.RunMeta
        Metadata about the model and MCMC sampling run.
    """
    # Replace None with "" in RV dims.
    rv_dims = {
        name: ((dname or "") for dname in dims) for name, dims in model.named_vars_to_dims.items()
    }
    free_rv_names = [rv.name for rv in model.free_RVs]
    variables = [
        mcb.Variable(
            name,
            str(var_dtypes[name]),
            list(var_shapes[name]),
            dims=list(rv_dims[name]) if name in rv_dims else [],
            is_deterministic=(name not in free_rv_names),
        )
        for name in var_dtypes.keys()
    ]

    sample_stats = [
        mcb.Variable("tune", "bool"),
    ]

    # In PyMC the sampler stats are grouped by the sampler.
    # ⚠ PyMC currently does not inform backends about shapes/dims of sampler stats.
    steps = flatten_steps(step)
    for s, sm in enumerate(steps):
        for statstypes in sm.stats_dtypes:
            for statname, dtype in statstypes.items():
                sname = f"sampler_{s}__{statname}"
                svar = mcb.Variable(
                    name=sname,
                    dtype=np.dtype(dtype).name,
                    # This 👇 is needed until samplers provide shapes ahead of time.
                    undefined_ndim=True,
                )
                sample_stats.append(svar)

    coordinates = [
        mcb.Coordinate(dname, mcb.npproto.utils.ndarray_from_numpy(np.array(cvals)))
        for dname, cvals in model.coords.items()
        if cvals is not None
    ]
    meta = mcb.RunMeta(
        rid=hagelkorn.random(),
        variables=variables,
        coordinates=coordinates,
        sample_stats=sample_stats,
        data=find_data(model),
    )
    return meta


class ChainRecordAdapter(IBaseTrace):
    """Wraps an McBackend ``Chain`` as an ``IBaseTrace``."""

    def __init__(self, chain: mcb.Chain, stats_bijection: StatsBijection) -> None:
        # Assign attributes required by IBaseTrace
        self.chain = chain.cmeta.chain_number
        self.varnames = [v.name for v in chain.rmeta.variables]
        stats_dtypes = {s.name: np.dtype(s.dtype) for s in chain.rmeta.sample_stats}
        self.sampler_vars = [
            {stepname: stats_dtypes[flatname] for flatname, stepname in sstats}
            for sstats in stats_bijection._stat_groups
        ]

        self._chain = chain
        self._statsbj = stats_bijection
        super().__init__()

    def record(self, draw: Mapping[str, np.ndarray], stats: Sequence[Mapping[str, Any]]):
        return self._chain.append(draw, self._statsbj.map(stats))

    def __len__(self):
        return len(self._chain)

    def get_values(self, varname: str, burn=0, thin=1) -> np.ndarray:
        return self._chain.get_draws(varname, slice(0, None, thin))

    def get_sampler_stats(
        self, stat_name: str, sampler_idx: Optional[int] = None, burn=0, thin=1
    ) -> np.ndarray:
        # Fetching for a specific sampler is easy
        if sampler_idx is not None:
            return self._chain.get_stats(
                f"sampler_{sampler_idx}__{stat_name}", slice(0, None, thin)
            )
        # To fetch for all samplers, we must collect the arrays one by one.
        stats_dict = {
            stat.name: self._chain.get_stats(stat.name, slice(0, None, thin))
            for stat in self._chain.rmeta.sample_stats
            if stat_name in stat.name
        }
        if not stats_dict:
            raise KeyError(f"No stat '{stat_name}' was recorded.")
        stats_list = self._statsbj.rmap(stats_dict)
        stats_arrays = []
        for sd in stats_list:
            if not sd:
                stats_arrays.append(np.empty((), dtype=object))
            else:
                stats_arrays.append(tuple(sd.values())[0])
        if sampler_idx is not None:
            return stats_arrays[sampler_idx]
        return np.array(stats_arrays).T

    def _slice(self, idx: slice) -> "IBaseTrace":
        # Get the integer indices
        start, stop, step = idx.indices(len(self))
        indices = np.arange(start, stop, step)

        # Create a NumPyChain for the sliced data
        nchain = mcb.backends.numpy.NumPyChain(
            self._chain.cmeta, self._chain.rmeta, preallocate=len(indices)
        )

        # Copy at selected indices and append them to the new chain.
        # This may be slow, but NumPyChain currently don't have a batch-insert or slice API.
        vnames = [v.name for v in nchain.variables.values()]
        snames = [s.name for s in nchain.sample_stats.values()]
        for i in indices:
            draw = self._chain.get_draws_at(i, var_names=vnames)
            stats = self._chain.get_stats_at(i, stat_names=snames)
            nchain.append(draw, stats)
        return ChainRecordAdapter(nchain, self._statsbj)

    def point(self, idx: int) -> Dict[str, np.ndarray]:
        return self._chain.get_draws_at(idx, [v.name for v in self._chain.variables.values()])
