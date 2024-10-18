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
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import arviz as az
import numcodecs
import numpy as np
import xarray as xr
import zarr

from pytensor.tensor.variable import TensorVariable
from zarr.storage import BaseStore
from zarr.sync import Synchronizer

from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata,
    find_constants,
    find_observations,
)
from pymc.backends.base import BaseTrace
from pymc.blocking import StatDtype, StatShape
from pymc.model.core import Model, modelcontext
from pymc.step_methods.compound import (
    BlockedStep,
    CompoundStep,
    StatsBijection,
    get_stats_dtypes_shapes_from_steps,
)
from pymc.util import get_default_varnames


class ZarrChain(BaseTrace):
    def __init__(
        self,
        store: BaseStore | MutableMapping,
        stats_bijection: StatsBijection,
        synchronizer: Synchronizer | None = None,
        model: Model | None = None,
        vars: Sequence[TensorVariable] | None = None,
        test_point: Sequence[dict[str, np.ndarray]] | None = None,
    ):
        super().__init__(name="zarr", model=model, vars=vars, test_point=test_point)
        self.draw_idx = 0
        self._posterior = zarr.open_group(
            store, synchronizer=synchronizer, path="posterior", mode="a"
        )
        self._sample_stats = zarr.open_group(
            store, synchronizer=synchronizer, path="sample_stats", mode="a"
        )
        self._sampling_state = zarr.open_group(
            store, synchronizer=synchronizer, path="_sampling_state", mode="a"
        )
        self.stats_bijection = stats_bijection

    def setup(self, draws: int, chain: int, sampler_vars: Sequence[dict] | None):  # type: ignore[override]
        self.chain = chain

    def record(self, draw: Mapping[str, np.ndarray], stats: Sequence[Mapping[str, Any]]):
        chain = self.chain
        draw_idx = self.draw_idx
        for var_name, var_value in zip(self.varnames, self.fn(draw)):
            self._posterior[var_name].set_orthogonal_selection(
                (chain, draw_idx),
                var_value,
            )
        for var_name, var_value in self.stats_bijection.map(stats).items():
            self._sample_stats[var_name].set_orthogonal_selection(
                (chain, draw_idx),
                var_value,
            )
        self.draw_idx += 1

    def record_sampling_state(self, step):
        self._sampling_state.sampling_state.set_coordinate_selection(
            self.chain, np.array([step.sampling_state], dtype="object")
        )
        self._sampling_state.draw_idx.set_coordinate_selection(self.chain, self.draw_idx)


FILL_VALUE_TYPE = float | int | bool | str | np.datetime64 | np.timedelta64 | None
DEFAULT_FILL_VALUES: dict[Any, FILL_VALUE_TYPE] = {
    np.floating: np.nan,
    np.integer: 0,
    np.bool_: False,
    np.str_: "",
    np.datetime64: np.datetime64(0, "Y"),
    np.timedelta64: np.timedelta64(0, "Y"),
}


def get_initial_fill_value_and_codec(
    dtype: Any,
) -> tuple[FILL_VALUE_TYPE, np.typing.DTypeLike, numcodecs.abc.Codec | None]:
    _dtype = np.dtype(dtype)
    fill_value: FILL_VALUE_TYPE = None
    codec = None
    try:
        fill_value = DEFAULT_FILL_VALUES[_dtype]
    except KeyError:
        for key in DEFAULT_FILL_VALUES:
            if np.issubdtype(_dtype, key):
                fill_value = DEFAULT_FILL_VALUES[key]
                break
        else:
            codec = numcodecs.Pickle()
    return fill_value, _dtype, codec


class ZarrTrace:
    def __init__(
        self,
        store: BaseStore | MutableMapping | None = None,
        synchronizer: Synchronizer | None = None,
        model: Model | None = None,
        vars: Sequence[TensorVariable] | None = None,
        include_transformed: bool = False,
        draws_per_chunk: int = 1,
    ):
        model = modelcontext(model)
        self.model = model

        self.synchronizer = synchronizer
        self.root = zarr.group(
            store=store,
            overwrite=True,
            synchronizer=synchronizer,
        )
        self.coords, self.vars_to_dims = coords_and_dims_for_inferencedata(model)
        self.draws_per_chunk = int(draws_per_chunk)
        assert self.draws_per_chunk >= 1

        if vars is None:
            vars = model.unobserved_value_vars

        unnamed_vars = {var for var in vars if var.name is None}
        assert not unnamed_vars, f"Can't trace unnamed variables: {unnamed_vars}"
        self.varnames = get_default_varnames(
            [var.name for var in vars], include_transformed=include_transformed
        )
        self.vars = [var for var in vars if var.name in self.varnames]

        self.fn = model.compile_fn(self.vars, inputs=model.value_vars, on_unused_input="ignore")

        # Get variable shapes. Most backends will need this
        # information.
        test_point = model.initial_point()
        var_values = list(zip(self.varnames, self.fn(test_point)))
        self.var_dtype_shapes = {var: (value.dtype, value.shape) for var, value in var_values}
        self._is_base_setup = False

    @property
    def posterior(self) -> zarr.Group:
        return self.root.posterior

    @property
    def sample_stats(self) -> zarr.Group:
        return self.root.sample_stats

    @property
    def constant_data(self) -> zarr.Group:
        return self.root.constant_data

    @property
    def observed_data(self) -> zarr.Group:
        return self.root.observed_data

    @property
    def _sampling_state(self) -> zarr.Group:
        return self.root._sampling_state

    def init_trace(self, chains: int, draws: int, step: BlockedStep | CompoundStep):
        self.create_group(
            name="constant_data",
            data_dict=find_constants(self.model),
        )

        self.create_group(
            name="observed_data",
            data_dict=find_observations(self.model),
        )

        self.init_group_with_empty(
            group=self.root.create_group(name="posterior", overwrite=True),
            var_dtype_and_shape=self.var_dtype_shapes,
            chains=chains,
            draws=draws,
        )
        stats_dtypes_shapes = get_stats_dtypes_shapes_from_steps(
            [step] if isinstance(step, BlockedStep) else step.methods
        )
        self.init_group_with_empty(
            group=self.root.create_group(name="sample_stats", overwrite=True),
            var_dtype_and_shape=stats_dtypes_shapes,
            chains=chains,
            draws=draws,
        )

        self.init_sampling_state_group(chains=chains)

        self.straces = [
            ZarrChain(
                store=self.root.store,
                synchronizer=self.synchronizer,
                model=self.model,
                vars=self.vars,
                test_point=None,
                stats_bijection=StatsBijection(step.stats_dtypes),
            )
        ]
        for chain, strace in enumerate(self.straces):
            strace.setup(draws=draws, chain=chain, sampler_vars=None)

    def consolidate(self):
        if not isinstance(self.root.store, zarr.storage.ConsolidatedMetadataStore):
            self.root = zarr.consolidate_metadata(self.root.store)

    def init_sampling_state_group(self, chains: int):
        state = self.root.create_group(name="_sampling_state", overwrite=True)
        sampling_state = state.empty(
            name="sampling_state",
            overwrite=True,
            shape=(chains,),
            chunks=(1,),
            dtype="object",
            object_codec=numcodecs.Pickle(),
        )
        sampling_state.attrs.update({"_ARRAY_DIMENSIONS": ["chain"]})
        draw_idx = state.array(
            name="draw_idx",
            overwrite=True,
            data=np.zeros(chains, dtype="int"),
            chunks=(1,),
            dtype="int",
            fill_value=-1,
        )
        draw_idx.attrs.update({"_ARRAY_DIMENSIONS": ["chain"]})
        chain = state.array(name="chain", data=np.arange(chains))
        chain.attrs.update({"_ARRAY_DIMENSIONS": ["chain"]})

    def init_group_with_empty(
        self,
        group: zarr.Group,
        var_dtype_and_shape: dict[str, tuple[StatDtype, StatShape]],
        chains: int,
        draws: int,
    ) -> zarr.Group:
        group_coords: dict[str, Any] = {"chain": range(chains), "draw": range(draws)}
        for name, (_dtype, shape) in var_dtype_and_shape.items():
            fill_value, dtype, object_codec = get_initial_fill_value_and_codec(_dtype)
            shape = shape or ()
            array = group.full(
                name=name,
                dtype=dtype,
                fill_value=fill_value,
                object_codec=object_codec,
                shape=(chains, draws, *shape),
                chunks=(1, self.draws_per_chunk, *shape),
            )
            try:
                dims = self.vars_to_dims[name]
                for dim in dims:
                    group_coords[dim] = self.coords[dim]
            except KeyError:
                dims = []
                for i, shape_i in enumerate(shape):
                    dim = f"{name}_dim_{i}"
                    dims.append(dim)
                    group_coords[dim] = np.arange(shape_i, dtype="int")
            dims = ("chain", "draw", *dims)
            array.attrs.update({"_ARRAY_DIMENSIONS": dims})
        for dim, coord in group_coords.items():
            array = group.array(name=dim, data=coord, fill_value=None)
            array.attrs.update({"_ARRAY_DIMENSIONS": [dim]})
        return group

    def create_group(self, name: str, data_dict: dict[str, np.ndarray]) -> zarr.Group | None:
        group: zarr.Group | None = None
        if data_dict:
            group_coords = {}
            group = self.root.create_group(name=name, overwrite=True)
            for var_name, var_value in data_dict.items():
                fill_value, dtype, object_codec = get_initial_fill_value_and_codec(var_value.dtype)
                array = group.array(
                    name=var_name,
                    data=var_value,
                    fill_value=fill_value,
                    dtype=dtype,
                    object_codec=object_codec,
                )
                try:
                    dims = self.vars_to_dims[var_name]
                    for dim in dims:
                        group_coords[dim] = self.coords[dim]
                except KeyError:
                    dims = []
                    for i in range(var_value.ndim):
                        dim = f"{var_name}_dim_{i}"
                        dims.append(dim)
                        group_coords[dim] = np.arange(var_value.shape[i], dtype="int")
                array.attrs.update({"_ARRAY_DIMENSIONS": dims})
            for dim, coord in group_coords.items():
                array = group.array(name=dim, data=coord, fill_value=None)
                array.attrs.update({"_ARRAY_DIMENSIONS": [dim]})
        return group

    def to_inferencedata(self) -> az.InferenceData:
        self.consolidate()
        # The ConsolidatedMetadataStore looks like an empty store from xarray's point of view
        # we need to actually grab the underlying store so that xarray doesn't produce completely
        # empty arrays
        store = self.root.store.store
        groups = {}
        for name, _ in self.root.groups():
            if name.startswith("_"):
                continue
            data = xr.open_zarr(store, group=name, mask_and_scale=False)
            groups[name] = data.load() if az.rcParams["data.load"] == "eager" else data
        return az.InferenceData(**groups)
