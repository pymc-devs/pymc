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
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

import arviz as az
import numpy as np
import xarray as xr

from arviz.data.base import make_attrs
from arviz.data.inference_data import WARMUP_TAG
from pytensor.tensor.variable import TensorVariable

import pymc

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
from pymc.util import UNSET, _UnsetType, get_default_varnames, is_transformed_name

try:
    import numcodecs
    import zarr

    from numcodecs.abc import Codec
    from zarr import Group
    from zarr.storage import BaseStore, default_compressor
    from zarr.sync import Synchronizer

    _zarr_available = True
except ImportError:
    from typing import TYPE_CHECKING, TypeVar

    if not TYPE_CHECKING:
        Codec = TypeVar("Codec")
        Group = TypeVar("Group")
        BaseStore = TypeVar("BaseStore")
        Synchronizer = TypeVar("Synchronizer")
    _zarr_available = False


class ZarrChain(BaseTrace):
    """Interface object to interact with a single chain in a :class:`~.ZarrTrace`.

    Parameters
    ----------
    store : zarr.storage.BaseStore | collections.abc.MutableMapping
        The store object where the zarr groups and arrays will be stored and read from.
        This store must exist before creating a ``ZarrChain`` object. ``ZarrChain`` are
        only intended to be used as interfaces to the individual chains of
        :class:`~.ZarrTrace` objects. This means that the :class:`~.ZarrTrace` should
        be the one that creates the store that is then provided to a ``ZarrChain``.
    stats_bijection : pymc.step_methods.compound.StatsBijection
        An object that maps between a list of step method stats and a dictionary of
        said stats with the accompanying stepper index.
    synchronizer : zarr.sync.Synchronizer | None
        The synchronizer to use for the underlying zarr arrays.
    model : Model
        If None, the model is taken from the `with` context.
    vars : Sequence[TensorVariable] | None
        Sampling values will be stored for these variables. If None,
        `model.unobserved_RVs` is used.
    test_point : dict[str, numpy.ndarray] | None
        This is not used and is inherited from the signature of :class:`~.BaseTrace`,
        which uses it to determine the shape and dtype of `vars`.
    draws_per_chunk : int
        The number of draws that make up a chunk in the variable's posterior array.
        The interface only writes the samples to the store once a chunk is completely
        filled.
    """

    def __init__(
        self,
        store: BaseStore | MutableMapping,
        stats_bijection: StatsBijection,
        synchronizer: Synchronizer | None = None,
        model: Model | None = None,
        vars: Sequence[TensorVariable] | None = None,
        test_point: dict[str, np.ndarray] | None = None,
        draws_per_chunk: int = 1,
        fn: Callable | None = None,
    ):
        if not _zarr_available:
            raise RuntimeError("You must install zarr to be able to create ZarrChain instances")
        super().__init__(name="zarr", model=model, vars=vars, test_point=test_point, fn=fn)
        self._step_method: BlockedStep | CompoundStep | None = None
        self.unconstrained_variables = {
            var.name for var in self.vars if is_transformed_name(var.name)
        }
        self.draw_idx = 0
        self._buffers: dict[str, dict[str, list]] = {
            "posterior": {},
            "sample_stats": {},
        }
        self._buffered_draws = 0
        self.draws_per_chunk = int(draws_per_chunk)
        assert self.draws_per_chunk > 0
        self._posterior = zarr.open_group(
            store, synchronizer=synchronizer, path="posterior", mode="a"
        )
        if self.unconstrained_variables:
            self._unconstrained_posterior = zarr.open_group(
                store, synchronizer=synchronizer, path="unconstrained_posterior", mode="a"
            )
            self._buffers["unconstrained_posterior"] = {}
        self._sample_stats = zarr.open_group(
            store, synchronizer=synchronizer, path="sample_stats", mode="a"
        )
        self._sampling_state = zarr.open_group(
            store, synchronizer=synchronizer, path="_sampling_state", mode="a"
        )
        self.stats_bijection = stats_bijection

    def link_stepper(self, step_method: BlockedStep | CompoundStep):
        """Provide a reference to the step method used during sampling.

        This reference can be used to facilite writing the stepper's sampling state
        each time the samples are flushed into the storage.
        """
        self._step_method = step_method

    def setup(self, draws: int, chain: int, sampler_vars: Sequence[dict] | None):  # type: ignore[override]
        self.chain = chain
        self.total_draws = draws
        self.draws_until_flush = min([self.draws_per_chunk, draws - self.draw_idx])
        self.clear_buffers()

    def clear_buffers(self):
        for group in self._buffers:
            self._buffers[group] = {}
        self._buffered_draws = 0

    def buffer(self, group, var_name, value):
        buffer = self._buffers[group]
        if var_name not in buffer:
            buffer[var_name] = []
        buffer[var_name].append(value)

    def record(
        self, draw: Mapping[str, np.ndarray], stats: Sequence[Mapping[str, Any]]
    ) -> bool | None:
        """Record the step method's returned draw and stats.

        The draws and stats are first stored in an internal buffer. Once the buffer is
        filled, the samples and stats are written (flushed) onto the desired zarr store.

        Returns
        -------
        flushed : bool | None
            Returns ``True`` only if the data was written onto the desired zarr store.
            Any other time that the recorded draw and stats are written into the
            internal buffer, ``None`` is returned.

        See Also
        --------
        :meth:`~ZarrChain.flush`
        """
        unconstrained_variables = self.unconstrained_variables
        for var_name, var_value in zip(self.varnames, self.fn(**draw)):
            if var_name in unconstrained_variables:
                self.buffer(group="unconstrained_posterior", var_name=var_name, value=var_value)
            else:
                self.buffer(group="posterior", var_name=var_name, value=var_value)
        for var_name, var_value in self.stats_bijection.map(stats).items():
            self.buffer(group="sample_stats", var_name=var_name, value=var_value)
        self._buffered_draws += 1
        if self._buffered_draws == self.draws_until_flush:
            self.flush()
            return True
        return None

    def record_sampling_state(self, step: BlockedStep | CompoundStep | None = None):
        """Record the sampling state information to the store's ``_sampling_state`` group.

        The sampling state includes the number of draws taken so far (``draw_idx``) and
        the step method's ``sampling_state``.

        Parameters
        ----------
        step : BlockedStep | CompoundStep | None
            The step method from which to take the ``sampling_state``. If ``None``,
            the ``step`` is taken to be the step method that was linked to the
            ``ZarrChain`` when calling :meth:`~ZarrChain.link_stepper`. If this method was never
            called, no step method ``sampling_state`` information is stored in the
            chain.
        """
        if step is None:
            step = self._step_method
        if step is not None:
            self.store_sampling_state(step.sampling_state)
        self._sampling_state.draw_idx.set_coordinate_selection(self.chain, self.draw_idx)

    def store_sampling_state(self, sampling_state):
        self._sampling_state.sampling_state.set_coordinate_selection(
            self.chain, np.array([sampling_state], dtype="object")
        )

    def flush(self):
        """Write the data stored in the internal buffer to the desired zarr store.

        After writing the draws and stats returned by each step of the step method,
        the :meth:`~ZarrChain.record_sampling_state` is called, the internal buffer is cleared and
        the number of steps until the next flush is determined.
        """
        chain = self.chain
        draw_slice = slice(self.draw_idx, self.draw_idx + self.draws_until_flush)
        for group_name, buffer in self._buffers.items():
            group = getattr(self, f"_{group_name}")
            for var_name, var_value in buffer.items():
                group[var_name].set_orthogonal_selection(
                    (chain, draw_slice),
                    np.stack(var_value),
                )
        self.draw_idx += self.draws_until_flush
        self.record_sampling_state()
        self.clear_buffers()
        self.draws_until_flush = min([self.draws_per_chunk, self.total_draws - self.draw_idx])


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
) -> tuple[FILL_VALUE_TYPE, np.typing.DTypeLike, Codec | None]:
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
    """Object that stores and enables access to MCMC draws stored in a :class:`zarr.hierarchy.Group` objects.

    This class creats a zarr hierarchy to represent the sampling information which is
    intended to mimic :class:`arviz.InferenceData`. The hierarchy looks like this:

    | root
    | |--> constant_data
    | |--> observed_data
    | |--> posterior
    | |--> unconstrained_posterior
    | |--> sample_stats
    | |--> warmup_posterior
    | |--> warmup_unconstrained_posterior
    | |--> warmup_sample_stats
    | |--> _sampling_state

    The root group is created when the ``ZarrTrace`` object is initialized. The rest of
    the groups are created once :meth:`~ZarrChain.init_trace` is called with a few exceptions:
    unconstrained_posterior is only created if ``include_transformed = True``, and the
    groups prefixed with ``warmup_`` are created only after calling
    :meth:`~ZarrTrace.split_warmup_groups`.

    Since ``ZarrTrace`` objects are intended to be as close to
    :class:`arviz.InferenceData` objects as possible, the groups store the dimension
    and coordinate information following the `xarray zarr standard <https://xarray.pydata.org/en/v2023.11.0/internals/zarr-encoding-spec.html>`_.

    Parameters
    ----------
    store : zarr.storage.BaseStore | collections.abc.MutableMapping | None
        The store object where the zarr groups and arrays will be stored and read from.
        Any zarr compatible storage object works. Keep in mind that if ``None`` is
        provided, a :class:`zarr.storage.MemoryStore` will be used, which means that
        information won't be visible to other processes and won't persist after the
        ``ZarrTrace`` life-cycle ends. If you want to have persistent storage, please
        use one of the multiple disk backed zarr storage options, e.g.
        :class:`~zarr.storage.DirectoryStore` or :class:`~zarr.storage.ZipStore`.
    synchronizer : zarr.sync.Synchronizer | None
        The synchronizer to use for the underlying zarr arrays.
    compressor : numcodec.abc.Codec | None | pymc.util.UNSET
        The compressor to use for the underlying zarr arrays. If ``None``, no compressor
        is used. If ``UNSET``, zarr's default compressor is used.
    draws_per_chunk : int
        The number of draws that make up a chunk in the variable's posterior array.
        Each variable's array shape is set to ``(n_chains, n_draws, *rv_shape)``, but
        the chunks are set to ``(1, draws_per_chunk, *rv_shape)``. This means that each
        chain will have it's own chunk to read or write to, allowing for concurrent
        write operations of different chains not to interfere with each other, and that
        multiple draws can belong to the same chunk. The variable's core dimension
        however, will never be split across different chunks.
    include_transformed : bool
        If ``True``, the transformed, unconstrained value variables are included in the
        storage group.

    Notes
    -----
    ``ZarrTrace`` objects represent the storage information. If the underlying store
    persists on disk or over the network (e.g. with a :class:`zarr.storage.FSStore`)
    multiple process will be able to concurrently access the same storage and read or
    write to it.

    The intended division of labour is for ``ZarrTrace`` to handle the creation and
    management of the zarr group and storage objects and arrays, and for individual
    :class:`~.ZarrChain` objects to handle recording MCMC samples to the trace. This
    division was chosen to stay close to the existing `pymc.backends.base.MultiTrace`
    and `pymc.backends.ndarray.NDArray` way of working with the existing samplers.

    One extra feature of ``ZarrTrace`` is that it enables direct access to any array's
    metadata. ``ZarrTrace`` takes advantage of this to tag arrays as ``deterministic``
    or ``freeRV`` depending on what kind of variable they were in the defining model.

    See Also
    --------
    :class:`~pymc.backends.zarr.ZarrChain`
    """

    def __init__(
        self,
        store: BaseStore | MutableMapping | None = None,
        synchronizer: Synchronizer | None = None,
        compressor: Codec | None | _UnsetType = UNSET,
        draws_per_chunk: int = 1,
        include_transformed: bool = False,
    ):
        if not _zarr_available:
            raise RuntimeError("You must install zarr to be able to create ZarrTrace instances")
        self.synchronizer = synchronizer
        if compressor is UNSET:
            compressor = default_compressor
        self.compressor = compressor
        self.root = zarr.group(
            store=store,
            overwrite=True,
            synchronizer=synchronizer,
        )

        self.draws_per_chunk = int(draws_per_chunk)
        assert self.draws_per_chunk >= 1

        self.include_transformed = include_transformed

        self._is_base_setup = False

    def groups(self) -> list[str]:
        return [str(group_name) for group_name, _ in self.root.groups()]

    @property
    def posterior(self) -> Group:
        return self.root.posterior

    @property
    def unconstrained_posterior(self) -> Group:
        return self.root.unconstrained_posterior

    @property
    def sample_stats(self) -> Group:
        return self.root.sample_stats

    @property
    def constant_data(self) -> Group:
        return self.root.constant_data

    @property
    def observed_data(self) -> Group:
        return self.root.observed_data

    @property
    def _sampling_state(self) -> Group:
        return self.root._sampling_state

    def init_trace(
        self,
        chains: int,
        draws: int,
        tune: int,
        step: BlockedStep | CompoundStep,
        model: Model | None = None,
        vars: Sequence[TensorVariable] | None = None,
        test_point: dict[str, np.ndarray] | None = None,
    ):
        """Initialize the trace groups and arrays.

        This function creates and fills with default values the groups below the
        ``ZarrTrace.root`` group. It creates the ``constant_data``, ``observed_data``,
        ``posterior``, ``unconstrained_posterior`` (if ``include_transformed = True``),
        ``sample_stats``, and ``_sampling_state`` zarr groups, and all of the relevant
        arrays that must be stored there.

        Every array in the posterior and sample stats groups will have the
        (chains, tune + draws) batch dimensions to the left of the core dimensions of
        the model's random variable or the step method's stat shape. The warmup (tuning
        draws) and the posterior samples are split at a later stage, once
        :meth:`~ZarrTrace.split_warmup_groups` is called.

        After the creation if the zarr hierarchies, it initializes the list of
        :class:`~pymc.backends.zarr.Zarrchain` instances (one for each chain) under the
        ``straces`` attribute. These objects serve as the interface to record draws and
        samples generated by the step methods for each chain.

        Parameters
        ----------
        chains : int
            The number of chains to use to initialize the arrays.
        draws : int
            The number of posterior draws to use to initialize the arrays.
        tune : int
            The number of tuning steps to use to initialize the arrays.
        step : pymc.step_methods.compound.BlockedStep | pymc.step_methods.compound.CompoundStep
            The step method that will be used to generate the draws and stats.
        model : pymc.model.core.Model | None
            If None, the model is taken from the ``with`` context.
        vars : Sequence[TensorVariable] | None
            Sampling values will be stored for these variables. If ``None``,
            ``model.unobserved_RVs`` is used.
        test_point : dict[str, numpy.ndarray] | None
            This is not used and is a product of the inheritance of :class:`ZarrChain`
            from :class:`~.BaseTrace`, which uses it to determine the shape and dtype
            of `vars`.
        """
        if self._is_base_setup:
            raise RuntimeError("The ZarrTrace has already been initialized")  # pragma: no cover
        model = modelcontext(model)
        self.model = model
        self.coords, self.vars_to_dims = coords_and_dims_for_inferencedata(model)
        if vars is None:
            vars = model.unobserved_value_vars

        unnamed_vars = {var for var in vars if var.name is None}
        assert not unnamed_vars, f"Can't trace unnamed variables: {unnamed_vars}"
        self.varnames = get_default_varnames(
            [var.name for var in vars], include_transformed=self.include_transformed
        )
        self.vars = [var for var in vars if var.name in self.varnames]

        self.fn = model.compile_fn(
            self.vars,
            inputs=model.value_vars,
            on_unused_input="ignore",
            point_fn=False,
        )

        # Get variable shapes. Most backends will need this
        # information.
        if test_point is None:
            test_point = model.initial_point()
        var_values = list(zip(self.varnames, self.fn(**test_point)))
        self.var_dtype_shapes = {
            var: (value.dtype, value.shape)
            for var, value in var_values
            if not is_transformed_name(var)
        }
        extra_var_attrs = {
            var: {
                "kind": "freeRV"
                if is_transformed_name(var) or model[var] in model.free_RVs
                else "deterministic"
            }
            for var in self.var_dtype_shapes
        }
        self.unc_var_dtype_shapes = {
            var: (value.dtype, value.shape) for var, value in var_values if is_transformed_name(var)
        }
        extra_unc_var_attrs = {var: {"kind": "freeRV"} for var in self.unc_var_dtype_shapes}

        self.create_group(
            name="constant_data",
            data_dict=find_constants(self.model),
        )

        self.create_group(
            name="observed_data",
            data_dict=find_observations(self.model),
        )

        # Create the posterior that includes warmup draws
        self.init_group_with_empty(
            group=self.root.create_group(name="posterior", overwrite=True),
            var_dtype_and_shape=self.var_dtype_shapes,
            chains=chains,
            draws=tune + draws,
            extra_var_attrs=extra_var_attrs,
        )

        # Create the unconstrained posterior group that includes warmup draws
        if self.include_transformed and self.unc_var_dtype_shapes:
            self.init_group_with_empty(
                group=self.root.create_group(name="unconstrained_posterior", overwrite=True),
                var_dtype_and_shape=self.unc_var_dtype_shapes,
                chains=chains,
                draws=tune + draws,
                extra_var_attrs=extra_unc_var_attrs,
            )

        # Create the sample stats that include warmup draws
        stats_dtypes_shapes = get_stats_dtypes_shapes_from_steps(
            [step] if isinstance(step, BlockedStep) else step.methods
        )
        self.init_group_with_empty(
            group=self.root.create_group(name="sample_stats", overwrite=True),
            var_dtype_and_shape=stats_dtypes_shapes,
            chains=chains,
            draws=tune + draws,
        )

        self.init_sampling_state_group(tune=tune, chains=chains)

        self.straces = [
            ZarrChain(
                store=self.root.store,
                synchronizer=self.synchronizer,
                model=self.model,
                vars=self.vars,
                test_point=test_point,
                stats_bijection=StatsBijection(step.stats_dtypes),
                draws_per_chunk=self.draws_per_chunk,
                fn=self.fn,
            )
            for _ in range(chains)
        ]
        for chain, strace in enumerate(self.straces):
            strace.setup(draws=tune + draws, chain=chain, sampler_vars=None)

    def split_warmup_groups(self):
        """Split the warmup and standard groups.

        This method takes the entries in the arrays in the posterior, sample_stats
        and unconstrained_posterior that happened in the tuning phase and moves them
        into the warmup_ groups. If the ``warmup_posterior`` group already exists, then
        nothing is done.

        See Also
        --------
        :meth:`~ZarrTrace.split_warmup`
        """
        if "warmup_posterior" not in self.groups():
            self.split_warmup("posterior", error_if_already_split=False)
            self.split_warmup("sample_stats", error_if_already_split=False)
            try:
                self.split_warmup("unconstrained_posterior", error_if_already_split=False)
            except KeyError:
                pass

    @property
    def tuning_steps(self):
        try:
            return int(self._sampling_state.tuning_steps.get_basic_selection())
        except AttributeError:  # pragma: no cover
            raise ValueError(
                "ZarrTrace has not been initialized and there is no tuning step information available"
            )

    @property
    def sampling_time(self):
        try:
            return float(self._sampling_state.sampling_time.get_basic_selection())
        except AttributeError:  # pragma: no cover
            raise ValueError(
                "ZarrTrace has not been initialized and there is no sampling time information available"
            )

    @sampling_time.setter
    def sampling_time(self, value):
        self._sampling_state.sampling_time.set_basic_selection((), float(value))

    def init_sampling_state_group(self, tune: int, chains: int):
        state = self.root.create_group(name="_sampling_state", overwrite=True)
        sampling_state = state.empty(
            name="sampling_state",
            overwrite=True,
            shape=(chains,),
            chunks=(1,),
            dtype="object",
            object_codec=numcodecs.Pickle(),
            compressor=self.compressor,
        )
        sampling_state.attrs.update({"_ARRAY_DIMENSIONS": ["chain"]})
        draw_idx = state.array(
            name="draw_idx",
            overwrite=True,
            data=np.zeros(chains, dtype="int"),
            chunks=(1,),
            dtype="int",
            fill_value=-1,
            compressor=self.compressor,
        )
        draw_idx.attrs.update({"_ARRAY_DIMENSIONS": ["chain"]})

        state.array(
            name="tuning_steps",
            data=tune,
            overwrite=True,
            dtype="int",
            fill_value=0,
            compressor=self.compressor,
        )
        state.array(
            name="sampling_time",
            data=0.0,
            dtype="float",
            fill_value=0.0,
            compressor=self.compressor,
        )
        state.array(
            name="sampling_start_time",
            data=0.0,
            dtype="float",
            fill_value=0.0,
            compressor=self.compressor,
        )

        chain = state.array(
            name="chain",
            data=np.arange(chains),
            compressor=self.compressor,
        )

        chain.attrs.update({"_ARRAY_DIMENSIONS": ["chain"]})

        state.empty(
            name="global_warnings",
            dtype="object",
            object_codec=numcodecs.Pickle(),
            shape=(0,),
        )

    def init_group_with_empty(
        self,
        group: Group,
        var_dtype_and_shape: dict[str, tuple[StatDtype, StatShape]],
        chains: int,
        draws: int,
        extra_var_attrs: dict | None = None,
    ) -> Group:
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
                compressor=self.compressor,
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
            attrs = extra_var_attrs[name] if extra_var_attrs is not None else {}
            attrs.update({"_ARRAY_DIMENSIONS": dims})
            array.attrs.update(attrs)
        for dim, coord in group_coords.items():
            array = group.array(
                name=dim,
                data=coord,
                fill_value=None,
                compressor=self.compressor,
            )
            array.attrs.update({"_ARRAY_DIMENSIONS": [dim]})
        return group

    def create_group(self, name: str, data_dict: dict[str, np.ndarray]) -> Group | None:
        group: Group | None = None
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
                    compressor=self.compressor,
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
                array = group.array(
                    name=dim,
                    data=coord,
                    fill_value=None,
                    compressor=self.compressor,
                )
                array.attrs.update({"_ARRAY_DIMENSIONS": [dim]})
        return group

    def split_warmup(self, group_name: str, error_if_already_split: bool = True):
        """Split the arrays of a group into the warmup and regular groups.

        This function takes the first ``self.tuning_steps`` draws of supplied
        ``group_name`` and moves them into a new zarr group called
        ``f"warmup_{group_name}"``.

        Parameters
        ----------
        group_name : str
            The name of the group that should be split.
        error_if_already_split : bool
            If ``True`` and if the ``f"warmup_{group_name}"`` group already exists in
            the root hierarchy, a ``ValueError`` is raised. If this flag is ``False``
            but the warmup group already exists, the contents of that group are
            overwritten.
        """
        if error_if_already_split and f"{WARMUP_TAG}{group_name}" in {
            group_name for group_name, _ in self.root.groups()
        }:
            raise RuntimeError(f"Warmup data for {group_name} has already been split")
        posterior_group = self.root[group_name]
        tune = self.tuning_steps
        warmup_group = self.root.create_group(f"{WARMUP_TAG}{group_name}", overwrite=True)
        if tune == 0:
            try:
                self.root.pop(f"{WARMUP_TAG}{group_name}")
            except KeyError:
                pass
            return
        for name, array in posterior_group.arrays():
            array_attrs = array.attrs.asdict()
            if name == "draw":
                warmup_array = warmup_group.array(
                    name="draw",
                    data=np.arange(tune),
                    dtype="int",
                    compressor=self.compressor,
                )
                posterior_array = posterior_group.array(
                    name=name,
                    data=np.arange(len(array) - tune),
                    dtype="int",
                    overwrite=True,
                    compressor=self.compressor,
                )
                posterior_array.attrs.update(array_attrs)
            else:
                dims = array.attrs["_ARRAY_DIMENSIONS"]
                warmup_idx: slice | tuple[slice, slice]
                if len(dims) >= 2 and dims[:2] == ["chain", "draw"]:
                    must_overwrite_posterior = True
                    warmup_idx = (slice(None), slice(None, tune, None))
                    posterior_idx = (slice(None), slice(tune, None, None))
                else:
                    must_overwrite_posterior = False
                    warmup_idx = slice(None)
                fill_value, dtype, object_codec = get_initial_fill_value_and_codec(array.dtype)
                warmup_array = warmup_group.array(
                    name=name,
                    data=array[warmup_idx],
                    chunks=array.chunks,
                    dtype=dtype,
                    fill_value=fill_value,
                    object_codec=object_codec,
                    compressor=self.compressor,
                )
                if must_overwrite_posterior:
                    posterior_array = posterior_group.array(
                        name=name,
                        data=array[posterior_idx],
                        chunks=array.chunks,
                        dtype=dtype,
                        fill_value=fill_value,
                        object_codec=object_codec,
                        overwrite=True,
                        compressor=self.compressor,
                    )
                    posterior_array.attrs.update(array_attrs)
            warmup_array.attrs.update(array_attrs)

    def to_inferencedata(self, save_warmup: bool = False) -> az.InferenceData:
        """Convert ``ZarrTrace`` to :class:`~.arviz.InferenceData`.

        This converts all the groups in the ``ZarrTrace.root`` hierarchy into an
        ``InferenceData`` object. The only exception is that ``_sampling_state`` is
        excluded.

        Parameters
        ----------
        save_warmup : bool
            If ``True``, all of the warmup groups are stored in the inference data
            object.

        Notes
        -----
        ``xarray`` and in turn ``arviz`` require the zarr groups to have consolidated
        metadata. To achieve this, a new consolidated store is constructed by calling
        :func:`zarr.consolidate_metadata` on the root's store. This means that the
        returned ``InferenceData`` object will operate on a different storage unit
        than the calling ``ZarrTrace``, so future changes to the ``ZarrTrace`` won't be
        automatically reflected in the returned ``InferenceData`` object.
        """
        self.split_warmup_groups()
        # Xarray complains if we try to open a zarr hierarchy that doesn't have consolidated metadata
        consolidated_root = zarr.consolidate_metadata(self.root.store)
        # The ConsolidatedMetadataStore looks like an empty store from xarray's point of view
        # we need to actually grab the underlying store so that xarray doesn't produce completely
        # empty arrays
        store = consolidated_root.store.store
        groups = {}
        try:
            global_attrs = {
                "tuning_steps": self.tuning_steps,
                "sampling_time": self.sampling_time,
            }
        except AttributeError:
            global_attrs = {}  # pragma: no cover
        for name, _ in self.root.groups():
            if name.startswith("_") or (not save_warmup and name.startswith(WARMUP_TAG)):
                continue
            data = xr.open_zarr(store, group=name, mask_and_scale=False)
            attrs = {**data.attrs, **global_attrs}
            data.attrs = make_attrs(attrs=attrs, library=pymc)
            groups[name] = data.load() if az.rcParams["data.load"] == "eager" else data
        return az.InferenceData(**groups)
