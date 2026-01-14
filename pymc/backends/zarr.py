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
import itertools
import re

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any, cast

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
    CompoundStepState,
    StatsBijection,
    StepMethodState,
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


class TraceAlreadyInitialized(RuntimeError): ...


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
        include_transformed: bool = True,
    ):
        if not _zarr_available:
            raise RuntimeError("You must install zarr to be able to create ZarrChain instances")
        super().__init__(name="zarr", model=model, vars=vars, test_point=test_point, fn=fn)
        self.include_transformed = include_transformed
        self._step_method: BlockedStep | CompoundStep | None = None
        self.unconstrained_variables = {
            var.name for var in self.vars if is_transformed_name(var.name)
        } | {inp.name for inp in self.fn.maker.inputs if is_transformed_name(inp.name)}
        self.draw_idx = 0
        self._buffers: dict[str, dict[str, list]] = {
            "posterior": {},
            "sample_stats": {},
            "warmup_posterior": {},
            "warmup_sample_stats": {},
        }
        self._buffered_draws = 0
        self.draws_per_chunk = int(draws_per_chunk)
        assert self.draws_per_chunk > 0
        self._warmup_posterior = zarr.open_group(
            store, synchronizer=synchronizer, path="warmup_posterior", mode="a"
        )
        self._posterior = zarr.open_group(
            store, synchronizer=synchronizer, path="posterior", mode="a"
        )
        if self.unconstrained_variables and include_transformed:
            self._warmup_unconstrained_posterior = zarr.open_group(
                store, synchronizer=synchronizer, path="warmup_unconstrained_posterior", mode="a"
            )
            self._unconstrained_posterior = zarr.open_group(
                store, synchronizer=synchronizer, path="unconstrained_posterior", mode="a"
            )
            self._buffers["unconstrained_posterior"] = {}
            self._buffers["warmup_unconstrained_posterior"] = {}
        self._warmup_sample_stats = zarr.open_group(
            store, synchronizer=synchronizer, path="warmup_sample_stats", mode="a"
        )
        self._sample_stats = zarr.open_group(
            store, synchronizer=synchronizer, path="sample_stats", mode="a"
        )
        self._sampling_state: Group = zarr.open_group(
            store, synchronizer=synchronizer, path="_sampling_state", mode="a"
        )
        self.stats_bijection = stats_bijection

    def setup(self, draws: int, chain: int, sampler_vars: Sequence[dict] | None, tune: int = 0):  # type: ignore[override]
        self.chain = chain
        self.draws = draws
        self.tune = tune
        self.draw_idx = self._sampling_state.draw_idx[chain] or 0
        self.update_draws_until_flush()
        self.clear_buffers()

    def clear_buffers(self):
        for group in self._buffers:
            self._buffers[group] = {}
        self._buffered_draws = 0

    def buffer(self, group, var_name, value):
        group_name = f"warmup_{group}" if self.in_warmup else group
        buffer = self._buffers[group_name]
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
        include_transformed = self.include_transformed
        for var_name, var_value in zip(self.varnames, self.fn(**draw)):
            if var_name in unconstrained_variables:
                if include_transformed:
                    self.buffer(
                        group="unconstrained_posterior",
                        var_name=var_name,
                        value=var_value,
                    )
            else:
                self.buffer(group="posterior", var_name=var_name, value=var_value)
        for var_name, var_value in self.stats_bijection.map(stats).items():
            self.buffer(group="sample_stats", var_name=var_name, value=var_value)
        self._buffered_draws += 1
        if self._buffered_draws == self.draws_until_flush:
            self.flush(draw)
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

    def store_sampling_state(self, sampling_state: StepMethodState | CompoundStepState):
        self._sampling_state.sampling_state.set_coordinate_selection(
            self.chain, np.array([sampling_state], dtype="object")
        )

    def flush(self, mcmc_point: Mapping[str, np.ndarray] | None = None):
        """Write the data stored in the internal buffer to the desired zarr store.

        After writing the draws and stats returned by each step of the step method,
        the :meth:`~ZarrChain.record_sampling_state` is called, the internal buffer is cleared and
        the number of steps until the next flush is determined.
        """
        chain = self.chain
        offset = 0 if self.in_warmup else self.tune
        draw_slice = slice(self.draw_idx - offset, self.draw_idx + self.draws_until_flush - offset)
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
        self.update_draws_until_flush()
        if mcmc_point is not None:
            self.set_mcmc_point(mcmc_point)

    def update_draws_until_flush(self):
        self.in_warmup = self.draw_idx < self.tune
        self.draws_until_flush = min(
            [
                self.draws_per_chunk,
                self.tune - self.draw_idx
                if self.in_warmup
                else self.draws + self.tune - self.draw_idx,
            ]
        )

    def completed_draws_and_divergences(self, chain_specific: bool = True) -> tuple[int, int]:
        """Get number of completed draws and divergences in the traces.

        This is a helper function to start the ProgressBarManager when resuming sampling
        from an existing trace.

        Parameters
        ----------
        chain_specific : bool
            If ``True``, only the completed draws and divergences on the current chain
            are returned. If ``False``, the draws and divergences across all chains are
            returned

        Returns
        -------
        draws : int
            Number of draws in the current chain or across all chains.
        divergences : int
            Number of divergences in the current chain or across all chains.
        """
        # No need to iterate over ZarrChain instances because the zarr group is
        # shared between them
        idx: int | slice
        if chain_specific:
            idx = self.chain
        else:
            idx = slice(None)
        diverging_stat_sums = [
            np.sum(array[idx])
            for stat_name, array in self._sample_stats.arrays()
            if "diverging" in stat_name
        ]
        return int(np.sum(self._sampling_state.draw_idx[idx])), int(sum(diverging_stat_sums))

    def get_stored_draw_and_state(self) -> tuple[int, StepMethodState | CompoundStepState | None]:
        chain = getattr(self, "chain", None)
        draw_idx = 0
        sampling_state: StepMethodState | None = None
        if chain is not None:
            draw_idx = self._sampling_state.draw_idx[chain]
            sampling_state = cast(StepMethodState, self._sampling_state.sampling_state[chain])
        return draw_idx, sampling_state

    def set_mcmc_point(self, mcmc_point: Mapping[str, np.ndarray]):
        for var_name, value in mcmc_point.items():
            self._sampling_state.mcmc_point[var_name].set_basic_selection(
                self.chain,
                value,
            )

    def get_mcmc_point(self) -> dict[str, np.ndarray]:
        return {
            str(var_name): np.asarray(array[self.chain])
            for var_name, array in self._sampling_state.mcmc_point.arrays()
        }


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
    the groups are created once :meth:`~ZarrChain.init_trace` is called with the exception
    that the unconstrained_posterior and warmup_unconstrained_posterior groups are
    only created if ``include_transformed = True``.

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

    @property
    def is_root_populated(self) -> bool:
        groups = set(self.root.group_keys())
        out = groups >= {
            "posterior",
            "sample_stats",
            "warmup_posterior",
            "warmup_sample_stats",
            "_sampling_state",
        }
        if self.include_transformed and any(
            is_transformed_name(name) for name in getattr(self, "varnames", [])
        ):
            out &= groups >= {"unconstrained_posterior", "warmup_unconstrained_posterior"}
        return out

    @property
    def _is_base_setup(self) -> bool:
        return self.is_root_populated and getattr(self, "straces", 0) > 0

    def groups(self) -> list[str]:
        return [str(group_name) for group_name, _ in self.root.groups()]

    @property
    def posterior(self) -> Group:
        return self.root.posterior

    @property
    def warmup_posterior(self) -> Group:
        return self.root.warmup_posterior

    @property
    def unconstrained_posterior(self) -> Group:
        return self.root.unconstrained_posterior

    @property
    def warmup_unconstrained_posterior(self) -> Group:
        return self.root.warmup_unconstrained_posterior

    @property
    def sample_stats(self) -> Group:
        return self.root.sample_stats

    @property
    def warmup_sample_stats(self) -> Group:
        return self.root.warmup_sample_stats

    @property
    def constant_data(self) -> Group:
        return self.root.constant_data

    @property
    def observed_data(self) -> Group:
        return self.root.observed_data

    @property
    def _sampling_state(self) -> Group:
        return self.root._sampling_state

    def parse_varnames(
        self,
        model: Model | None = None,
        vars: Sequence[TensorVariable] | None = None,
    ) -> tuple[list[TensorVariable], list[str]]:
        if vars is None:
            vars = modelcontext(model).unobserved_value_vars

        unnamed_vars = {var for var in vars if var.name is None}
        assert not unnamed_vars, f"Can't trace unnamed variables: {unnamed_vars}"
        var_names = get_default_varnames(
            [var.name for var in vars], include_transformed=self.include_transformed
        )
        vars = [var for var in vars if var.name in var_names]
        return vars, var_names

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
        ``ZarrTrace.root`` group. It creates the ``constant_data`` (only if the model
        has ``Data`` containers), ``observed_data`` (only if the model has observed),
        ``posterior``, ``unconstrained_posterior`` (if ``include_transformed = True``),
        ``sample_stats``, and ``_sampling_state`` zarr groups, and all of the relevant
        arrays that must be stored there.

        Every array in the posterior and sample stats groups will have the
        (chains, draws) batch dimensions to the left of the core dimensions of
        the model's random variable or the step method's stat shape. The warmup (tuning
        draws) posterior and sample stats will have (chains, tune) batch dimensions
        instead.

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
        if self.is_root_populated:
            raise TraceAlreadyInitialized("The ZarrTrace has already been initialized")
        model = modelcontext(model)
        self.coords, self.vars_to_dims = coords_and_dims_for_inferencedata(model)
        vars, varnames = self.parse_varnames(model, vars)

        fn = model.compile_fn(
            vars,
            inputs=model.value_vars,
            on_unused_input="ignore",
            point_fn=False,
        )

        # Get variable shapes. Most backends will need this
        # information.
        if test_point is None:
            test_point = model.initial_point()
        var_values = list(zip(varnames, fn(**test_point)))
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
            data_dict=find_constants(model),
        )

        self.create_group(
            name="observed_data",
            data_dict=find_observations(model),
        )

        # Create the posterior and warmup posterior groups
        self.init_group_with_empty(
            group=self.root.create_group(name="posterior", overwrite=True),
            var_dtype_and_shape=self.var_dtype_shapes,
            chains=chains,
            draws=draws,
            extra_var_attrs=extra_var_attrs,
        )
        self.init_group_with_empty(
            group=self.root.create_group(name="warmup_posterior", overwrite=True),
            var_dtype_and_shape=self.var_dtype_shapes,
            chains=chains,
            draws=tune,
            extra_var_attrs=extra_var_attrs,
        )

        # Create the unconstrained posterior and warmup groups
        if self.include_transformed and self.unc_var_dtype_shapes:
            self.init_group_with_empty(
                group=self.root.create_group(name="unconstrained_posterior", overwrite=True),
                var_dtype_and_shape=self.unc_var_dtype_shapes,
                chains=chains,
                draws=draws,
                extra_var_attrs=extra_unc_var_attrs,
            )
            self.init_group_with_empty(
                group=self.root.create_group(name="warmup_unconstrained_posterior", overwrite=True),
                var_dtype_and_shape=self.unc_var_dtype_shapes,
                chains=chains,
                draws=tune,
                extra_var_attrs=extra_unc_var_attrs,
            )

        # Create the sample stats and warmup groups
        stats_dtypes_shapes = get_stats_dtypes_shapes_from_steps(
            [step] if isinstance(step, BlockedStep) else step.methods
        )
        self.init_group_with_empty(
            group=self.root.create_group(name="sample_stats", overwrite=True),
            var_dtype_and_shape=stats_dtypes_shapes,
            chains=chains,
            draws=draws,
        )
        self.init_group_with_empty(
            group=self.root.create_group(name="warmup_sample_stats", overwrite=True),
            var_dtype_and_shape=stats_dtypes_shapes,
            chains=chains,
            draws=tune,
        )

        self.init_sampling_state_group(
            tune=tune,
            draws=draws,
            chains=chains,
            mcmc_point=test_point,
        )
        self.link_model_and_step(
            chains=chains,
            draws=draws,
            tune=tune,
            step=step,
            model=model,
            vars=vars,
            var_names=varnames,
            test_point=test_point,
            fn=fn,
        )

    def link_model_and_step(
        self,
        chains: int,
        draws: int,
        tune: int,
        step: BlockedStep | CompoundStep,
        vars: Sequence[TensorVariable],
        var_names: Sequence[str],
        model: Model | None = None,
        test_point: dict[str, np.ndarray] | None = None,
        fn: Callable | None = None,
    ):
        model = modelcontext(model)
        self.model = model
        self.varnames = var_names
        self.vars = vars
        if fn is None:
            self.fn = cast(
                Callable,
                model.compile_fn(
                    self.vars,
                    inputs=model.value_vars,
                    on_unused_input="ignore",
                    point_fn=False,
                ),
            )
        else:
            self.fn = fn
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
                include_transformed=self.include_transformed,
            )
            for _ in range(chains)
        ]
        for chain, strace in enumerate(self.straces):
            strace.setup(draws=draws, tune=tune, chain=chain, sampler_vars=None)

    @property
    def tuning_steps(self):
        try:
            return int(self._sampling_state.tuning_steps.get_basic_selection())
        except AttributeError:  # pragma: no cover
            raise ValueError(
                "ZarrTrace has not been initialized and there is no tuning step information available"
            )

    @property
    def draws(self):
        try:
            return int(self._sampling_state.draws.get_basic_selection())
        except AttributeError:  # pragma: no cover
            raise ValueError(
                "ZarrTrace has not been initialized and there is no draw information available"
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

    def init_sampling_state_group(
        self, tune: int, draws: int, chains: int, mcmc_point: dict[str, np.ndarray]
    ):
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
            name="draws",
            data=draws,
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

        zarr_mcmc_point = state.create_group("mcmc_point", overwrite=True)
        for var_name, test_value in mcmc_point.items():
            fill_value, dtype, object_codec = get_initial_fill_value_and_codec(test_value.dtype)
            zarr_mcmc_point.full(
                name=var_name,
                dtype=dtype,
                fill_value=fill_value,
                object_codec=object_codec,
                shape=(chains, *test_value.shape),
                chunks=(1, *test_value.shape),
                compressor=self.compressor,
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
                if len(shape) > 0:
                    self.vars_to_dims[name] = []
                for i, shape_i in enumerate(shape):
                    dim = f"{name}_dim_{i}"
                    coord = np.arange(shape_i, dtype="int")
                    dims.append(dim)
                    self.vars_to_dims[name].append(dim)
                    group_coords[dim] = coord
                    self.coords[dim] = coord
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
                    if var_value.ndim > 0:
                        self.vars_to_dims[var_name] = []
                    for i in range(var_value.ndim):
                        dim = f"{var_name}_dim_{i}"
                        coord = np.arange(var_value.shape[i], dtype="int")
                        dims.append(dim)
                        self.vars_to_dims[var_name].append(dim)
                        group_coords[dim] = coord
                        self.coords[dim] = coord
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

    def resize(
        self,
        tune: int | None = None,
        draws: int | None = None,
    ) -> "ZarrTrace":
        if not self.is_root_populated:
            raise RuntimeError(
                "The ZarrTrace has not been initialized yet. You must call resize on "
                "an instance that has already been initialized."
            )
        old_tuning = self.tuning_steps
        old_draws = self.draws
        desired_tune = tune or old_tuning
        desired_draws = draws or old_draws
        draws_in_chains = self._sampling_state.draw_idx[:]

        # For us to be able to resize, a few conditions must be met:
        # 1. If we want to change the number of tuning steps, the draws_in_chains must
        #    not be bigger than the old tune, and it must not be bigger than the desired
        #    tune. If the first condition weren't true, the sampler would have already
        #    stopped tuning, and it would be wrong to relabel some samples to belong to
        #    the tuning phase. If the second condition weren't true, the sampler would
        #    have continued tuning instead after the desired number of tuning steps had
        #    been taken.
        # 2. If we want to change the number of posterior draws, the draws_in_chains
        #    minus the old number of tuning steps must be less or equal to the desired
        #    number of draws. If this condition is not met, the sampler will have taken
        #    extra steps and we wont have stored the sampling state information at the
        #    end of the desired number of draws.
        change_tune = False
        change_draws = False
        if old_tuning != desired_tune:
            # Attempting to change the number of tuning steps
            if any(draws_in_chains > old_tuning):
                raise ValueError(
                    "Cannot change the number of tuning steps in the trace. "
                    "Some chains have finished their tuning phase and have "
                    "already performed steps in the posterior sampling regime."
                )
            elif any(draws_in_chains >= desired_tune):
                raise ValueError(
                    "Cannot change the number of tuning steps in the trace. "
                    "Some chains have already taken more steps than the desired number "
                    "of tuning steps. Please increase the desired number of tuning "
                    f"steps to at least {max(draws_in_chains)}."
                )
            change_tune = True
        if old_draws != desired_draws:
            # Attempting to change the number of draws
            if any((draws_in_chains - old_tuning) > desired_draws):
                raise ValueError(
                    "Cannot change the number of draws in the trace. "
                    "Some chains have already taken more steps than the desired number "
                    "of draws. Please increase the desired number of draws "
                    f"to at least {max(draws_in_chains) - old_tuning}."
                )
            change_draws = True
        if change_tune:
            self._resize_tuning_steps(desired_tune)
        if change_draws:
            self._resize_draws(desired_draws)
        return self

    def _resize_tuning_steps(self, desired_tune: int):
        groups = ["warmup_posterior", "warmup_sample_stats"]
        if "warmup_unconstrained_posterior" in dict(self.root.groups()):
            groups.append("warmup_unconstrained_posterior")
        for group in groups:
            self._resize_arrays_in_group(group=group, axis=1, new_size=desired_tune)
            zarr_draw = getattr(self.root, group).draw
            zarr_draw.resize(desired_tune)
            zarr_draw.set_basic_selection(
                slice(None), np.arange(desired_tune, dtype=zarr_draw.dtype)
            )
        self._sampling_state.tuning_steps.set_basic_selection((), desired_tune)

    def _resize_draws(self, desired_draws: int):
        groups = ["posterior", "sample_stats"]
        if "unconstrained_posterior" in dict(self.root.groups()):
            groups.append("unconstrained_posterior")
        for group in groups:
            self._resize_arrays_in_group(group=group, axis=1, new_size=desired_draws)
            zarr_draw = getattr(self.root, group).draw
            zarr_draw.resize(desired_draws)
            zarr_draw.set_basic_selection(
                slice(None), np.arange(desired_draws, dtype=zarr_draw.dtype)
            )
        self._sampling_state.draws.set_basic_selection((), desired_draws)

    def _resize_arrays_in_group(self, group: str, axis: int, new_size: int):
        zarr_group: Group = getattr(self.root, group)
        for _, array in zarr_group.arrays():
            dims = array.attrs.get("_ARRAY_DIMENSIONS", [])
            if len(dims) >= 2 and dims[1] == "draw":
                new_shape = list(array.shape)
                new_shape[axis] = new_size
                array.resize(new_shape)

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

    @classmethod
    def from_store(
        cls: type["ZarrTrace"],
        store: BaseStore | MutableMapping,
        synchronizer: Synchronizer | None = None,
    ) -> "ZarrTrace":
        if not _zarr_available:
            raise RuntimeError("You must install zarr to be able to create ZarrTrace instances")
        self: ZarrTrace = object.__new__(cls)
        self.root = zarr.group(
            store=store,
            overwrite=False,
            synchronizer=synchronizer,
        )
        self.synchronizer = synchronizer
        self.compressor = default_compressor

        groups = set(self.root.group_keys())
        assert groups >= {
            "posterior",
            "sample_stats",
            "warmup_posterior",
            "warmup_sample_stats",
            "constant_data",
            "observed_data",
            "_sampling_state",
        }

        if "posterior" in groups:
            for _, array in self.posterior.arrays():
                dims = array.attrs.get("_ARRAY_DIMENSIONS", [])
                if len(dims) >= 2 and dims[1] == "draw":
                    draws_per_chunk = int(array.chunks[1])
                    break
            else:
                draws_per_chunk = 1

        self.draws_per_chunk = int(draws_per_chunk)
        assert self.draws_per_chunk >= 1

        self.include_transformed = "unconstrained_posterior" in groups
        arrays = itertools.chain(
            self.posterior.arrays(),
            self.constant_data.arrays(),
            self.observed_data.arrays(),
        )
        if self.include_transformed:
            arrays = itertools.chain(arrays, self.unconstrained_posterior.arrays())
        varnames = []
        coords = {}
        vars_to_dims = {}
        for name, array in arrays:
            dims = array.attrs["_ARRAY_DIMENSIONS"]
            if dims[:2] == ["chain", "draw"]:
                # Random Variable
                vars_to_dims[name] = dims[2:]
                varnames.append(name)
            elif len(dims) == 1 and name == dims[0]:
                # Coordinate
                # We store all model coordinates, which means we have to exclude chain
                # and draw
                if name not in ["chain", "draw"]:
                    coords[name] = np.asarray(array)
            else:
                # Constant data or observation
                vars_to_dims[name] = dims
        self.varnames = varnames
        self.coords = coords
        self.vars_to_dims = vars_to_dims
        return self

    def assert_model_and_step_are_compatible(
        self,
        step: BlockedStep | CompoundStep,
        model: Model,
        vars: list[TensorVariable] | None = None,
    ):
        zarr_groups = set(self.root.group_keys())
        arrays_ = itertools.chain(
            self.posterior.arrays(),
            self.constant_data.arrays() if "constant_data" in zarr_groups else [],
            self.observed_data.arrays() if "observed_data" in zarr_groups else [],
        )
        if self.include_transformed:
            arrays_ = itertools.chain(arrays_, self.unconstrained_posterior.arrays())
        arrays = list(arrays_)
        zarr_varnames = []
        zarr_coords = {}
        zarr_vars_to_dims = {}
        zarr_deterministics = []
        zarr_free_vars = []
        for name, array in arrays:
            dims = array.attrs["_ARRAY_DIMENSIONS"]
            if dims[:2] == ["chain", "draw"]:
                # Random Variable
                zarr_vars_to_dims[name] = dims[2:]
                zarr_varnames.append(name)
                if array.attrs["kind"] == "freeRV":
                    zarr_free_vars.append(name)
                else:
                    zarr_deterministics.append(name)
            elif len(dims) == 1 and name == dims[0]:
                # Coordinate
                if name not in ["chain", "draw"]:
                    zarr_coords[name] = np.asarray(array)
            else:
                # Constant data or observation
                zarr_vars_to_dims[name] = dims
        zarr_constant_data = (
            [name for name in self.constant_data.array_keys() if name not in zarr_coords]
            if "constant_data" in zarr_groups
            else []
        )
        zarr_observed_data = (
            [name for name in self.observed_data.array_keys() if name not in zarr_coords]
            if "observed_data" in zarr_groups
            else []
        )
        autogenerated_dims = {dim for dim in zarr_coords if re.search(r"_dim_\d+$", dim)}

        # Check deterministics, free RVs and transformed RVs
        _, var_names = self.parse_varnames(model, vars)
        assert set(var_names) == set(zarr_free_vars + zarr_deterministics), (
            "The model deterministics and random variables given the sampled var_names "
            "do not match with the stored deterministics variables in the trace."
        )
        for name, array in arrays:
            if name not in zarr_free_vars or name not in zarr_deterministics:
                continue
            model_var = model[name]
            assert np.dtype(model_var.dtype) == np.dtype(array.dtype), (
                "The model deterministics and random variables given the sampled "
                "var_names do not match with the stored deterministics variables in "
                "the trace."
            )

        # Check coordinates
        assert (set(zarr_coords) - set(autogenerated_dims)) == set(model.coords) and all(
            np.array_equal(np.asarray(zarr_coords[dim]), np.asarray(coord))
            for dim, coord in model.coords.items()
        ), "Model coordinates don't match the coordinates stored in the trace"
        vars_to_explicit_dims = {}
        for name, dims in zarr_vars_to_dims.items():
            if len(dims) == 0 or all(dim in autogenerated_dims for dim in dims):
                # These variables wont be included in the named_vars_to_dims
                continue
            vars_to_explicit_dims[name] = [
                dim if dim not in autogenerated_dims else None for dim in dims
            ]
        assert set(vars_to_explicit_dims) == set(model.named_vars_to_dims) and all(
            vars_to_explicit_dims[name] == list(dims)
            for name, dims in model.named_vars_to_dims.items()
        ), "Some model variables have different dimensions than those stored in the trace."

        # Check constant data
        model_constant_data = find_constants(model)
        assert set(zarr_constant_data) == set(model_constant_data), (
            "The model constant data does not match with the stored constant data"
        )
        for name, model_data in model_constant_data.items():
            assert np.array_equal(self.constant_data[name], model_data, equal_nan=True), (
                "The model constant data does not match with the stored constant data"
            )

        # Check observed data
        model_observed_data = find_observations(model)
        assert set(zarr_observed_data) == set(model_observed_data), (
            "The model observed data does not match with the stored observed data"
        )
        for name, model_data in model_observed_data.items():
            assert np.array_equal(self.observed_data[name], model_data, equal_nan=True), (
                "The model observed data does not match with the stored observed data"
            )

        # Check sample stats given the step method
        stats_dtypes_shapes = get_stats_dtypes_shapes_from_steps(
            [step] if isinstance(step, BlockedStep) else step.methods
        )
        assert (set(stats_dtypes_shapes) | {"chain", "draw"}) == set(
            self.sample_stats.array_keys()
        ), "The step method sample stats do not match the ones stored in the trace."
        for name, array in self.sample_stats.arrays():
            if name in ("chain", "draw"):
                continue
            assert np.dtype(stats_dtypes_shapes[name][0]) == np.dtype(array.dtype), (
                "The step method sample stats do not match the ones stored in the trace."
            )

        assert step.sampling_state.is_compatible(self._sampling_state.sampling_state[0]), (
            "The step method sampling state class is incompatible with what's stored in the trace."
        )
