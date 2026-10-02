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
import base64
import importlib
import pickle
import warnings

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from pytensor.tensor.variable import TensorVariable
from xarray import DataTree

import pymc

from pymc.backends import _ZarrChainBase, _ZarrTraceBase
from pymc.backends.arviz import (
    coords_and_dims_for_inferencedata,
    find_constants,
    find_observations,
    make_attrs,
)
from pymc.backends.base import BaseTrace
from pymc.blocking import StatDtype, StatShape
from pymc.model.core import BaseModel, modelcontext
from pymc.step_methods.compound import (
    BlockedStep,
    CompoundStep,
    StatsBijection,
    get_stats_dtypes_shapes_from_steps,
)
from pymc.step_methods.state import (
    DataClassState,
    row_to_state,
    state_array_spec_map,
    state_class_map,
    state_to_row,
)
from pymc.util import UNSET, _UnsetType, get_default_varnames, is_transformed_name

try:
    import zarr

    from zarr import Group
    from zarr.abc.store import Store
    from zarr.codecs import ZstdCodec
    from zarr.dtype import Struct, VariableLengthUTF8, ZDType

    _zarr_available = True
except ImportError:
    from typing import TYPE_CHECKING, TypeVar

    if not TYPE_CHECKING:
        Store = TypeVar("Store")
        ZstdCodec = TypeVar("ZstdCodec")
        VariableLengthUTF8 = TypeVar("VariableLengthUTF8")
        ZDType = TypeVar("ZDType")
        Struct = TypeVar("Struct")
        Group = TypeVar("Group")
    _zarr_available = False


WARMUP_TAG = "warmup_"

if TYPE_CHECKING:
    from pymc.stats.convergence import SamplerWarning

# Pickle protocol used to serialize arbitrary python objects. It is recorded in
# the root group attributes (``pymc_pickle_protocol``) so that readers know how
# the pickled values were encoded.
PICKLE_PROTOCOL: int = pickle.HIGHEST_PROTOCOL

# Attribute used to tag arrays that store arbitrary python objects. Zarr v3 has
# no object dtype, so such values are pickled and base64 encoded into
# variable length utf8 strings. Readers must decode them with
# ``decode_object_value``.
OBJECT_CODEC_ATTR = "pymc_object_codec"


def encode_object_value(value: Any) -> str:
    """Encode an arbitrary python object as a base64-encoded pickle string."""
    return base64.b64encode(pickle.dumps(value, protocol=PICKLE_PROTOCOL)).decode("ascii")


def decode_object_value(value: Any, protocol: int | None = None) -> Any:
    """Decode a value stored with :func:`encode_object_value`.

    Empty strings (the fill value of object arrays) are decoded as ``None``.

    The ``protocol`` argument is advisory metadata (``pickle.loads`` infers the
    protocol from the serialized stream); it defaults to
    :data:`PICKLE_PROTOCOL`.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if not value:
        return None
    if protocol is None:
        protocol = PICKLE_PROTOCOL
    return pickle.loads(base64.b64decode(str(value)))


class ZarrChain(_ZarrChainBase, BaseTrace):
    """Interface object to interact with a single chain in a :class:`~.ZarrTrace`.

    Parameters
    ----------
    store : zarr.abc.store.Store | collections.abc.MutableMapping
        The store object where the zarr groups and arrays will be stored and read from.
        This store must exist before creating a ``ZarrChain`` object. ``ZarrChain`` are
        only intended to be used as interfaces to the individual chains of
        :class:`~.ZarrTrace` objects. This means that the :class:`~.ZarrTrace` should
        be the one that creates the store that is then provided to a ``ZarrChain``.
    stats_bijection : pymc.step_methods.compound.StatsBijection
        An object that maps between a list of step method stats and a dictionary of
        said stats with the accompanying stepper index.
    model : BaseModel
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
        filled. The default of 100 amortizes the cost of zarr writes; setting it to
        ``1`` gives the highest crash resilience, at the cost of a large slowdown.
    """

    def __init__(
        self,
        store: Store | MutableMapping,
        stats_bijection: StatsBijection,
        model: BaseModel | None = None,
        vars: Sequence[TensorVariable] | None = None,
        test_point: dict[str, np.ndarray] | None = None,
        draws_per_chunk: int = 100,
        fn: Callable | None = None,
        warning_columns: dict[str, list[str]] | None = None,
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
        self._posterior = zarr.open_group(store, path="posterior", mode="a")
        if self.unconstrained_variables:
            self._unconstrained_posterior = zarr.open_group(
                store, path="unconstrained_posterior", mode="a"
            )
            self._buffers["unconstrained_posterior"] = {}
        self._sample_stats = zarr.open_group(store, path="sample_stats", mode="a")
        self._sampling_state = zarr.open_group(store, path="_sampling_state", mode="a")
        self.stats_bijection = stats_bijection
        # Struct dtype and state class used to encode the chain's sampling state;
        # set later, in init_trace, via setup_state_dtype
        self._state_dtype: np.dtype | None = None
        self._state_class: type[DataClassState] | None = None
        # Flat stat names whose object dtype values are stored as typed columns
        self.warning_columns: dict[str, list[str]] = warning_columns or {}

    def link_stepper(self, step_method: BlockedStep | CompoundStep):
        """Provide a reference to the step method used during sampling.

        This reference can be used to facilitate writing the stepper's sampling state
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
        self,
        draw: Mapping[str, np.ndarray],
        stats: Sequence[Mapping[str, Any]],
        *,
        in_warmup: bool,
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
        self.buffer(group="sample_stats", var_name="in_warmup", value=bool(in_warmup))
        self._buffered_draws += 1
        if self._buffered_draws == self.draws_until_flush:
            self.flush()
            return True
        return None

    def record_sampling_state(self, step: BlockedStep | CompoundStep | None = None):
        """Record the sampling state information to the store's ``_sampling_state`` group.

        The number of draws taken so far (``draw_idx``) is stored as an integer array.
        The step method's ``sampling_state`` is an arbitrary python object, which has
        no zarr v3 dtype representation. It is pickled, base64 encoded and stored as a
        variable length utf8 string. It can be read back with
        :attr:`~ZarrChain.sampling_state`.

        Parameters
        ----------
        step : BlockedStep | CompoundStep | None
            The step method from which to take the ``sampling_state``. If ``None``,
            the ``step`` is taken to be the step method that was linked to the
            ``ZarrChain`` when calling :meth:`~ZarrChain.link_stepper`. If this method
            was never called, no step method ``sampling_state`` information is stored
            in the chain.
        """
        if step is None:
            step = self._step_method
        if step is not None:
            self.store_sampling_state(step.sampling_state)
        self._sampling_state["draw_idx"][self.chain] = self.draw_idx  # type: ignore[index]

    def setup_state_dtype(self, state_dtype: np.dtype, state_cls: type[DataClassState]):
        """Provide the struct dtype and state class used to encode sampling states."""
        self._state_dtype = state_dtype
        self._state_class = state_cls

    def store_sampling_state(self, sampling_state):
        state_array = self._sampling_state["sampling_state"]
        state_array.attrs.update(
            {
                "pymc_state_class": f"{type(sampling_state).__module__}.{type(sampling_state).__qualname__}"
            }
        )
        if self._state_dtype is not None and self._state_dtype.names:
            state_array.attrs.update({"pymc_state_classes": state_class_map(sampling_state)})
            state_array.attrs.update({"pymc_state_specs": state_array_spec_map(sampling_state)})
            row = state_to_row(sampling_state, self._state_dtype)
            # The zarr struct array stores array fields as raw bytes; the row buffer
            # is converted through its bytes so that the values are preserved
            native_row = np.frombuffer(row.tobytes(), dtype=state_array.dtype)[()]
            state_array[self.chain] = native_row
        else:  # stateless step: pickled utf8 array
            state_array[self.chain] = encode_object_value(sampling_state)

    @property
    def sampling_state(self):
        """The last sampling state recorded for this chain (if any)."""
        state_array = self._sampling_state["sampling_state"]
        state_class = self._state_class
        if state_class is None:
            state_class_name = state_array.attrs["pymc_state_class"]
            module_name, _, class_name = state_class_name.rpartition(".")
            state_class = getattr(importlib.import_module(module_name), class_name)
        if self._state_dtype is not None and self._state_dtype.names:
            return row_to_state(
                state_array[self.chain],
                state_class,
                classes=state_array.attrs.get("pymc_state_classes"),
                specs=state_array.attrs.get("pymc_state_specs"),
            )
        return decode_object_value(state_array[self.chain])

    def flush(self):
        """Write the data stored in the internal buffer to the desired zarr store.

        After writing the draws and stats returned by each step of the step method,
        the :meth:`~ZarrChain.record_sampling_state` is called, the internal buffer is
        cleared and the number of steps until the next flush is determined.
        """
        chain = self.chain
        draw_slice = slice(self.draw_idx, self.draw_idx + self.draws_until_flush)
        for group_name, buffer in self._buffers.items():
            group = getattr(self, f"_{group_name}")
            for var_name, var_value in buffer.items():
                columns = self.warning_columns.get(var_name)
                if columns is not None:
                    for column, attr in zip(columns, ("kind", "message", "level", "step")):
                        if attr == "kind":
                            values = np.array(
                                [0 if v is None else v.kind.value for v in var_value],
                                dtype="int64",
                            )
                        elif attr == "step":
                            values = np.array(
                                [-1 if v is None or v.step is None else v.step for v in var_value],
                                dtype="int64",
                            )
                        else:
                            values = np.array(
                                ["" if v is None else getattr(v, attr) for v in var_value],
                                dtype=object,
                            )
                        group[column].set_orthogonal_selection((chain, draw_slice), values)
                    continue
                array = group[var_name]
                if array.attrs.get(OBJECT_CODEC_ATTR):
                    values = np.array(
                        [encode_object_value(value) for value in var_value], dtype=object
                    )
                else:
                    values = np.stack(var_value)
                array.set_orthogonal_selection((chain, draw_slice), values)
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


def get_initial_fill_value_and_dtype(
    dtype: Any,
) -> tuple[FILL_VALUE_TYPE, Any, bool]:
    """Find the fill value and dtype used to initialize a zarr array.

    Object dtypes have no zarr v3 representation. They are stored as variable length
    utf8 strings, with the individual values pickled and base64 encoded by the caller
    (see :func:`encode_object_value`).

    Returns
    -------
    fill_value, dtype, is_object
    """
    if dtype is np.object_ or dtype == np.dtype("object"):
        return None, VariableLengthUTF8(), True
    if isinstance(dtype, ZDType):
        return dtype.default_scalar(), dtype, False
    _dtype = np.dtype(dtype)
    fill_value: FILL_VALUE_TYPE = None
    try:
        fill_value = DEFAULT_FILL_VALUES[_dtype]
    except KeyError:
        for key in DEFAULT_FILL_VALUES:
            if np.issubdtype(_dtype, key):
                fill_value = DEFAULT_FILL_VALUES[key]
                break
    return fill_value, _dtype, False


class ZarrTrace(_ZarrTraceBase):
    """Object that stores and enables access to MCMC draws stored in zarr groups.

    This class creates a zarr hierarchy to represent the sampling information which is
    intended to mimic :class:`xarray.DataTree`. The hierarchy looks like this:

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
    the groups are created once :meth:`~ZarrTrace.init_trace` is called with a few exceptions:
    unconstrained_posterior is only created if ``include_transformed = True``, and the
    groups prefixed with ``warmup_`` are created only after calling
    :meth:`~ZarrTrace.split_warmup_groups`.

    Since ``ZarrTrace`` objects are intended to be as close to
    :class:`xarray.DataTree` objects as possible, the groups store the dimension
    and coordinate information following the `xarray zarr v3 encoding specification
    <https://docs.xarray.dev/en/stable/internals/zarr-encoding-spec.html>`_.
    Arrays store their dimensions in the zarr v3 ``dimension_names`` metadata field.

    Parameters
    ----------
    store : zarr.abc.store.Store | collections.abc.MutableMapping | None
        The store object where the zarr groups and arrays will be stored and read from.
        Any zarr compatible storage object works. Keep in mind that if ``None`` is
        provided, a :class:`zarr.storage.MemoryStore` will be used, which means that
        information won't be visible to other processes and won't persist after the
        ``ZarrTrace`` life-cycle ends. If you want to have persistent storage, please
        use one of the multiple disk backed zarr storage options, e.g.
        :class:`~zarr.storage.LocalStore` or :class:`~zarr.storage.ZipStore`.
        Note that :class:`~zarr.storage.ZipStore` must be created with
        ``mode="w"`` to be writable, and that its contents are only persisted
        once ``.close()`` is called on it.
    compressors : Sequence | None | pymc.util.UNSET
        The compressors to use for the underlying zarr arrays. If ``None``, no
        compressor is used. If ``UNSET``, a default zarr ``ZstdCodec`` is used.
    draws_per_chunk : int
        The number of draws that make up a chunk in the variable's posterior array.
        Each variable's array shape is set to ``(n_chains, n_draws, *rv_shape)``, but
        the chunks are set to ``(1, draws_per_chunk, *rv_shape)``. This means that each
        chain will have it's own chunk to read or write to, allowing for concurrent
        write operations of different chains not to interfere with each other, and that
        multiple draws can belong to the same chunk. The variable's core dimension
        however, will never be split across different chunks. The default of 100
        amortizes the cost of zarr writes; setting it to ``1`` gives the highest crash
        resilience, at the cost of a large slowdown.
    include_transformed : bool
        If ``True``, the transformed, unconstrained value variables are included in the
        storage group.

    Notes
    -----
    ``ZarrTrace`` objects represent the storage information. If the underlying store
    persists on disk or over the network (e.g. with a :class:`zarr.storage.LocalStore`
    pointing to a cloud bucket) multiple processes will be able to concurrently access
    the same storage and read or write to it.

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
        store: Store | MutableMapping | None = None,
        compressors: Sequence | None | _UnsetType = UNSET,
        draws_per_chunk: int = 100,
        include_transformed: bool = False,
    ):
        if not _zarr_available:
            raise RuntimeError("You must install zarr to be able to create ZarrTrace instances")
        if compressors is UNSET:
            compressors = [ZstdCodec()]
        self.compressors = (
            list(compressors) if compressors is not None else None  # type: ignore[arg-type]
        )
        self.root = zarr.group(store=store, overwrite=True, zarr_format=3)
        self.root.attrs.update({"pymc_pickle_protocol": PICKLE_PROTOCOL})

        self.draws_per_chunk = int(draws_per_chunk)
        assert self.draws_per_chunk >= 1

        self.include_transformed = include_transformed

        # Sampler warnings collected during convergence checks. Zarr v3 has no
        # object dtype, so these are only kept in memory.
        self.global_warnings: list = []

        self._is_base_setup = False

    def groups(self) -> list[str]:
        return [str(group_name) for group_name, _ in self.root.groups()]

    @property
    def posterior(self) -> Group:
        return self.root["posterior"]  # type: ignore[return-value]

    @property
    def unconstrained_posterior(self) -> Group:
        return self.root["unconstrained_posterior"]  # type: ignore[return-value]

    @property
    def sample_stats(self) -> Group:
        return self.root["sample_stats"]  # type: ignore[return-value]

    @property
    def constant_data(self) -> Group:
        return self.root["constant_data"]  # type: ignore[return-value]

    @property
    def observed_data(self) -> Group:
        return self.root["observed_data"]  # type: ignore[return-value]

    @property
    def _sampling_state(self) -> Group:
        return self.root["_sampling_state"]  # type: ignore[return-value]

    def init_trace(
        self,
        chains: int,
        draws: int,
        tune: int,
        step: BlockedStep | CompoundStep,
        model: BaseModel | None = None,
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
        stats_dtypes_shapes = {"in_warmup": (bool, [])} | stats_dtypes_shapes
        # Object dtype stats (SamplerWarning warnings) are not stored as pickled
        # objects, but as typed columns derived from the warning objects
        expanded_stats: dict[str, tuple[Any, Any]] = {}
        self.warning_stat_names: tuple[str, ...] = ()
        warning_columns: dict[str, list[str]] = {}
        for stat_name, (dtype, shape) in stats_dtypes_shapes.items():
            if np.dtype(dtype) == np.dtype(object) and stat_name.endswith("warning"):
                self.warning_stat_names += (stat_name,)
                warning_columns[stat_name] = [
                    f"{stat_name}_type",
                    f"{stat_name}_message",
                    f"{stat_name}_level",
                    f"{stat_name}_step",
                ]
                expanded_stats.update(
                    {
                        # 0 is the "no warning" sentinel; WarningType values start at 1
                        f"{stat_name}_type": (np.int64, []),
                        f"{stat_name}_message": (VariableLengthUTF8(), []),
                        f"{stat_name}_level": (VariableLengthUTF8(), []),
                        # -1 is the "no step" sentinel for SamplerWarning.step
                        f"{stat_name}_step": (np.int64, []),
                    }
                )
            else:
                expanded_stats[stat_name] = (dtype, shape)
        self.init_group_with_empty(
            group=self.root.create_group(name="sample_stats", overwrite=True),
            var_dtype_and_shape=expanded_stats,
            chains=chains,
            draws=tune + draws,
        )

        state_dtype = step.sampling_state.struct_dtype()
        state_cls = type(step.sampling_state)
        self.init_sampling_state_group(
            tune=tune, chains=chains, state_dtype=state_dtype, state_cls=state_cls
        )

        self.straces = [
            ZarrChain(
                store=self.root.store,
                model=self.model,
                vars=self.vars,
                test_point=test_point,
                stats_bijection=StatsBijection(step.stats_dtypes),
                draws_per_chunk=self.draws_per_chunk,
                fn=self.fn,
                warning_columns=warning_columns,
            )
            for _ in range(chains)
        ]
        for chain, strace in enumerate(self.straces):
            strace.setup(draws=tune + draws, chain=chain, sampler_vars=None)
            strace.setup_state_dtype(state_dtype, state_cls)  # type: ignore[attr-defined]

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
            return int(self._sampling_state["tuning_steps"][()])
        except KeyError:  # pragma: no cover
            raise ValueError(
                "ZarrTrace has not been initialized and there is no tuning step information available"
            )

    @property
    def sampling_time(self):
        try:
            return float(self._sampling_state["sampling_time"][()])
        except KeyError:  # pragma: no cover
            raise ValueError(
                "ZarrTrace has not been initialized and there is no sampling time information available"
            )

    @sampling_time.setter
    def sampling_time(self, value):
        self._sampling_state["sampling_time"][()] = float(value)

    def init_sampling_state_group(
        self, tune: int, chains: int, state_dtype: np.dtype, state_cls: type[DataClassState]
    ):
        state = self.root.create_group(name="_sampling_state", overwrite=True)
        if state_dtype.names:
            sampling_state = state.create_array(  # type: ignore[arg-type]
                name="sampling_state",
                shape=(chains,),
                chunks=(1,),
                dtype=Struct.from_native_dtype(state_dtype),
                fill_value=np.zeros((), dtype=state_dtype)[()],
                compressors=self.compressors,
                dimension_names=["chain"],
            )
        else:
            # Steps without a sampling state (empty DataClassState) have no
            # struct representation; the state is stored as a pickled utf8 array
            sampling_state = state.create_array(
                name="sampling_state",
                shape=(chains,),
                chunks=(1,),
                dtype=VariableLengthUTF8(),  # type: ignore[arg-type]
                fill_value="",
                compressors=self.compressors,
                dimension_names=["chain"],
            )
            sampling_state.attrs.update({OBJECT_CODEC_ATTR: "pickle_base64"})
        sampling_state.attrs.update(
            {"pymc_state_class": f"{state_cls.__module__}.{state_cls.__qualname__}"}
        )

        state.create_array(
            name="draw_idx",
            data=np.zeros(chains, dtype="int"),
            chunks=(1,),
            fill_value=-1,
            compressors=self.compressors,
            dimension_names=["chain"],
        )

        state.create_array(
            name="tuning_steps",
            data=np.array(tune),
            fill_value=0,
            compressors=self.compressors,
        )
        state.create_array(
            name="sampling_time",
            data=np.array(0.0),
            fill_value=0.0,
            compressors=self.compressors,
        )
        state.create_array(
            name="sampling_start_time",
            data=np.array(0.0),
            fill_value=0.0,
            compressors=self.compressors,
        )

        state.create_array(
            name="chain",
            data=np.arange(chains),
            dimension_names=["chain"],
            compressors=self.compressors,
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
            fill_value, dtype, is_object = get_initial_fill_value_and_dtype(_dtype)
            shape = shape or ()
            attributes = extra_var_attrs[name] if extra_var_attrs is not None else {}
            if is_object:
                attributes = {**attributes, OBJECT_CODEC_ATTR: "pickle_base64"}
            try:
                core_dims = self.vars_to_dims[name]
                for dim in core_dims:
                    group_coords[dim] = self.coords[dim]
            except KeyError:
                core_dims = []
                for i, shape_i in enumerate(shape):
                    dim = f"{name}_dim_{i}"
                    core_dims.append(dim)
                    assert shape_i is not None, f"{dim} shape is None"
                    group_coords[dim] = np.arange(shape_i, dtype="int")
            dims = ("chain", "draw", *core_dims)
            array = group.create_array(  # type: ignore[arg-type]
                name=name,
                dtype=dtype,
                fill_value=fill_value,
                shape=(chains, draws, *shape),  # type: ignore[arg-type]
                chunks=(1, self.draws_per_chunk, *shape),  # type: ignore[arg-type]
                compressors=self.compressors,
                dimension_names=dims,
                attributes=attributes,
            )
        for dim, coord in group_coords.items():
            group.create_array(
                name=dim,
                data=np.asarray(coord),
                dimension_names=[dim],
                compressors=self.compressors,
            )
        return group

    def create_group(self, name: str, data_dict: dict[str, np.ndarray]) -> Group | None:
        group: Group | None = None
        if data_dict:
            group_coords = {}
            group = self.root.create_group(name=name, overwrite=True)
            for var_name, var_value in data_dict.items():
                _, _, is_object = get_initial_fill_value_and_dtype(var_value.dtype)
                if is_object:
                    var_value = np.array(
                        [encode_object_value(value) for value in var_value.ravel()],
                        dtype=object,
                    ).reshape(var_value.shape)
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
                group.create_array(  # type: ignore[arg-type,union-attr]
                    name=var_name,
                    data=var_value,
                    compressors=self.compressors,
                    dimension_names=dims,
                )
            for dim, coord in group_coords.items():
                group.create_array(  # type: ignore[arg-type,union-attr]
                    name=dim,
                    data=np.asarray(coord),
                    dimension_names=[dim],
                    compressors=self.compressors,
                )
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
            the root hierarchy, a ``RuntimeError`` is raised. If this flag is ``False``
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
                del self.root[f"{WARMUP_TAG}{group_name}"]
            except KeyError:
                pass
            return
        for name, array in posterior_group.arrays():  # type: ignore[union-attr]
            dims = array.metadata.dimension_names or ()  # type: ignore[union-attr]
            array_attrs = array.attrs.asdict()
            if name == "draw":
                warmup_group.create_array(
                    name="draw",
                    data=np.arange(tune),
                    dimension_names=["draw"],
                    compressors=self.compressors,
                )
                posterior_group.create_array(  # type: ignore[union-attr]
                    name=name,
                    data=np.arange(array.shape[0] - tune),
                    dimension_names=["draw"],
                    overwrite=True,
                    compressors=self.compressors,
                )
            else:
                if len(dims) >= 2 and dims[:2] == ("chain", "draw"):
                    warmup_idx: slice | tuple[slice, slice] = (
                        slice(None),
                        slice(None, tune, None),
                    )
                    posterior_idx = (slice(None), slice(tune, None, None))
                else:
                    warmup_idx = slice(None)
                warmup_group.create_array(  # type: ignore[union-attr]
                    name=name,
                    data=array[warmup_idx],  # type: ignore[arg-type]
                    chunks=array.chunks,
                    compressors=self.compressors,
                    dimension_names=dims,
                    attributes=array_attrs,
                )
                if len(dims) >= 2 and dims[:2] == ("chain", "draw"):
                    posterior_group.create_array(  # type: ignore[union-attr]
                        name=name,
                        data=array[posterior_idx],  # type: ignore[arg-type]
                        chunks=array.chunks,
                        overwrite=True,
                        compressors=self.compressors,
                        dimension_names=dims,
                        attributes=array_attrs,
                    )

    @property
    def warnings(self) -> "list[list[SamplerWarning]]":
        """The non-empty :class:`~pymc.stats.convergence.SamplerWarning` objects.

        reconstructed from the typed warning columns of the ``sample_stats`` group,
        in draw order, as one list per chain. Warning payloads that cannot be typed
        (``extra``, divergence points) are not reconstructed; the warning's
        interpolated ``message`` retains the human-readable content.
        """
        from pymc.stats.convergence import SamplerWarning, WarningType

        warnings_per_chain: list[list[SamplerWarning]] = []
        for chain in range(len(self.straces)):
            chain_warnings: list[SamplerWarning] = []
            for stat_name in self.warning_stat_names:
                type_array = self.sample_stats[f"{stat_name}_type"]
                message_array = self.sample_stats[f"{stat_name}_message"]
                level_array = self.sample_stats[f"{stat_name}_level"]
                step_array = self.sample_stats[f"{stat_name}_step"]
                for draw in range(type_array.shape[1]):  # type: ignore[union-attr]
                    kind_code = int(type_array[chain, draw])  # type: ignore[arg-type,index]
                    if kind_code == 0:
                        continue
                    step_code = int(step_array[chain, draw])  # type: ignore[arg-type,index]
                    chain_warnings.append(
                        SamplerWarning(
                            WarningType(kind_code),
                            str(message_array[chain, draw]),  # type: ignore[index]
                            str(level_array[chain, draw]),  # type: ignore[index]
                            None if step_code < 0 else step_code,
                        )
                    )
            warnings_per_chain.append(chain_warnings)
        return warnings_per_chain

    def to_datatree(self, save_warmup: bool = False, eager: bool = False) -> DataTree:
        """Convert ``ZarrTrace`` to :class:`~xarray.DataTree`.

        The zarr group hierarchy naturally translates into a ``DataTree``, so this
        conversion opens the whole hierarchy with :func:`xarray.open_datatree`. The
        ``_sampling_state`` group is excluded, and the warmup groups are only
        included if ``save_warmup`` is ``True``. Each group's attributes are
        extended with the global ``tuning_steps`` and ``sampling_time`` metadata.

        Parameters
        ----------
        save_warmup : bool
            If ``True``, all of the warmup groups are stored in the data tree.
        eager : bool
            If ``True``, all of the data is loaded into memory. If ``False``, the
            data is lazily loaded when accessed.

        Notes
        -----
        ``xarray`` requires the zarr groups to have consolidated metadata, which is
        written by calling :func:`zarr.consolidate_metadata` on the root's store.
        Consolidated metadata is not (yet) part of the zarr v3 specification, so
        zarr will warn about it. The returned ``DataTree`` operates on freshly
        opened datasets, so future changes to the ``ZarrTrace`` are not reflected
        in it.
        """
        self.split_warmup_groups()
        # Xarray complains if we try to open a zarr hierarchy that doesn't have
        # consolidated metadata
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=zarr.errors.ZarrUserWarning)
            zarr.consolidate_metadata(self.root.store)
        try:
            global_attrs = {
                "tuning_steps": self.tuning_steps,
                "sampling_time": self.sampling_time,
            }
        except (KeyError, ValueError):
            global_attrs = {}  # pragma: no cover
        tree = xr.open_datatree(self.root.store, engine="zarr", mask_and_scale=False)  # type: ignore[arg-type]
        for name, node in list(tree.children.items()):
            if name.startswith("_") or (not save_warmup and name.startswith(WARMUP_TAG)):
                del tree[name]
                continue
            node.attrs.update(global_attrs)
        return tree.load() if eager else tree

    def to_inferencedata(self, save_warmup: bool = False, eager: bool = False) -> DataTree:
        """Convert ``ZarrTrace`` to :class:`~.xarray.DataTree` with arviz attributes.

        This converts all the groups in the ``ZarrTrace.root`` hierarchy into an
        ``DataTree`` object with the arviz inference library attributes applied.
        The only exception is that ``_sampling_state`` is excluded.

        Parameters
        ----------
        save_warmup : bool
            If ``True``, all of the warmup groups are stored in the inference data
            object.
        eager : bool
            If ``True``, all of the data is loaded into memory. If ``False``, the data
            is lazily loaded when accessed.

        Notes
        -----
        ``xarray`` and in turn ``arviz`` require the zarr groups to have consolidated
        metadata. See :meth:`~ZarrTrace.to_datatree` for more details.
        """
        tree = self.to_datatree(save_warmup=save_warmup, eager=eager)
        for node in tree.children.values():
            node.attrs = make_attrs(attrs={**node.attrs}, inference_library=pymc)
        return tree
