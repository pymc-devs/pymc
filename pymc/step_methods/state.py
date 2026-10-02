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
import hashlib
import importlib
import pickle

from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, fields
from typing import Any, ClassVar, cast, get_type_hints

import numpy as np

from pymc.util import (
    RandomGeneratorState,
    get_state_from_generator,
    random_generator_from_state,
)

dataclass_state = dataclass(kw_only=True)

# Size in bytes of the fixed buffer that holds a pickled generator state inside
# the struct row. The largest standard bit generator (MT19937) needs about 3 KB;
# encoders raise when a state does not fit, so data is never silently truncated.
GENERATOR_STATE_SIZE = 4096
# Size in bytes of the fixed string buffer that holds the base64 SHA-256 hash
# of pickled values (44 characters for a SHA-256 digest).
HASH_STRING_SIZE = 64
# Size in bytes of the fixed buffer that holds bit generator names
# ("PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937")
BIT_GENERATOR_NAME_SIZE = 16


@dataclass_state
class DataClassState:
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]] = {}

    def struct_dtype(self) -> np.dtype:
        """Return the numpy structured dtype that describes this state.

        Scalar fields are typed natively (bool, int, float). Array fields keep
        their dtype and shape. Nested :class:`DataClassState` fields recurse into
        nested dtypes, and lists of states get one field per member. Random
        generators are described by their bit generator name, their pickled
        state (in a fixed size bytes buffer, see ``GENERATOR_STATE_SIZE``), and
        a base64 SHA-256 hash of the state bytes.

        Fields that have no fixed-size representation (str, list, dict, ...)
        are not part of the struct row. They are pickled separately, see
        :func:`pickle_field_map`.
        """
        return np.dtype(
            [
                field_spec
                for field in fields(self)  # type: ignore[arg-type]
                for field_spec in _field_dtype_specs(field.name, getattr(self, field.name))
            ]
        )


def _is_state_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(v, DataClassState | WithSamplingState) for v in value)
    )


def _member_state(value: Any) -> DataClassState:
    if isinstance(value, WithSamplingState):
        return value.sampling_state
    return value


def _field_dtype_specs(name: str, value: Any) -> list[tuple]:
    """Return the (name, dtype[, shape]) spec(s) that describe a state field."""
    if _is_state_list(value):
        return [
            (f"{name}_{i}", _member_state(member).struct_dtype()) for i, member in enumerate(value)
        ]
    if isinstance(value, bool):
        return [(name, "?")]
    if isinstance(value, int):
        return [(name, "i8")]
    if isinstance(value, float):
        return [(name, "f8")]
    if isinstance(value, np.random.Generator | RandomGeneratorState):
        return [
            (
                name,
                np.dtype(
                    [
                        ("bit_generator_name", f"S{BIT_GENERATOR_NAME_SIZE}"),
                        ("state", f"S{GENERATOR_STATE_SIZE}"),
                        ("state_hash", f"S{HASH_STRING_SIZE}"),
                    ]
                ),
            )
        ]
    if isinstance(value, DataClassState):
        return [(name, value.struct_dtype())]
    if isinstance(value, WithSamplingState):
        return [(name, value.sampling_state.struct_dtype())]
    if isinstance(value, np.ndarray):
        return [(name, value.dtype, value.shape)]
    # Fields without a fixed-size representation (str, list, dict, ...) are
    # pickled separately instead of taking part in the struct row
    return []


def _generator_row(rng: np.random.Generator | RandomGeneratorState, dtype: np.dtype) -> np.void:
    """Encode a generator (or its state) as the generator struct row."""
    if isinstance(rng, np.random.Generator):
        rng = get_state_from_generator(rng)
    name = rng.bit_generator_state["bit_generator"]
    payload = pickle.dumps(rng)
    if len(payload) > GENERATOR_STATE_SIZE:
        raise ValueError(
            f"Pickled generator state is {len(payload)} bytes, which does not fit "
            f"the {GENERATOR_STATE_SIZE} bytes buffer."
        )
    return cast(
        "np.void",
        np.array(
            (
                name.encode("ascii"),
                payload,
                _base64_hash(payload).encode("ascii"),
            ),
            dtype=dtype,
        )[()],
    )


def _state_row(state: DataClassState, dtype: np.dtype) -> np.ndarray:
    """Encode a state as the row described by ``dtype``."""
    fields_map = dtype.fields if dtype.fields is not None else {}
    row = np.empty((), dtype=dtype)
    for field in fields(state):
        name = field.name
        value = getattr(state, name)
        if _is_state_list(value):  # one struct field per member state
            for i, member in enumerate(value):
                member_field = f"{name}_{i}"
                row[member_field] = _state_row(_member_state(member), fields_map[member_field][0])
            continue
        field_spec = fields_map.get(name)
        if field_spec is None:
            # Not part of the struct row: the value is pickled separately
            continue
        field_dtype = field_spec[0]
        if field_dtype.names is not None:  # structured sub-dtype
            if field_dtype.names == ("bit_generator_name", "state", "state_hash"):
                row[name] = _generator_row(value, field_dtype)
            else:
                row[name] = _state_row(value, field_dtype)
        elif field_dtype.shape != ():  # array field
            row[name] = value
        else:
            row[name] = value
    return row


def state_to_row(state: DataClassState, dtype: np.dtype | None = None) -> np.ndarray:
    """Encode a sampling state as a single structured numpy row.

    See :meth:`DataClassState.struct_dtype` for the dtype layout. Values that do
    not fit their fixed buffers raise ``ValueError`` so that data is never
    silently truncated.
    """
    return _state_row(state, dtype if dtype is not None else state.struct_dtype())


def pickle_field_map(state: DataClassState, prefix: str = "") -> dict[str, Any]:
    """Return the state fields that cannot be stored in the struct row.

    Fields without a fixed-size representation (str, list, dict, ...) are
    collected here, keyed by their field path, so that they can be pickled and
    stored without a size limit.
    """
    pickled = {}
    for field in fields(state):
        value = getattr(state, field.name)
        path = f"{prefix}{field.name}"
        if isinstance(value, DataClassState):
            pickled.update(pickle_field_map(value, prefix=f"{path}."))
        elif _is_state_list(value):
            for i, member in enumerate(value):
                pickled.update(pickle_field_map(_member_state(member), prefix=f"{path}_{i}."))
        elif not _field_dtype_specs(field.name, value):
            pickled[path] = value
    return pickled


def encode_pickled_field(value: Any) -> tuple[str, str]:
    """Return the base64 pickle of a value and the base64 hash of its payload."""
    payload = pickle.dumps(value)
    return base64.b64encode(payload).decode("ascii"), _base64_hash(payload)


def decode_pickled_field(payload: str, stored_hash: str, name: str) -> Any:
    """Decode a value encoded by :func:`encode_pickled_field`.

    The hash is verified before unpickling; empty payloads decode as ``None``.
    """
    if not payload:
        return None
    raw = base64.b64decode(payload)
    if _base64_hash(raw) != stored_hash:
        raise ValueError(
            f"Stored value of field {name!r} does not match its recorded hash. "
            "The stored data may be corrupted."
        )
    return pickle.loads(raw)


def state_class_map(state: DataClassState, prefix: str = "") -> dict[str, str]:
    """Return the fully qualified class name of every nested state field.

    Runtime subclasses are not recoverable from a struct dtype alone (the dtype
    is built from the declared field annotations), so the map is stored as
    metadata and used to reconstruct states with their original classes.
    """
    class_map = {}
    for field in fields(state):
        value = getattr(state, field.name)
        if isinstance(value, DataClassState):
            path = f"{prefix}{field.name}"
            cls = type(value)
            class_map[path] = f"{cls.__module__}.{cls.__qualname__}"
            class_map.update(state_class_map(value, prefix=f"{path}."))
        elif _is_state_list(value):
            for i, member in enumerate(value):
                member_state = _member_state(member)
                path = f"{prefix}{field.name}_{i}"
                cls = type(member_state)
                class_map[path] = f"{cls.__module__}.{cls.__qualname__}"
                class_map.update(state_class_map(member_state, prefix=f"{path}."))
    return class_map


def state_array_spec_map(state: DataClassState, prefix: str = "") -> dict[str, list]:
    """Return the dtype and shape of every ndarray field in the state.

    zarr's struct dtype stores ndarray fields as raw bytes, losing their dtype
    and shape, so the specs are stored as metadata and used to reconstruct the
    arrays on read.
    """
    specs = {}
    for field in fields(state):
        value = getattr(state, field.name)
        if isinstance(value, DataClassState):
            path = f"{prefix}{field.name}"
            specs.update(state_array_spec_map(value, prefix=f"{path}."))
        elif _is_state_list(value):
            for i, member in enumerate(value):
                path = f"{prefix}{field.name}_{i}"
                specs.update(state_array_spec_map(_member_state(member), prefix=f"{path}."))
        elif isinstance(value, np.ndarray) and value.dtype.names is None:
            path = f"{prefix}{field.name}"
            specs[path] = [value.dtype.str, list(value.shape)]
    return specs


def row_to_state(
    row: np.void,
    state_cls: type[DataClassState],
    classes: dict[str, str] | None = None,
    specs: dict[str, list] | None = None,
    pickled: dict[str, Any] | None = None,
    prefix: str = "",
) -> DataClassState:
    """Decode a structured row produced by :func:`state_to_row`.

    Generator states are verified against their stored base64 SHA-256 hash
    before unpickling. Fields that are not part of the struct row are taken from
    ``pickled`` (see :func:`pickle_field_map`), where they are stored encoded.
    Nested fields are decoded with their runtime classes when available (from
    ``classes``, see :func:`state_class_map`), falling back to the declared
    field annotations.
    """
    classes = classes or {}
    specs = specs or {}
    pickled = pickled or {}
    hints = get_type_hints(state_cls)
    fields_map = row.dtype.fields if row.dtype.fields is not None else {}
    kwargs: dict[str, Any] = {}
    for field in fields(state_cls):
        name = field.name
        field_spec = fields_map.get(name)
        if field_spec is None and f"{name}_0" not in fields_map:
            # Not part of the struct row: the value is pickled separately
            kwargs[name] = pickled.get(f"{prefix}{name}")
            continue
        if field_spec is None:
            # List fields are stored as one struct field per member
            member_fields = []
            i = 0
            while f"{name}_{i}" in fields_map:
                member_fields.append(f"{name}_{i}")
                i += 1
            member_hint = hints[name]
            if hasattr(member_hint, "__args__") and member_hint.__args__:
                default_member_cls = member_hint.__args__[0]
            else:
                default_member_cls = None
            members = []
            for i, member_field in enumerate(member_fields):
                path = f"{prefix}{member_field}"
                member_cls_name = classes.get(path)
                if member_cls_name is not None:
                    module_name, _, class_name = member_cls_name.rpartition(".")
                    member_cls = getattr(importlib.import_module(module_name), class_name)
                else:
                    member_cls = default_member_cls
                members.append(
                    row_to_state(
                        row[member_field],
                        member_cls,
                        classes=classes,
                        specs=specs,
                        pickled=pickled,
                        prefix=f"{path}.",
                    )
                )
            kwargs[name] = members
            continue
        field_dtype = field_spec[0]
        value = row[name]
        if field_dtype.names is not None:  # structured sub-dtype
            if field_dtype.names == ("bit_generator_name", "state", "state_hash"):
                kwargs[name] = _generator_from_row(value)
            else:
                path = f"{prefix}{name}"
                nested_cls_name = classes.get(path)
                if nested_cls_name is not None:
                    module_name, _, class_name = nested_cls_name.rpartition(".")
                    nested_cls = getattr(importlib.import_module(module_name), class_name)
                else:
                    nested_cls = hints[name]
                kwargs[name] = row_to_state(
                    value,
                    nested_cls,
                    classes=classes,
                    specs=specs,
                    pickled=pickled,
                    prefix=f"{path}.",
                )
        elif field_dtype.kind == "V":  # array field
            # Subarray fields become raw bytes through zarr struct roundtrips;
            # the dtype and shape are recovered from the stored specs
            path = f"{prefix}{name}"
            if path in specs:
                dtype_str, shape = specs[path]
                kwargs[name] = np.frombuffer(bytes(value), dtype=np.dtype(dtype_str)).reshape(shape)
            else:
                kwargs[name] = value
        elif hints.get(name) is np.ndarray:  # 0-d array field
            kwargs[name] = np.asarray(value, dtype=field_dtype)
        else:
            kwargs[name] = value.item()
    return state_cls(**kwargs)


def _generator_from_row(row: np.void) -> RandomGeneratorState | None:
    payload = bytes(row["state"]).rstrip(b"\x00")
    if payload:
        _check_hash(payload, row["state_hash"], "rng state")
    return pickle.loads(payload) if payload else None


def _unpickle_with_hash(payload: bytes, stored_hash: bytes, name: str) -> Any:
    if not payload:
        return None
    _check_hash(payload, stored_hash, name)
    return pickle.loads(payload)


def _check_hash(payload: bytes, stored_hash: bytes, name: str) -> None:
    if _base64_hash(payload).encode("ascii") != stored_hash:
        raise ValueError(
            f"Stored value of field {name!r} does not match its recorded hash. "
            "The stored data may be corrupted."
        )


def _base64_hash(payload: bytes) -> str:
    return base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")


def equal_dataclass_values(v1, v2):
    if isinstance(v1, bool | int | float | np.generic | np.ndarray) and isinstance(
        v2, bool | int | float | np.generic | np.ndarray
    ):
        # Numeric values are compared by value, regardless of their
        # representation (python scalars, numpy scalars, or arrays)
        arr1, arr2 = np.asarray(v1), np.asarray(v2)
        if arr1.dtype.kind in "fc" and arr2.dtype.kind in "fc":
            return bool(np.array_equal(arr1, arr2, equal_nan=True))
        return bool(np.array_equal(arr1, arr2))
    if v1.__class__ != v2.__class__:
        return False
    if isinstance(v1, (list, tuple)):  # noqa: UP038
        return len(v1) == len(v2) and all(
            equal_dataclass_values(v1i, v2i) for v1i, v2i in zip(v1, v2, strict=True)
        )
    elif isinstance(v1, dict):
        if set(v1) != set(v2):
            return False
        return all(equal_dataclass_values(v1[k], v2[k]) for k in v1)
    elif isinstance(v1, np.ndarray):
        return bool(np.array_equal(v1, v2, equal_nan=True))
    elif isinstance(v1, np.generic) and isinstance(v2, np.ndarray | np.generic):
        # Scalar and 0-d array representations of the same value are equal
        return bool(np.array_equal(v1, v2, equal_nan=True))
    elif isinstance(v1, np.random.Generator):
        return equal_dataclass_values(v1.bit_generator.state, v2.bit_generator.state)
    elif isinstance(v1, DataClassState):
        return set(fields(v1)) == set(fields(v2)) and all(
            equal_dataclass_values(getattr(v1, f1.name), getattr(v2, f2.name))
            for f1, f2 in zip(fields(v1), fields(v2), strict=True)
        )
    else:
        return v1 == v2


class WithSamplingState:
    """Mixin class that adds the ``sampling_state`` property to an object.

    The object's type must define the ``_state_class`` as a valid
    :py:class:`~pymc.step_method.DataClassState`. Once that happens, the
    object's ``sampling_state`` property can be read or set to get
    the state represented as objects of the ``_state_class`` type.
    """

    _state_class: type[DataClassState] = DataClassState

    def struct_dtype(self) -> np.dtype:
        """Return the numpy structured dtype that describes this object's state."""
        return self.sampling_state.struct_dtype()

    @property
    def sampling_state(self) -> DataClassState:
        state_class = self._state_class
        kwargs = {}
        for field in fields(state_class):
            is_tensor_name = field.metadata.get("tensor_name", False)
            val: Any
            if is_tensor_name:
                val = [var.name for var in getattr(self, "vars")]
            else:
                val = getattr(self, field.name, field.default)
            if val is MISSING:
                raise AttributeError(
                    f"{type(self).__name__!r} object has no attribute {field.name!r}"
                )
            _val: Any
            if isinstance(val, WithSamplingState):
                _val = val.sampling_state
            elif isinstance(val, np.random.Generator):
                _val = get_state_from_generator(val)
            else:
                _val = val
            kwargs[field.name] = deepcopy(_val)
        return state_class(**kwargs)

    @sampling_state.setter
    def sampling_state(self, state: DataClassState):
        state_class = self._state_class
        assert isinstance(state, state_class), (
            f"Encountered invalid state class '{state.__class__}'. State must be '{state_class}'"
        )
        for field in fields(state_class):
            is_tensor_name = field.metadata.get("tensor_name", False)
            state_val = deepcopy(getattr(state, field.name))
            if isinstance(state_val, RandomGeneratorState):
                state_val = random_generator_from_state(state_val)
            is_frozen = field.metadata.get("frozen", False)
            self_val: Any
            if is_tensor_name:
                self_val = [var.name for var in getattr(self, "vars")]
                assert is_frozen
            else:
                self_val = getattr(self, field.name, field.default)
            if is_frozen:
                if not equal_dataclass_values(state_val, self_val):
                    raise ValueError(
                        "The received sampling state must have the same values for the "
                        f"frozen fields. Field {field.name!r} has different values. "
                        f"Expected {self_val} but got {state_val}"
                    )
            else:
                if isinstance(state_val, DataClassState):
                    assert isinstance(self_val, WithSamplingState)
                    self_val.sampling_state = state_val
                    setattr(self, field.name, self_val)
                else:
                    setattr(self, field.name, state_val)
