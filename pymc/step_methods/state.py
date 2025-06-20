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
import importlib

from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, fields
from numbers import Number
from typing import Any, ClassVar, ForwardRef, Generic, TypeVar

import numpy as np
import typeguard

from pymc.util import RandomGeneratorState, get_state_from_generator, random_generator_from_state

dataclass_state = dataclass(kw_only=True)


@dataclass_state
class DataClassState:
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]] = {}

    def is_compatible(self, other: Any) -> bool:
        return compatible_dataclass_values(self, other)


def resolve_typehint(hint: Any, anchor: object = None) -> Any:
    if isinstance(hint, str | ForwardRef):
        if anchor is None:
            globalns = globals()
        else:
            module = importlib.import_module(anchor.__module__)
            globalns = vars(module)
        recursive_guard: frozenset[str] = frozenset()
        hint = (
            eval(hint, globalns)
            if isinstance(hint, str)
            else hint._evaluate(globalns, None, recursive_guard=recursive_guard)
        )
    return hint


def compatible_dataclass_values(v1: Any, v2: Any, typehint: Any = None) -> bool:
    if typehint is not None:
        # Some dataclass fields support multiple different types of values
        # e.g. SomeClass | None union types. If v1 is a `SomeClass` object
        # and v2 is `None`, just checking the values would determine that
        # they are incompatible with each other. We need to check that
        # they are compatible with the typehint signature instead.
        # If they are compatible with the type hint, then we say that the
        # values are compatible.
        # If the values aren't compatible with the type hint (this could
        # happen because python isn't strongly typed!), then we resort to
        # comparing the values directly.
        try:
            assert typehint is not Any
            typeguard.check_type(v1, typehint)
            typeguard.check_type(v2, typehint)
            return True
        except Exception:
            pass
    if v1.__class__ != v2.__class__ and not (isinstance(v1, Number) and isinstance(v2, Number)):
        # Numbers might have different classes (e.g. float("32") and np.float64(32))
        # but numbers are compatible with each other
        return False
    if isinstance(v1, tuple):
        return len(v1) == len(v2) or all(
            compatible_dataclass_values(v1i, v2i) for v1i, v2i in zip(v1, v2, strict=True)
        )
    elif isinstance(v1, dict):
        return set(v1) == set(v2) or all(compatible_dataclass_values(v1[k], v2[k]) for k in v1)
    elif isinstance(v1, np.ndarray):
        return v1.dtype == v2.dtype
    elif isinstance(v1, np.random.Generator):
        return True
    elif isinstance(v1, DataClassState):
        if set(fields(v1)) != set(fields(v2)):
            return False
        for field in fields(v1):
            val1 = getattr(v1, field.name)
            val2 = getattr(v2, field.name)
            if not (isinstance(val1, DataClassState) and isinstance(val2, DataClassState)):
                typehint = resolve_typehint(field.type, anchor=v1)
            else:
                typehint = None
            if not compatible_dataclass_values(val1, val2, typehint):
                return False
    return True


def equal_dataclass_values(v1, v2):
    if v1.__class__ != v2.__class__ and not (isinstance(v1, Number) and isinstance(v2, Number)):
        # Numbers might have different classes (e.g. float("32") and np.float64(32))
        # but numbers are equal based on their value and not their type
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
    elif isinstance(v1, np.random.Generator):
        return equal_dataclass_values(v1.bit_generator.state, v2.bit_generator.state)
    elif isinstance(v1, DataClassState):
        return set(fields(v1)) == set(fields(v2)) and all(
            equal_dataclass_values(getattr(v1, f1.name), getattr(v2, f2.name))
            for f1, f2 in zip(fields(v1), fields(v2), strict=True)
        )
    else:
        return v1 == v2


SamplingStateType = TypeVar("SamplingStateType", bound=DataClassState)


class WithSamplingState(Generic[SamplingStateType]):
    """Mixin class that adds the ``sampling_state`` property to an object.

    The object's type must define the ``_state_class`` as a valid
    :py:class:`~pymc.step_method.DataClassState`. Once that happens, the
    object's ``sampling_state`` property can be read or set to get
    the state represented as objects of the ``_state_class`` type.
    """

    _state_class: type[SamplingStateType]

    @property
    def sampling_state(self) -> SamplingStateType:
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
        assert self.sampling_state.is_compatible(state), (
            "The supplied state is incompatible with the current sampling state."
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
