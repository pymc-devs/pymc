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
from copy import deepcopy
from dataclasses import MISSING, Field, dataclass, fields
from typing import Any, ClassVar

import numpy as np

from pymc.util import RandomGeneratorState, get_state_from_generator, random_generator_from_state

dataclass_state = dataclass(kw_only=True)


@dataclass_state
class DataClassState:
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]] = {}


def equal_dataclass_values(v1, v2):
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
