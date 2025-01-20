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
from dataclasses import field

import numpy as np
import pytest

from pymc.step_methods.state import DataClassState, WithSamplingState, dataclass_state
from tests.helpers import equal_sampling_states


@dataclass_state
class State1(DataClassState):
    a: int
    b: float
    c: str
    d: np.ndarray
    e: list
    f: dict


@dataclass_state
class State2(DataClassState):
    mutable_field: float
    state1: State1
    extra_info1: np.ndarray = field(metadata={"frozen": True})
    extra_info2: list = field(metadata={"frozen": True})
    extra_info3: dict = field(metadata={"frozen": True})


class A(WithSamplingState):
    _state_class = State1

    def __init__(self, a=1, b=2.0, c="c", d=None, e=None, f=None):
        self.a = a
        self.b = b
        self.c = c
        if d is None:
            d = np.array([1, 2])
        if e is None:
            e = [1, 2, 3]
        if f is None:
            f = {"a": 1, "b": "c"}
        self.d = d
        self.e = e
        self.f = f


class B(WithSamplingState):
    _state_class = State2

    def __init__(
        self,
        a=1,
        b=2.0,
        c="c",
        d=None,
        e=None,
        f=None,
        mutable_field=1.0,
        extra_info1=None,
        extra_info2=None,
        extra_info3=None,
    ):
        self.state1 = A(a=a, b=b, c=c, d=d, e=e, f=f)
        self.mutable_field = mutable_field
        if extra_info1 is None:
            extra_info1 = np.array([3, 4, 5])
        if extra_info2 is None:
            extra_info2 = [5, 6, 7]
        if extra_info3 is None:
            extra_info3 = {"foo": "bar"}
        self.extra_info1 = extra_info1
        self.extra_info2 = extra_info2
        self.extra_info3 = extra_info3


@dataclass_state
class RngState(DataClassState):
    rng: np.random.Generator


class Step(WithSamplingState):
    _state_class = RngState

    def __init__(self, rng=None):
        self.rng = np.random.default_rng(rng)


def test_sampling_state():
    b1 = B()
    b2 = B(mutable_field=2.0)
    b3 = B(c=1, extra_info1=np.array([10, 20]))
    b4 = B(a=2, b=3.0, c="d")
    b5 = B(c=1)
    b6 = B(f={"a": 1, "b": "c", "d": None})

    b1_state = b1.sampling_state
    b2_state = b2.sampling_state
    b3_state = b3.sampling_state
    b4_state = b4.sampling_state

    assert equal_sampling_states(b1_state.state1, b2_state.state1)
    assert not equal_sampling_states(b1_state, b2_state)
    assert not equal_sampling_states(b1_state, b3_state)
    assert not equal_sampling_states(b1_state, b4_state)

    b1.sampling_state = b2_state
    assert equal_sampling_states(b1.sampling_state, b2_state)

    expected_error_message = (
        "The received sampling state must have the same values for the "
        "frozen fields. Field 'extra_info1' has different values. "
        r"Expected \[3 4 5\] but got \[10 20\]"
    )
    with pytest.raises(ValueError, match=expected_error_message):
        b1.sampling_state = b3_state

    with pytest.raises(AssertionError, match="Encountered invalid state class"):
        b1.sampling_state = b1_state.state1

    b1.sampling_state = b4_state
    assert equal_sampling_states(b1.sampling_state, b4_state)
    assert not equal_sampling_states(b1.sampling_state, b5.sampling_state)
    assert not equal_sampling_states(b1.sampling_state, b6.sampling_state)


@pytest.mark.parametrize(
    "step",
    [
        Step(),
        Step(1),
        Step(np.random.Generator(np.random.Philox(1))),
    ],
    ids=["default_rng", "default_rng(1)", "philox"],
)
def test_sampling_state_rng(step):
    original_state = step.sampling_state
    values1 = step.rng.random(100)

    final_state = step.sampling_state
    assert not equal_sampling_states(original_state, final_state)

    step.sampling_state = original_state
    values2 = step.rng.random(100)
    assert np.array_equal(values1, values2, equal_nan=True)
    assert equal_sampling_states(step.sampling_state, final_state)
