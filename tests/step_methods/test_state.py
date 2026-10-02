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

import pymc as pm

from pymc.step_methods import Metropolis
from pymc.step_methods.compound import CompoundStep, CompoundStepState
from pymc.step_methods.state import (
    DataClassState,
    WithSamplingState,
    dataclass_state,
)
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


@dataclass_state
class RngState(DataClassState):
    rng: np.random.Generator


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


def test_compound_state_row_roundtrip():
    """CompoundStep states store one struct field per member step."""
    from pymc.step_methods.state import (
        pickle_field_map,
        row_to_state,
        state_class_map,
        state_to_row,
    )

    with pm.Model() as model:
        a = pm.Normal("a")
        b = pm.Normal("b", a)
        step = CompoundStep(
            [
                Metropolis(vars=[a], model=model, rng=np.random.default_rng(1)),
                Metropolis(vars=[b], model=model, rng=np.random.default_rng(2)),
            ]
        )

    state = step.sampling_state
    dtype = state.struct_dtype()
    assert dtype.names == ("methods_0", "methods_1")
    assert dtype.fields["methods_0"][0].names is not None

    row = state_to_row(state)
    classes = state_class_map(state)
    assert classes["methods_0"] == "pymc.step_methods.metropolis.MetropolisState"
    restored = row_to_state(
        row, CompoundStepState, classes=classes, pickled=pickle_field_map(state)
    )
    assert equal_sampling_states(restored, state)
    # And members resume: the second member's stream continues from the snapshot
    restored_b = np.random.default_rng()
    restored_b.bit_generator.state = restored.methods[1].rng.bit_generator_state
    expected_stream = [step.methods[1].rng.random() for _ in range(2)]
    assert [restored_b.random() for _ in range(2)] == expected_stream


def test_state_row_roundtrip():
    from pymc.step_methods.state import pickle_field_map, row_to_state, state_to_row

    s = State1(a=1, b=2.0, c="c", d=np.array([1, 2]), e=[1, 2, 3], f={"a": 1})
    row = state_to_row(s)
    assert row.dtype == s.struct_dtype()
    pickled = pickle_field_map(s)
    assert pickled == {"c": "c", "e": [1, 2, 3], "f": {"a": 1}}
    assert equal_sampling_states(row_to_state(row, State1, pickled=pickled), s)

    # Pickled values roundtrip losslessly no matter their size
    s = State1(a=1, b=2.0, c="c" * 10_000, d=np.array([1, 2]), e=list(range(10_000)), f={"a": 1})
    pickled = pickle_field_map(s)
    assert equal_sampling_states(row_to_state(state_to_row(s), State1, pickled=pickled), s)

    # None pickled fields roundtrip as None
    s = State1(a=1, b=2.0, c=None, d=np.array([1, 2]), e=[1, 2, 3], f={"a": 1})
    restored = row_to_state(state_to_row(s), State1, pickled=pickle_field_map(s))
    assert restored.c is None
    assert equal_sampling_states(restored, s)

    # Nested states (and frozen fields) roundtrip
    b = B(a=1, b=2.0, c="c", d=np.array([1, 2]), e=[1, 2, 3], f={"a": 1})
    state = b.sampling_state
    pickled = pickle_field_map(state)
    assert "state1.c" in pickled
    assert "extra_info2" in pickled
    restored = row_to_state(state_to_row(state), State2, pickled=pickled)
    assert equal_sampling_states(restored, state)

    # Generators (stored as RandomGeneratorState) roundtrip and resume:
    # the restored generator must produce the same stream as the original
    # generator produced after the state was captured
    step = Step(np.random.default_rng(42))
    snapshot = step.sampling_state
    expected_stream = [step.rng.random() for _ in range(3)]
    restored = row_to_state(state_to_row(snapshot), RngState, pickled=pickle_field_map(snapshot))
    assert equal_sampling_states(restored, snapshot)
    rng = np.random.default_rng()
    rng.bit_generator.state = restored.rng.bit_generator_state
    assert [rng.random() for _ in range(3)] == expected_stream


def test_struct_dtype():
    s = State1(a=1, b=2.0, c="c", d=np.array([1, 2]), e=[1, 2, 3], f={"a": 1})
    dtype = s.struct_dtype()
    # Only fixed-size fields live in the struct row; pickled fields are stored
    # separately (they have no fixed size)
    assert dtype == np.dtype([("a", "i8"), ("b", "f8"), ("d", "i8", (2,))])

    # Nested states recurse into nested dtypes
    b = B(a=1, b=2.0, c="c", d=np.array([1, 2]), e=[1, 2, 3], f={"a": 1})
    nested = b.struct_dtype()
    assert nested.names == (
        "mutable_field",
        "state1",
        "extra_info1",
    )
    assert nested.fields["state1"][0] == dtype
    assert nested.fields["extra_info1"][0] == np.dtype(("i8", (3,)))

    # Generators become name + pickled state bytes + verified hash
    rng_state = RngState(rng=np.random.default_rng(42)).struct_dtype()
    rng_dtype = rng_state.fields["rng"][0]
    assert rng_dtype.names == ("bit_generator_name", "state", "state_hash")


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
