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

import numpy as np
import numpy.testing as npt
import pytest

import pymc as pm

from pymc.backends import base, ndarray
from tests.backends import fixtures as bf
from tests.helpers import equal_sampling_states

STATS1 = [{"a": np.float64, "b": bool}]

STATS2 = [
    {"a": np.float64},
    {
        "a": np.float64,
        "b": np.int64,
    },
]


class TestNDArray0dSampling(bf.SamplingTestCase):
    backend = ndarray.NDArray
    name = None
    shape = ()


class TestNDArray0dSamplingStats1(bf.SamplingTestCase):
    backend = ndarray.NDArray
    name = None
    sampler_vars = STATS1
    shape = ()


class TestNDArray0dSamplingStats2(bf.SamplingTestCase):
    backend = ndarray.NDArray
    name = None
    sampler_vars = STATS2
    shape = ()


class TestNDArray1dSampling(bf.SamplingTestCase):
    backend = ndarray.NDArray
    name = None
    shape = 2


class TestNDArray2dSampling(bf.SamplingTestCase):
    backend = ndarray.NDArray
    name = None
    shape = (2, 3)


class TestNDArrayStats(bf.StatsTestCase):
    backend = ndarray.NDArray
    name = None
    shape = (2, 3)


class TestNDArray0dSelection(bf.SelectionTestCase):
    backend = ndarray.NDArray
    name = None
    shape = ()
    sampler_vars = STATS1


class TestNDArray0dSelection2(bf.SelectionTestCase):
    backend = ndarray.NDArray
    name = None
    shape = ()
    sampler_vars = STATS2


class TestNDArray0dSelectionStats1(bf.SelectionTestCase):
    backend = ndarray.NDArray
    name = None
    shape = ()
    sampler_vars = STATS2


class TestNDArray0dSelectionStats2(bf.SelectionTestCase):
    backend = ndarray.NDArray
    name = None
    shape = ()


class TestNDArray1dSelection(bf.SelectionTestCase):
    backend = ndarray.NDArray
    name = None
    shape = 2


class TestNDArray2dSelection(bf.SelectionTestCase):
    backend = ndarray.NDArray
    name = None
    shape = (2, 3)


class TestMultiTrace(bf.ModelBackendSetupTestCase):
    name = None
    backend = ndarray.NDArray
    shape = ()

    def setup_method(self):
        super().setup_method()
        self.strace0 = self.strace

        super().setup_method()
        self.strace1 = self.strace

    def test_multitrace_nonunique(self):
        with pytest.raises(ValueError):
            base.MultiTrace([self.strace0, self.strace1])

    def test_multitrace_iter_notimplemented(self):
        mtrace = base.MultiTrace([self.strace0])
        with pytest.raises(NotImplementedError):
            for _ in mtrace:
                pass


class TestSqueezeCat:
    def setup_method(self):
        self.x = np.arange(10)
        self.y = np.arange(10, 20)

    def test_combine_false_squeeze_false(self):
        expected = [self.x, self.y]
        result = base._squeeze_cat([self.x, self.y], False, False)
        npt.assert_equal(result, expected)

    def test_combine_true_squeeze_false(self):
        expected = [np.concatenate([self.x, self.y])]
        result = base._squeeze_cat([self.x, self.y], True, False)
        npt.assert_equal(result, expected)

    def test_combine_false_squeeze_true_more_than_one_item(self):
        expected = [self.x, self.y]
        result = base._squeeze_cat([self.x, self.y], False, True)
        npt.assert_equal(result, expected)

    def test_combine_false_squeeze_true_one_item(self):
        expected = self.x
        result = base._squeeze_cat([self.x], False, True)
        npt.assert_equal(result, expected)

    def test_combine_true_squeeze_true(self):
        expected = np.concatenate([self.x, self.y])
        result = base._squeeze_cat([self.x, self.y], True, True)
        npt.assert_equal(result, expected)


class TestSamplingState:
    def setup_method(self):
        with pm.Model() as self.model:
            pm.Normal("x")
            self.step = pm.Metropolis()

    def test_nothing_recorded_without_step(self):
        trace = ndarray.NDArray(model=self.model)
        trace.record_sampling_state()
        assert trace.sampling_state is None

    def test_record_with_explicit_step(self):
        trace = ndarray.NDArray(model=self.model)
        trace.record_sampling_state(step=self.step)
        assert equal_sampling_states(trace.sampling_state, self.step.sampling_state)

    def test_record_with_linked_step(self):
        trace = ndarray.NDArray(model=self.model)
        trace.link_stepper(self.step)
        trace.record_sampling_state()
        assert equal_sampling_states(trace.sampling_state, self.step.sampling_state)

    def test_store_sampling_state(self):
        trace = ndarray.NDArray(model=self.model)
        state = self.step.sampling_state
        trace.store_sampling_state(state)
        assert trace.sampling_state is state

    def test_close_releases_step_method(self):
        trace = ndarray.NDArray(model=self.model)
        trace.setup(draws=1, chain=0)
        trace.link_stepper(self.step)
        trace.record_sampling_state()
        trace.close()
        assert trace._step_method is None
        assert trace.sampling_state is not None

    def test_slice_keeps_sampling_state(self):
        trace = ndarray.NDArray(model=self.model)
        trace.setup(draws=3, chain=0)
        for _ in range(3):
            trace.record({"x": np.array(0.0)}, in_warmup=False)
        trace.record_sampling_state(step=self.step)
        sliced = trace[1:]
        assert len(sliced) == 2
        assert sliced.sampling_state is trace.sampling_state

    @pytest.mark.parametrize("chains", [1, 2])
    def test_sample_records_final_state(self, chains):
        with self.model, pytest.warns(FutureWarning, match="return_inferencedata=False"):
            mtrace = pm.sample(
                draws=5,
                tune=5,
                chains=chains,
                cores=1,
                step=self.step,
                random_seed=42,
                return_inferencedata=False,
                compute_convergence_checks=False,
                progressbar=False,
            )
        states = [mtrace._straces[chain].sampling_state for chain in mtrace.chains]
        assert all(state is not None for state in states)
        # Sequential sampling leaves the step method in the final state of the last chain
        assert equal_sampling_states(states[-1], self.step.sampling_state)
        if chains > 1:
            assert not equal_sampling_states(states[0], states[1])

    def test_interrupted_sample_records_state(self):
        def callback(trace, draw):
            if len(trace) >= 3:
                raise KeyboardInterrupt()

        with self.model, pytest.warns(FutureWarning, match="return_inferencedata=False"):
            mtrace = pm.sample(
                draws=10,
                tune=0,
                chains=1,
                cores=1,
                step=self.step,
                random_seed=42,
                callback=callback,
                return_inferencedata=False,
                compute_convergence_checks=False,
                progressbar=False,
            )
        assert len(mtrace) == 3
        assert equal_sampling_states(mtrace._straces[0].sampling_state, self.step.sampling_state)
