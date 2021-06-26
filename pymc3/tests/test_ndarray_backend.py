#   Copyright 2020 The PyMC Developers
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

import pymc3 as pm

from pymc3.backends import base, ndarray
from pymc3.tests import backend_fixtures as bf

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

    def test_merge_traces_no_traces(self):
        with pytest.raises(ValueError):
            base.merge_traces([])

    def test_merge_traces_diff_lengths(self):
        with self.model:
            strace0 = self.backend(self.name)
            strace0.setup(self.draws, 1)
            for i in range(self.draws):
                strace0.record(self.test_point)
            strace0.close()
        mtrace0 = base.MultiTrace([self.strace0])

        with self.model:
            strace1 = self.backend(self.name)
            strace1.setup(2 * self.draws, 1)
            for i in range(2 * self.draws):
                strace1.record(self.test_point)
            strace1.close()
        mtrace1 = base.MultiTrace([strace1])

        with pytest.raises(ValueError):
            base.merge_traces([mtrace0, mtrace1])

    def test_merge_traces_nonunique(self):
        mtrace0 = base.MultiTrace([self.strace0])
        mtrace1 = base.MultiTrace([self.strace1])

        with pytest.raises(ValueError):
            base.merge_traces([mtrace0, mtrace1])


class TestMultiTrace_add_remove_values(bf.ModelBackendSampledTestCase):
    name = None
    backend = ndarray.NDArray
    shape = ()

    def test_add_values(self):
        mtrace = self.mtrace
        orig_varnames = list(mtrace.varnames)
        name = "new_var"
        vals = mtrace[orig_varnames[0]]
        mtrace.add_values({name: vals})
        assert len(orig_varnames) == len(mtrace.varnames) - 1
        assert name in mtrace.varnames
        assert np.all(mtrace[orig_varnames[0]] == mtrace[name])
        mtrace.remove_values(name)
        assert len(orig_varnames) == len(mtrace.varnames)
        assert name not in mtrace.varnames


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


class TestSaveLoad:
    @staticmethod
    def model():
        with pm.Model() as model:
            x = pm.Normal("x", 0, 1)
            y = pm.Normal("y", x, 1, observed=2)
            z = pm.Normal("z", x + y, 1)
        return model

    @classmethod
    def setup_class(cls):
        with TestSaveLoad.model():
            cls.trace = pm.sample(return_inferencedata=False)

    def test_save_new_model(self, tmpdir_factory):
        directory = str(tmpdir_factory.mktemp("data"))
        save_dir = pm.save_trace(self.trace, directory, overwrite=True)

        assert save_dir == directory
        with pm.Model() as model:
            w = pm.Normal("w", 0, 1)
            new_trace = pm.sample(return_inferencedata=False)

        with pytest.raises(OSError):
            _ = pm.save_trace(new_trace, directory)

        _ = pm.save_trace(new_trace, directory, overwrite=True)
        with model:
            new_trace_copy = pm.load_trace(directory)

        assert (new_trace["w"] == new_trace_copy["w"]).all()

    def test_save_and_load(self, tmpdir_factory):
        directory = str(tmpdir_factory.mktemp("data"))
        save_dir = pm.save_trace(self.trace, directory, overwrite=True)

        assert save_dir == directory

        trace2 = pm.load_trace(directory, model=TestSaveLoad.model())

        for var in ("x", "z"):
            assert (self.trace[var] == trace2[var]).all()

        assert self.trace.stat_names == trace2.stat_names
        for stat in self.trace.stat_names:
            assert all(self.trace[stat] == trace2[stat]), (
                "Restored value of statistic %s does not match stored value" % stat
            )

    def test_bad_load(self, tmpdir_factory):
        directory = str(tmpdir_factory.mktemp("data"))
        with pytest.raises(pm.TraceDirectoryError):
            pm.load_trace(directory, model=TestSaveLoad.model())

    def test_sample_posterior_predictive(self, tmpdir_factory):
        directory = str(tmpdir_factory.mktemp("data"))
        save_dir = pm.save_trace(self.trace, directory, overwrite=True)

        assert save_dir == directory

        seed = 10
        np.random.seed(seed)
        with TestSaveLoad.model():
            ppc = pm.sample_posterior_predictive(self.trace)
            ppcf = pm.fast_sample_posterior_predictive(self.trace)

        seed = 10
        np.random.seed(seed)
        with TestSaveLoad.model():
            trace2 = pm.load_trace(directory)
            ppc2 = pm.sample_posterior_predictive(trace2)
            ppc2f = pm.sample_posterior_predictive(trace2)

        for key, value in ppc.items():
            assert (value == ppc2[key]).all()

        for key, value in ppcf.items():
            assert (value == ppc2f[key]).all()
