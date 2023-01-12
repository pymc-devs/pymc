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

from pymc.backends import base, ndarray
from pymc.tests.backends import fixtures as bf

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
