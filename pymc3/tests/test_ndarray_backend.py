import unittest
import numpy as np
import numpy.testing as npt
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import base, ndarray


STATS1 = [{
    'a': np.float64,
    'b': np.bool
}]

STATS2 = [{
    'a': np.float64
}, {
    'a': np.float64,
    'b': np.int64,
}]


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


class TestNDArray0dSelection(bf.SelectionTestCase):
    backend = ndarray.NDArray
    name = None
    shape = ()
    sampler_vars = STATS2


class TestNDArray0dSelection(bf.SelectionTestCase):
    backend = ndarray.NDArray
    name = None
    shape = ()
    sampler_vars = STATS1


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

    def setUp(self):
        super(TestMultiTrace, self).setUp()
        self.strace0 = self.strace

        super(TestMultiTrace, self).setUp()
        self.strace1 = self.strace

    def test_multitrace_nonunique(self):
        self.assertRaises(ValueError,
                          base.MultiTrace, [self.strace0, self.strace1])

    def test_merge_traces_nonunique(self):
        mtrace0 = base.MultiTrace([self.strace0])
        mtrace1 = base.MultiTrace([self.strace1])

        self.assertRaises(ValueError,
                          base.merge_traces, [mtrace0, mtrace1])


class TestSqueezeCat(unittest.TestCase):

    def setUp(self):
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
