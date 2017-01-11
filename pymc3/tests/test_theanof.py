import itertools
import unittest
import numpy as np
import theano
from ..theanof import DataGenerator, GeneratorOp


def integers():
    i = 0
    while True:
        yield np.float32(i)
        i += 1


def integers_ndim(ndim):
    i = 0
    while True:
        yield np.ones((2,) * ndim) * i
        i += 1


class TestGenerator(unittest.TestCase):
    def test_basic(self):
        generator = DataGenerator(integers())
        gop = GeneratorOp(generator)()
        self.assertEqual(gop.tag.test_value, np.float32(0))
        f = theano.function([], gop)
        self.assertEqual(f(), np.float32(0))
        self.assertEqual(f(), np.float32(1))
        for i in range(2, 100):
            f()
        self.assertEqual(f(), np.float32(100))

    def test_ndim(self):
        for ndim in range(10):
            res = list(itertools.islice(integers_ndim(ndim), 0, 2))
            generator = DataGenerator(integers_ndim(ndim))
            gop = GeneratorOp(generator)()
            f = theano.function([], gop)
            self.assertEqual(ndim, res[0].ndim)
            np.testing.assert_equal(f(), res[0])
            np.testing.assert_equal(f(), res[1])

    def test_cloning_available(self):
        generator = DataGenerator(integers())
        gop = GeneratorOp(generator)()
        res = gop ** 2
        shared = theano.shared(np.float32(10))
        res1 = theano.clone(res, {gop: shared})
        f = theano.function([], res1)
        self.assertEqual(f(), np.float32(100))
