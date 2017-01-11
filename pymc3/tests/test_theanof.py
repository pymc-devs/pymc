import itertools
import unittest
import numpy as np
import theano
from ..theanof import DataGenerator, GeneratorOp


class TestGenerator(unittest.TestCase):
    def test_basic(self):
        def integers():
            i = 0
            while True:
                yield np.float32(i)
                i += 1
        generator = DataGenerator(integers())
        gop = GeneratorOp(generator)()
        f = theano.function([], gop)
        self.assertEqual(f(), np.float32(0))
        self.assertEqual(f(), np.float32(1))

    def test_ndim(self):
        for ndim in range(10):
            def integers():
                i = 0
                while True:
                    yield np.ones((2,) * ndim) * i
                    i += 1
            res = list(itertools.islice(integers(), 0, 2))
            generator = DataGenerator(integers())
            gop = GeneratorOp(generator)()
            f = theano.function([], gop)
            self.assertEqual(ndim, res[0].ndim)
            np.testing.assert_equal(f(), res[0])
            np.testing.assert_equal(f(), res[1])
