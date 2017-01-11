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
