import numpy as np
import theano
from theano.tests import unittest_tools as utt
from pymc3.math import LogDet, logdet
from .helpers import SeededTest

class TestLogDet(SeededTest):

    def setUp(self):
        utt.seed_rng()
        self.op_class = LogDet
        self.op = logdet

    def validate(self, input_mat):
        x = theano.tensor.matrix()
        f = theano.function([x], self.op(x))
        out = f(input_mat)
        svd_diag = np.linalg.svd(input_mat, compute_uv=False)
        numpy_out = np.sum(np.log(np.abs(svd_diag)))

        # Compare the result computed to the expected value.
        utt.assert_allclose(numpy_out, out)

        # Test gradient:
        utt.verify_grad(self.op, [input_mat])

    def test_basic(self):
        # Calls validate with different params
        test_case_1 = np.random.randn(3, 3) / np.sqrt(3)
        test_case_2 = np.random.randn(10, 10) / np.sqrt(10)
        self.validate(test_case_1.astype(theano.config.floatX))
        self.validate(test_case_2.astype(theano.config.floatX))
