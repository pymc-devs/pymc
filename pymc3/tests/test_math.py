import numpy as np
import theano
from theano.tests import unittest_tools as utt
from pymc3.math import LogDet, logdet, probit, invprobit
from .helpers import SeededTest


def test_probit():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(invprobit(probit(p)).eval(), p, atol=1e-5)


class TestLogDet(SeededTest):
    def setup_method(self):
        super(TestLogDet, self).setup_method()
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
