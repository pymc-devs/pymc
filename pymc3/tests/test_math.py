import numpy as np
import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt
from pymc3.math import (
    LogDet, logdet, probit, invprobit, expand_packed_triangular)
from .helpers import SeededTest
import pytest
from pymc3.theanof import floatX


def test_probit():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    np.testing.assert_allclose(invprobit(probit(p)).eval(), p, atol=1e-5)


class TestLogDet(SeededTest):
    def setup_method(self):
        super(TestLogDet, self).setup_method()
        utt.seed_rng()
        self.op_class = LogDet
        self.op = logdet

    @theano.configparser.change_flags(compute_test_value="ignore")
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

    @pytest.mark.skipif(theano.config.device in ["cuda", "gpu"],
                        reason="No logDet implementation on GPU.")
    def test_basic(self):
        # Calls validate with different params
        test_case_1 = np.random.randn(3, 3) / np.sqrt(3)
        test_case_2 = np.random.randn(10, 10) / np.sqrt(10)
        self.validate(test_case_1.astype(theano.config.floatX))
        self.validate(test_case_2.astype(theano.config.floatX))


def test_expand_packed_triangular():
    with pytest.raises(ValueError):
        x = tt.matrix('x')
        x.tag.test_value = np.array([[1.]])
        expand_packed_triangular(5, x)
    N = 5
    packed = tt.vector('packed')
    packed.tag.test_value = floatX(np.zeros(N * (N + 1) // 2))
    with pytest.raises(TypeError):
        expand_packed_triangular(packed.shape[0], packed)
    np.random.seed(42)
    vals = np.random.randn(N, N)
    lower = floatX(np.tril(vals))
    lower_packed = floatX(vals[lower != 0])
    upper = floatX(np.triu(vals))
    upper_packed = floatX(vals[upper != 0])
    expand_lower = expand_packed_triangular(N, packed, lower=True)
    expand_upper = expand_packed_triangular(N, packed, lower=False)
    expand_diag_lower = expand_packed_triangular(N, packed, lower=True, diagonal_only=True)
    expand_diag_upper = expand_packed_triangular(N, packed, lower=False, diagonal_only=True)
    assert np.all(expand_lower.eval({packed: lower_packed}) == lower)
    assert np.all(expand_upper.eval({packed: upper_packed}) == upper)
    assert np.all(expand_diag_lower.eval({packed: lower_packed}) == floatX(np.diag(vals)))
    assert np.all(expand_diag_upper.eval({packed: upper_packed}) == floatX(np.diag(vals)))
