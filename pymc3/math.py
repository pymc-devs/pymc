from __future__ import division
import sys
import theano.tensor as tt
# pylint: disable=unused-import
import theano
from theano.tensor import (
    constant, flatten, zeros_like, ones_like, stack, concatenate, sum, prod,
    lt, gt, le, ge, eq, neq, switch, clip, where, and_, or_, abs_, exp, log,
    cos, sin, tan, cosh, sinh, tanh, sqr, sqrt, erf, erfinv, dot, maximum,
    minimum, sgn, ceil, floor)
from theano.tensor.nlinalg import det, matrix_inverse, extract_diag, matrix_dot, trace
from theano.tensor.nnet import sigmoid
from theano.gof import Op, Apply
import numpy as np
# pylint: enable=unused-import


def logsumexp(x, axis=None):
    # Adapted from https://github.com/Theano/Theano/issues/1563
    x_max = tt.max(x, axis=axis, keepdims=True)
    return tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max


def invlogit(x, eps=sys.float_info.epsilon):
    return (1 - 2 * eps) / (1 + tt.exp(-x)) + eps


def logit(p):
    return tt.log(p / (1 - p))


class LogDet(Op):
    """Computes the logarithm of absolute determinant of a square
    matrix M, log(abs(det(M))), on CPU. Avoids det(M) overflow/
    underflow.

    Note: Once PR #3959 (https://github.com/Theano/Theano/pull/3959/) by harpone is merged,
    this must be removed. 
    """
    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs, params=None):
        try:
            (x,) = inputs
            (z,) = outputs
            s = np.linalg.svd(x, compute_uv=False)
            log_det = np.sum(np.log(np.abs(s)))
            z[0] = np.asarray(log_det, dtype=x.dtype)
        except Exception:
            print('Failed to compute logdet of {}.'.format(x))
            raise

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [x] = inputs
        return [gz * matrix_inverse(x).T]

    def __str__(self):
        return "LogDet"

logdet = LogDet()
