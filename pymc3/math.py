from __future__ import division
from theano.tensor import constant, flatten, zeros_like, ones_like, stack, concatenate, sum, prod, lt, gt, le, ge, eq, neq, switch, clip, where, and_, or_, abs_

from theano.tensor import exp, log, cos, sin, tan, cosh, sinh, tanh, sqr, sqrt, erf, erfinv, dot
from theano.tensor import maximum, minimum, sgn, ceil, floor

from theano.tensor.nlinalg import det, matrix_inverse, extract_diag, matrix_dot, trace
from theano.tensor.nnet import sigmoid
import theano
import theano.tensor as tt

def LogSumExp(x, axis=None):
    # Adapted from https://github.com/Theano/Theano/issues/1563
    x_max = tt.max(x, axis=axis, keepdims=True)
    return tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max
