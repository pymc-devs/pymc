from __future__ import division
import sys
import theano
import theano.tensor as tt
from theano.tensor import (constant, flatten, zeros_like, ones_like, stack, concatenate, sum, prod,
                           lt, gt, le, ge, eq, neq, switch, clip, where, and_, or_, abs_, exp, log,
                           cos, sin, tan, cosh, sinh, tanh, sqr, sqrt, erf, erfinv, dot, maximum,
                           minimum, sgn, ceil, floor)
from theano.tensor.nlinalg import det, matrix_inverse, extract_diag, matrix_dot, trace
from theano.tensor.nnet import sigmoid


def logsumexp(x, axis=None):
    # Adapted from https://github.com/Theano/Theano/issues/1563
    x_max = tt.max(x, axis=axis, keepdims=True)
    return tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max


def invlogit(x, eps=sys.float_info.epsilon):
    return (1 - 2 * eps) / (1 + tt.exp(-x)) + eps


def logit(p):
    return tt.log(p / (1 - p))
