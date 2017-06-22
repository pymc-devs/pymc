from __future__ import division
import sys
import theano.tensor as tt
# pylint: disable=unused-import
import theano
from theano.tensor import (
    constant, flatten, zeros_like, ones_like, stack, concatenate, sum, prod,
    lt, gt, le, ge, eq, neq, switch, clip, where, and_, or_, abs_, exp, log,
    cos, sin, tan, cosh, sinh, tanh, sqr, sqrt, erf, erfc, erfinv, erfcinv, dot,
    maximum, minimum, sgn, ceil, floor)
from theano.tensor.nlinalg import det, matrix_inverse, extract_diag, matrix_dot, trace
from theano.tensor.nnet import sigmoid
from theano.gof import Op, Apply
import numpy as np
from pymc3.theanof import floatX

# pylint: enable=unused-import


def tround(*args, **kwargs):
    """
    Temporary function to silence round warning in Theano. Please remove
    when the warning disappears.
    """
    kwargs['mode'] = 'half_to_even'
    return tt.round(*args, **kwargs)


def logsumexp(x, axis=None):
    # Adapted from https://github.com/Theano/Theano/issues/1563
    x_max = tt.max(x, axis=axis, keepdims=True)
    return tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def logaddexp(a, b):
    diff = b - a
    return tt.switch(diff > 0, 
                    b + tt.log1p(tt.exp(-diff)), 
                    a + tt.log1p(tt.exp(diff)))

def invlogit(x, eps=sys.float_info.epsilon):
    return (1. - 2. * eps) / (1. + tt.exp(-x)) + eps


def logit(p):
    return tt.log(p / (floatX(1) - p))


def flatten_list(tensors):
    return tt.concatenate([var.ravel() for var in tensors])


class LogDet(Op):
    """Compute the logarithm of the absolute determinant of a square
    matrix M, log(abs(det(M))) on the CPU. Avoids det(M) overflow/
    underflow.

    Note
    ----
    Once PR #3959 (https://github.com/Theano/Theano/pull/3959/) by harpone is merged,
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


def probit(p):
    return -sqrt(2.) * erfcinv(2. * p)


def invprobit(x):
    return .5 * erfc(-x / sqrt(2.))


def expand_packed_triangular(n, packed, lower=True, diagonal_only=False):
    R"""Convert a packed triangular matrix into a two dimensional array.

    Triangular matrices can be stored with better space efficiancy by
    storing the non-zero values in a one-dimensional array. We number
    the elements by row like this (for lower or upper triangular matrices)::

        [[0 - - -]     [[0 1 2 3]
         [1 2 - -]      [- 4 5 6]
         [3 4 5 -]      [- - 7 8]
         [6 7 8 9]]     [- - - 9]

    Parameters
    ----------
    n : int
        The number of rows of the triangular matrix.
    packed : theano.vector
        The matrix in packed format.
    lower : bool, default=True
        If true, assume that the matrix is lower triangular.
    diagonal_only : bool
        If true, return only the diagonal of the matrix.
    """
    if packed.ndim != 1:
        raise ValueError('Packed triagular is not one dimensional.')
    if not isinstance(n, int):
        raise TypeError('n must be an integer')

    if diagonal_only and lower:
        diag_idxs = np.arange(1, n + 1).cumsum() - 1
        return packed[diag_idxs]
    elif diagonal_only and not lower:
        diag_idxs = np.arange(2, n + 2)[::-1].cumsum() - n - 1
        return packed[diag_idxs]
    elif lower:
        out = tt.zeros((n, n), dtype=theano.config.floatX)
        idxs = np.tril_indices(n)
        return tt.set_subtensor(out[idxs], packed)
    elif not lower:
        out = tt.zeros((n, n), dtype=theano.config.floatX)
        idxs = np.triu_indices(n)
        return tt.set_subtensor(out[idxs], packed)
