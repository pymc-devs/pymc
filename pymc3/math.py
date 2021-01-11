#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import sys

from functools import partial, reduce

import numpy as np
import scipy as sp
import scipy.sparse  # pylint: disable=unused-import
import theano
import theano.sparse
import theano.tensor as tt
import theano.tensor.slinalg  # pylint: disable=unused-import

from scipy.linalg import block_diag as scipy_block_diag
from theano.graph.basic import Apply
from theano.graph.op import Op

# pylint: disable=unused-import
from theano.tensor import (
    abs_,
    and_,
    ceil,
    clip,
    concatenate,
    constant,
    cos,
    cosh,
    dot,
    eq,
    erf,
    erfc,
    erfcinv,
    erfinv,
    exp,
    flatten,
    floor,
    ge,
    gt,
    le,
    log,
    lt,
    maximum,
    minimum,
    neq,
    ones_like,
    or_,
    prod,
    sgn,
    sin,
    sinh,
    sqr,
    sqrt,
    stack,
    sum,
    switch,
    tan,
    tanh,
    where,
    zeros_like,
)
from theano.tensor.nlinalg import det, extract_diag, matrix_dot, matrix_inverse, trace
from theano.tensor.nnet import sigmoid

from pymc3.theanof import floatX, ix_, largest_common_dtype

# pylint: enable=unused-import


def kronecker(*Ks):
    r"""Return the Kronecker product of arguments:
          :math:`K_1 \otimes K_2 \otimes ... \otimes K_D`

    Parameters
    ----------
    Ks : Iterable of 2D array-like
        Arrays of which to take the product.

    Returns
    -------
    np.ndarray :
        Block matrix Kroncker product of the argument matrices.
    """
    return reduce(tt.slinalg.kron, Ks)


def cartesian(*arrays):
    """Makes the Cartesian product of arrays.

    Parameters
    ----------
    arrays: 1D array-like
            1D arrays where earlier arrays loop more slowly than later ones
    """
    N = len(arrays)
    return np.stack(np.meshgrid(*arrays, indexing="ij"), -1).reshape(-1, N)


def kron_matrix_op(krons, m, op):
    r"""Apply op to krons and m in a way that reproduces ``op(kronecker(*krons), m)``

    Parameters
    -----------
    krons : list of square 2D array-like objects
        D square matrices :math:`[A_1, A_2, ..., A_D]` to be Kronecker'ed
        :math:`A = A_1 \otimes A_2 \otimes ... \otimes A_D`
        Product of column dimensions must be :math:`N`
    m : NxM array or 1D array (treated as Nx1)
        Object that krons act upon

    Returns
    -------
    numpy array
    """

    def flat_matrix_op(flat_mat, mat):
        Nmat = mat.shape[1]
        flat_shape = flat_mat.shape
        mat2 = flat_mat.reshape((Nmat, -1))
        return op(mat, mat2).T.reshape(flat_shape)

    def kron_vector_op(v):
        return reduce(flat_matrix_op, krons, v)

    if m.ndim == 1:
        m = m[:, None]  # Treat 1D array as Nx1 matrix
    if m.ndim != 2:  # Has not been tested otherwise
        raise ValueError(f"m must have ndim <= 2, not {m.ndim}")
    res = kron_vector_op(m)
    res_shape = res.shape
    return tt.reshape(res, (res_shape[1], res_shape[0])).T


# Define kronecker functions that work on 1D and 2D arrays
kron_dot = partial(kron_matrix_op, op=tt.dot)
kron_solve_lower = partial(kron_matrix_op, op=tt.slinalg.solve_lower_triangular)
kron_solve_upper = partial(kron_matrix_op, op=tt.slinalg.solve_upper_triangular)


def flat_outer(a, b):
    return tt.outer(a, b).ravel()


def kron_diag(*diags):
    """Returns diagonal of a kronecker product.

    Parameters
    ----------
    diags: 1D arrays
           The diagonals of matrices that are to be Kroneckered
    """
    return reduce(flat_outer, diags)


def tround(*args, **kwargs):
    """
    Temporary function to silence round warning in Theano. Please remove
    when the warning disappears.
    """
    kwargs["mode"] = "half_to_even"
    return tt.round(*args, **kwargs)


def logsumexp(x, axis=None, keepdims=True):
    # Adapted from https://github.com/Theano/Theano/issues/1563
    x_max = tt.max(x, axis=axis, keepdims=True)
    x_max = tt.switch(tt.isinf(x_max), 0, x_max)
    res = tt.log(tt.sum(tt.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return res if keepdims else res.squeeze()


def logaddexp(a, b):
    diff = b - a
    return tt.switch(diff > 0, b + tt.log1p(tt.exp(-diff)), a + tt.log1p(tt.exp(diff)))


def logdiffexp(a, b):
    """log(exp(a) - exp(b))"""
    return a + log1mexp(a - b)


def logdiffexp_numpy(a, b):
    """log(exp(a) - exp(b))"""
    return a + log1mexp_numpy(a - b)


def invlogit(x, eps=sys.float_info.epsilon):
    """The inverse of the logit function, 1 / (1 + exp(-x))."""
    return (1.0 - 2.0 * eps) / (1.0 + tt.exp(-x)) + eps


def logbern(log_p):
    if np.isnan(log_p):
        raise FloatingPointError("log_p can't be nan.")
    return np.log(np.random.uniform()) < log_p


def logit(p):
    return tt.log(p / (floatX(1) - p))


def log1pexp(x):
    """Return log(1 + exp(x)), also called softplus.

    This function is numerically more stable than the naive approach.
    """
    return tt.nnet.softplus(x)


def log1mexp(x):
    r"""Return log(1 - exp(-x)).

    This function is numerically more stable than the naive approach.

    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    References
        ----------
        .. [Machler2012] Martin Mächler (2012).
            "Accurately computing `\log(1-\exp(- \mid a \mid))` Assessed by the Rmpfr
            package"

    """
    return tt.switch(tt.lt(x, 0.6931471805599453), tt.log(-tt.expm1(-x)), tt.log1p(-tt.exp(-x)))


def log1mexp_numpy(x):
    """Return log(1 - exp(-x)).
    This function is numerically more stable than the naive approach.
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    return np.where(x < 0.6931471805599453, np.log(-np.expm1(-x)), np.log1p(-np.exp(-x)))


def flatten_list(tensors):
    return tt.concatenate([var.ravel() for var in tensors])


class LogDet(Op):
    r"""Compute the logarithm of the absolute determinant of a square
    matrix M, log(abs(det(M))) on the CPU. Avoids det(M) overflow/
    underflow.

    Notes
    -----
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
            print(f"Failed to compute logdet of {x}.")
            raise

    def grad(self, inputs, g_outputs):
        [gz] = g_outputs
        [x] = inputs
        return [gz * matrix_inverse(x).T]

    def __str__(self):
        return "LogDet"


logdet = LogDet()


def probit(p):
    return -sqrt(2.0) * erfcinv(2.0 * p)


def invprobit(x):
    return 0.5 * erfc(-x / sqrt(2.0))


def expand_packed_triangular(n, packed, lower=True, diagonal_only=False):
    r"""Convert a packed triangular matrix into a two dimensional array.

    Triangular matrices can be stored with better space efficiancy by
    storing the non-zero values in a one-dimensional array. We number
    the elements by row like this (for lower or upper triangular matrices):

        [[0 - - -]     [[0 1 2 3]
         [1 2 - -]      [- 4 5 6]
         [3 4 5 -]      [- - 7 8]
         [6 7 8 9]]     [- - - 9]

    Parameters
    ----------
    n: int
        The number of rows of the triangular matrix.
    packed: theano.vector
        The matrix in packed format.
    lower: bool, default=True
        If true, assume that the matrix is lower triangular.
    diagonal_only: bool
        If true, return only the diagonal of the matrix.
    """
    if packed.ndim != 1:
        raise ValueError("Packed triagular is not one dimensional.")
    if not isinstance(n, int):
        raise TypeError("n must be an integer")

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


class BatchedDiag(Op):
    """
    Fast BatchedDiag allocation
    """

    __props__ = ()

    def make_node(self, diag):
        diag = tt.as_tensor_variable(diag)
        if diag.type.ndim != 2:
            raise TypeError("data argument must be a matrix", diag.type)

        return Apply(self, [diag], [tt.tensor3(dtype=diag.dtype)])

    def perform(self, node, ins, outs, params=None):
        (C,) = ins
        (z,) = outs

        bc = C.shape[0]
        dim = C.shape[-1]
        Cd = np.zeros((bc, dim, dim), C.dtype)
        bidx = np.repeat(np.arange(bc), dim)
        didx = np.tile(np.arange(dim), bc)
        Cd[bidx, didx, didx] = C.flatten()
        z[0] = Cd

    def grad(self, inputs, gout):
        (gz,) = gout
        idx = tt.arange(gz.shape[-1])
        return [gz[..., idx, idx]]

    def infer_shape(self, fgraph, nodes, shapes):
        return [(shapes[0][0],) + (shapes[0][1],) * 2]


def batched_diag(C):
    C = tt.as_tensor(C)
    dim = C.shape[-1]
    if C.ndim == 2:
        # diag -> matrices
        return BatchedDiag()(C)
    elif C.ndim == 3:
        # matrices -> diag
        idx = tt.arange(dim)
        return C[..., idx, idx]
    else:
        raise ValueError("Input should be 2 or 3 dimensional")


class BlockDiagonalMatrix(Op):
    __props__ = ("sparse", "format")

    def __init__(self, sparse=False, format="csr"):
        if format not in ("csr", "csc"):
            raise ValueError(f"format must be one of: 'csr', 'csc', got {format}")
        self.sparse = sparse
        self.format = format

    def make_node(self, *matrices):
        if not matrices:
            raise ValueError("no matrices to allocate")
        matrices = list(map(tt.as_tensor, matrices))
        if any(mat.type.ndim != 2 for mat in matrices):
            raise TypeError("all data arguments must be matrices")
        if self.sparse:
            out_type = theano.sparse.matrix(self.format, dtype=largest_common_dtype(matrices))
        else:
            out_type = theano.tensor.matrix(dtype=largest_common_dtype(matrices))
        return Apply(self, matrices, [out_type])

    def perform(self, node, inputs, output_storage, params=None):
        dtype = largest_common_dtype(inputs)
        if self.sparse:
            output_storage[0][0] = sp.sparse.block_diag(inputs, self.format, dtype)
        else:
            output_storage[0][0] = scipy_block_diag(*inputs).astype(dtype)

    def grad(self, inputs, gout):
        shapes = tt.stack([i.shape for i in inputs])
        index_end = shapes.cumsum(0)
        index_begin = index_end - shapes
        slices = [
            ix_(
                tt.arange(index_begin[i, 0], index_end[i, 0]),
                tt.arange(index_begin[i, 1], index_end[i, 1]),
            )
            for i in range(len(inputs))
        ]
        return [gout[0][slc] for slc in slices]

    def infer_shape(self, fgraph, nodes, shapes):
        first, second = zip(*shapes)
        return [(tt.add(*first), tt.add(*second))]


def block_diagonal(matrices, sparse=False, format="csr"):
    r"""See scipy.sparse.block_diag or
    scipy.linalg.block_diag for reference

    Parameters
    ----------
    matrices: tensors
    format: str (default 'csr')
        must be one of: 'csr', 'csc'
    sparse: bool (default False)
        if True return sparse format

    Returns
    -------
    matrix
    """
    if len(matrices) == 1:  # graph optimization
        return matrices[0]
    return BlockDiagonalMatrix(sparse=sparse, format=format)(*matrices)
