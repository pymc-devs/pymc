#   Copyright 2024 The PyMC Developers
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
import warnings

from functools import partial, reduce

import numpy as np
import pytensor
import pytensor.sparse
import pytensor.tensor as pt
import pytensor.tensor.slinalg

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor import (
    abs,
    and_,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    broadcast_to,
    ceil,
    clip,
    concatenate,
    constant,
    cos,
    cosh,
    cumprod,
    cumsum,
    dot,
    eq,
    erf,
    erfc,
    erfcinv,
    erfinv,
    exp,
    flatten,
    floor,
    full,
    full_like,
    ge,
    gt,
    le,
    log,
    log1pexp,
    logaddexp,
    logsumexp,
    lt,
    matmul,
    max,
    maximum,
    mean,
    min,
    minimum,
    neq,
    ones,
    ones_like,
    or_,
    prod,
    round,
    sgn,
    sigmoid,
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
    zeros,
    zeros_like,
)
from pytensor.tensor.linalg import solve_triangular
from pytensor.tensor.nlinalg import matrix_inverse
from pytensor.tensor.special import log_softmax, softmax

from pymc.pytensorf import floatX

__all__ = [
    "abs",
    "and_",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "broadcast_to",
    "ceil",
    "clip",
    "concatenate",
    "constant",
    "cos",
    "cosh",
    "cumprod",
    "cumsum",
    "dot",
    "eq",
    "erf",
    "erfc",
    "erfcinv",
    "erfinv",
    "exp",
    "full",
    "full_like",
    "flatten",
    "floor",
    "ge",
    "gt",
    "le",
    "log",
    "log1pexp",
    "logaddexp",
    "logsumexp",
    "lt",
    "matmul",
    "max",
    "maximum",
    "mean",
    "min",
    "minimum",
    "neq",
    "ones",
    "ones_like",
    "or_",
    "prod",
    "round",
    "sgn",
    "sigmoid",
    "sin",
    "sinh",
    "sqr",
    "sqrt",
    "stack",
    "sum",
    "switch",
    "tan",
    "tanh",
    "where",
    "zeros",
    "zeros_like",
    "kronecker",
    "cartesian",
    "kron_dot",
    "kron_solve_lower",
    "kron_solve_upper",
    "kron_diag",
    "flat_outer",
    "logdiffexp",
    "invlogit",
    "softmax",
    "log_softmax",
    "logbern",
    "logit",
    "log1mexp",
    "flatten_list",
    "logdet",
    "probit",
    "invprobit",
    "expand_packed_triangular",
    "batched_diag",
    "block_diagonal",
    "round",
]


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
    return reduce(pt.slinalg.kron, Ks)


def cartesian(*arrays):
    """Makes the Cartesian product of arrays.

    Parameters
    ----------
    arrays: N-D array-like
            N-D arrays where earlier arrays loop more slowly than later ones
    """
    N = len(arrays)
    arrays_np = [np.asarray(x) for x in arrays]
    arrays_2d = [x[:, None] if np.asarray(x).ndim == 1 else x for x in arrays_np]
    arrays_integer = [np.arange(len(x)) for x in arrays_2d]
    product_integers = np.stack(np.meshgrid(*arrays_integer, indexing="ij"), -1).reshape(-1, N)
    return np.concatenate(
        [array[product_integers[:, i]] for i, array in enumerate(arrays_2d)], axis=-1
    )


def kron_matrix_op(krons, m, op):
    r"""Apply op to krons and m in a way that reproduces ``op(kronecker(*krons), m)``

    Parameters
    ----------
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
    result = kron_vector_op(m)
    result_shape = result.shape
    return pt.reshape(result, (result_shape[1], result_shape[0])).T


# Define kronecker functions that work on 1D and 2D arrays
kron_dot = partial(kron_matrix_op, op=pt.dot)
kron_solve_lower = partial(kron_matrix_op, op=partial(solve_triangular, lower=True))
kron_solve_upper = partial(kron_matrix_op, op=partial(solve_triangular, lower=False))


def flat_outer(a, b):
    return pt.outer(a, b).ravel()


def kron_diag(*diags):
    """Returns diagonal of a kronecker product.

    Parameters
    ----------
    diags: 1D arrays
           The diagonals of matrices that are to be Kroneckered
    """
    return reduce(flat_outer, diags)


def logdiffexp(a, b):
    """log(exp(a) - exp(b))"""
    return a + pt.log1mexp(b - a)


def logdiffexp_numpy(a, b):
    """log(exp(a) - exp(b))"""
    warnings.warn(
        "pymc.math.logdiffexp_numpy is being deprecated.",
        FutureWarning,
        stacklevel=2,
    )
    return a + log1mexp_numpy(b - a, negative_input=True)


invlogit = sigmoid


def logbern(log_p):
    if np.isnan(log_p):
        raise FloatingPointError("log_p can't be nan.")
    return np.log(np.random.uniform()) < log_p


def logit(p):
    return pt.log(p / (floatX(1) - p))


def log1mexp(x, *, negative_input=False):
    r"""Return log(1 - exp(-x)).

    This function is numerically more stable than the naive approach.

    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf

    References
    ----------
    .. [Machler2012] Martin Mächler (2012).
       "Accurately computing `\log(1-\exp(- \mid a \mid))` Assessed by the Rmpfr package"

    """
    if not negative_input:
        warnings.warn(
            "pymc.math.log1mexp will expect a negative input in a future "
            "version of PyMC.\n To suppress this warning set `negative_input=True`",
            FutureWarning,
            stacklevel=2,
        )
        x = -x

    return pt.log1mexp(x)


def log1mexp_numpy(x, *, negative_input=False):
    """Return log(1 - exp(x)).
    This function is numerically more stable than the naive approach.
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    warnings.warn(
        "pymc.math.log1mexp_numpy is being deprecated.",
        FutureWarning,
        stacklevel=2,
    )
    x = np.asarray(x, dtype="float")

    if not negative_input:
        warnings.warn(
            "pymc.math.log1mexp_numpy will expect a negative input in a future "
            "version of PyMC.\n To suppress this warning set `negative_input=True`",
            FutureWarning,
            stacklevel=2,
        )
        x = -x

    out = np.empty_like(x)
    mask = x < -0.6931471805599453  # log(1/2)
    out[mask] = np.log1p(-np.exp(x[mask]))
    mask = ~mask
    out[mask] = np.log(-np.expm1(x[mask]))
    return out


def flatten_list(tensors):
    return pt.concatenate([var.ravel() for var in tensors])


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
        x = pytensor.tensor.as_tensor_variable(x)
        o = pytensor.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs, params=None):
        try:
            (x,) = inputs
            (z,) = outputs
            s = np.linalg.svd(x, compute_uv=False)
            log_det = np.sum(np.log(np.abs(s)))
            z[0] = np.asarray(log_det, dtype=x.dtype)
        except Exception:
            print(f"Failed to compute logdet of {x}.", file=sys.stdout)
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

    Triangular matrices can be stored with better space efficiency by
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
    packed: pytensor.vector
        The matrix in packed format.
    lower: bool, default=True
        If true, assume that the matrix is lower triangular.
    diagonal_only: bool
        If true, return only the diagonal of the matrix.
    """
    if packed.ndim != 1:
        raise ValueError("Packed triangular is not one dimensional.")
    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    if diagonal_only and lower:
        diag_idxs = np.arange(1, n + 1).cumsum() - 1
        return packed[diag_idxs]
    elif diagonal_only and not lower:
        diag_idxs = np.arange(2, n + 2)[::-1].cumsum() - n - 1
        return packed[diag_idxs]
    elif lower:
        out = pt.zeros((n, n), dtype=pytensor.config.floatX)
        idxs = np.tril_indices(n)
        # tag as lower triangular to enable pytensor rewrites
        out = pt.set_subtensor(out[idxs], packed)
        out.tag.lower_triangular = True
        return out
    elif not lower:
        out = pt.zeros((n, n), dtype=pytensor.config.floatX)
        idxs = np.triu_indices(n)
        # tag as upper triangular to enable pytensor rewrites
        out = pt.set_subtensor(out[idxs], packed)
        out.tag.upper_triangular = True
        return out


class BatchedDiag(Op):
    """
    Fast BatchedDiag allocation
    """

    __props__ = ()

    def make_node(self, diag):
        diag = pt.as_tensor_variable(diag)
        if diag.type.ndim != 2:
            raise TypeError("data argument must be a matrix", diag.type)

        return Apply(self, [diag], [pt.tensor3(dtype=diag.dtype)])

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
        idx = pt.arange(gz.shape[-1])
        return [gz[..., idx, idx]]

    def infer_shape(self, fgraph, nodes, shapes):
        return [(shapes[0][0],) + (shapes[0][1],) * 2]


def batched_diag(C):
    C = pt.as_tensor(C)
    dim = C.shape[-1]
    if C.ndim == 2:
        # diag -> matrices
        return BatchedDiag()(C)
    elif C.ndim == 3:
        # matrices -> diag
        idx = pt.arange(dim)
        return C[..., idx, idx]
    else:
        raise ValueError("Input should be 2 or 3 dimensional")


def block_diagonal(matrices, sparse=False, format="csr"):
    r"""See pt.slinalg.block_diag or
    pytensor.sparse.basic.block_diag for reference

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
    warnings.warn(
        "pymc.math.block_diagonal is deprecated in favor of `pytensor.tensor.linalg.block_diag` and `pytensor.sparse.block_diag` functions. This function will be removed in a future release",
    )
    if len(matrices) == 1:  # graph optimization
        return matrices[0]
    if sparse:
        return pytensor.sparse.basic.block_diag(*matrices, format=format)
    else:
        return pt.slinalg.block_diag(*matrices)
