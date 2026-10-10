#   Copyright 2024 - present The PyMC Developers
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
"""Recognize a value variable that is an affine function of independent Gaussian leaves.

A linear function of independent Gaussians is itself Gaussian, so its density is
available in closed form even when the map is non-invertible (wide / low-rank
matrices, sums of independent normals) and the usual invert-and-Jacobian
machinery (``MeasurableMatMul``, the elemwise transforms) cannot derive it.

Instead of inverting, we *propagate moments*: walk the affine subgraph feeding a
value variable, accumulate a symbolic ``(mean, cov)`` over its last axis, and
substitute a plain ``MvNormal`` whose existing ``_logprob`` produces the density.

The accumulated covariance is kept dense (e.g. ``W @ W.T + diag(d**2)`` for the
low-rank case). Exploiting its structure for an ``O(D K**2)`` logp would require a
Woodbury / matrix-determinant-lemma rewrite in PyTensor, which does not exist
yet; until then the emitted ``MvNormal`` logp does a dense ``O(D**3)`` cholesky.
"""

import pytensor.tensor as pt

from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.graph.traversal import ancestors
from pytensor.scalar.basic import Add as ScalarAdd
from pytensor.scalar.basic import Mul as ScalarMul
from pytensor.tensor.elemwise import DimShuffle, Elemwise
from pytensor.tensor.math import _matmul
from pytensor.tensor.random.basic import MvNormalRV, NormalRV, multivariate_normal
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorVariable

from pymc.logprob.abstract import MeasurableOp
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.logprob.transforms import LocTransform, MeasurableTransform, ScaleTransform
from pymc.logprob.utils import check_potential_measurability

# A moment triple ``(mean, cov, leaves)`` describes a variable as Gaussian over its
# last axis: ``mean`` has shape ``(..., n)``, ``cov`` shape ``(..., n, n)``, and
# ``leaves`` is the frozenset of RandomVariable outputs feeding it (used to check
# that independent terms depend on disjoint leaves so their covariances add).


def _diag_from_vec(v):
    """Build a ``(..., n, n)`` matrix with diagonal ``v`` (shape ``(..., n)``)."""
    return pt.eye(v.shape[-1]) * v[..., None, :]


def _rv_leaves(var) -> frozenset:
    return frozenset(
        a
        for a in ancestors([var])
        if a.owner is not None and isinstance(a.owner.op, RandomVariable)
    )


def _is_squeeze_last_axis(op, inp) -> bool:
    """True if ``op`` is a DimShuffle that just drops the last (length-1) axis."""
    if not isinstance(op, DimShuffle):
        return False
    return op.new_order == tuple(range(inp.type.ndim - 1))


def _is_expand_last_axis(op) -> bool:
    """True if ``op`` is a DimShuffle that just appends a new last axis."""
    if not isinstance(op, DimShuffle):
        return False
    new_order = op.new_order
    return (
        len(new_order) == op.input_ndim + 1
        and new_order[-1] == "x"
        and new_order[:-1] == tuple(range(op.input_ndim))
    )


def _normal_leaf_moments(var):
    """Moments of an independent ``Normal`` vector over its last axis."""
    if var.type.ndim < 1:
        return None
    rng, size, mu, sigma = var.owner.inputs
    shape = var.shape
    mean = pt.broadcast_to(mu, shape)
    cov = _diag_from_vec(pt.broadcast_to(sigma, shape) ** 2)
    return mean, cov, frozenset({var})


def _column_moments(col):
    """Moments of a column Gaussian ``(..., n, 1)`` along its ``n`` axis.

    Returns ``(mean_col, cov, leaves)`` with ``mean_col`` shape ``(..., n, 1)`` and
    ``cov`` shape ``(..., n, n)``, or ``None``.
    """
    node = col.owner
    if node is None:
        return None
    op = node.op

    # ``ExpandDims(g)`` of a vector Gaussian (un-lifted form).
    if _is_expand_last_axis(op):
        inner = _affine_gaussian_moments(node.inputs[0])
        if inner is None:
            return None
        mean_v, cov, leaves = inner
        return mean_v[..., None], cov, leaves

    # A ``Normal`` RV lifted directly into ``(..., n, 1)`` shape.
    if isinstance(op, NormalRV):
        rng, size, mu, sigma = node.inputs
        shape = col.shape
        mean_col = pt.broadcast_to(mu, shape)
        cov = _diag_from_vec((pt.broadcast_to(sigma, shape) ** 2)[..., 0])
        return mean_col, cov, frozenset({col})

    return None


def _moments_of_matvec(var):
    """Moments of ``A @ g`` written as ``Squeeze(Matmul(A, g_col))``."""
    node = var.owner
    if not _is_squeeze_last_axis(node.op, node.inputs[0]):
        return None
    mm = node.inputs[0]
    if mm.owner is None or mm.owner.op != _matmul:
        return None
    A, col = mm.owner.inputs
    # The constant operand must carry no hidden measurable dependency.
    if check_potential_measurability([A]):
        return None
    inner = _column_moments(col)
    if inner is None:
        return None
    mean_col, cov_g, leaves = inner
    mean = (A @ mean_col)[..., 0]
    cov = A @ cov_g @ A.mT
    return mean, cov, leaves


def _moments_of_add(node, *, require_two=False):
    """Moments of a sum of independent Gaussian terms plus constant shifts."""
    shifts = []
    terms = []
    for inp in node.inputs:
        if not _rv_leaves(inp):
            shifts.append(inp)
            continue
        moments = _affine_gaussian_moments(inp)
        if moments is None:
            return None
        terms.append(moments)

    if not terms or (require_two and len(terms) < 2):
        return None

    # Covariances only add if the terms depend on disjoint leaves.
    all_leaves: frozenset = frozenset()
    for _, _, leaves in terms:
        if all_leaves & leaves:
            return None
        all_leaves |= leaves

    mean = terms[0][0]
    cov = terms[0][1]
    for term_mean, term_cov, _ in terms[1:]:
        mean = mean + term_mean
        cov = cov + term_cov
    for shift in shifts:
        mean = mean + shift
    return mean, cov, all_leaves


def _moments_of_scale(node):
    """Moments of an elementwise ``c * g`` with constant ``c``."""
    gaussian = None
    consts = []
    for inp in node.inputs:
        if _rv_leaves(inp):
            if gaussian is not None:
                return None
            gaussian = inp
        else:
            consts.append(inp)
    if gaussian is None:
        return None
    inner = _affine_gaussian_moments(gaussian)
    if inner is None:
        return None
    mean, cov, leaves = inner
    c = consts[0] if len(consts) == 1 else pt.mul(*consts)
    mean = c * mean
    cov = cov * c[..., None, :] * c[..., :, None]
    return mean, cov, leaves


def _moments_through_transform(node):
    """Read moments through an already-measurable Loc/Scale transform."""
    op = node.op
    transform = op.transform_elemwise
    base = node.inputs[op.measurable_input_idx]
    inner = _affine_gaussian_moments(base)
    if inner is None:
        return None
    mean, cov, leaves = inner
    others = [inp for i, inp in enumerate(node.inputs) if i != op.measurable_input_idx]
    if isinstance(transform, LocTransform):
        loc = others[0] if len(others) == 1 else pt.add(*others)
        return mean + loc, cov, leaves
    if isinstance(transform, ScaleTransform):
        c = others[0] if len(others) == 1 else pt.mul(*others)
        return c * mean, cov * c[..., None, :] * c[..., :, None], leaves
    # Any other transform (Exp, Log, Abs, ...) is non-affine.
    return None


def _affine_gaussian_moments(var):
    """Interpret ``var`` as an affine function of independent Gaussian leaves.

    Returns the dense moment triple ``(mean, cov, leaves)`` over ``var``'s last
    axis, or ``None`` if ``var`` is not such an affine function.
    """
    node = var.owner
    if node is None:
        return None
    op = node.op

    if isinstance(op, NormalRV):
        return _normal_leaf_moments(var)

    if isinstance(op, MvNormalRV):
        rng, size, mu, cov = node.inputs
        return pt.broadcast_arrays(mu, cov[..., -1])[0], cov, frozenset({var})

    if isinstance(op, MeasurableTransform):
        return _moments_through_transform(node)

    if isinstance(op, DimShuffle):
        return _moments_of_matvec(var)

    if isinstance(op, Elemwise):
        if isinstance(op.scalar_op, ScalarAdd):
            return _moments_of_add(node)
        if isinstance(op.scalar_op, ScalarMul):
            return _moments_of_scale(node)

    return None


def _square_or_unknown(A) -> bool:
    """True unless ``A`` is statically known to be non-square (so we should fire)."""
    m, n = A.type.shape[-2], A.type.shape[-1]
    return m is None or n is None or m == n


def _emit_mvnormal(mean, cov, leaves) -> list[TensorVariable]:
    rng = next((leaf.owner.inputs[0] for leaf in leaves), None)
    return [multivariate_normal(mean, cov, rng=rng)]


@node_rewriter(tracks=[DimShuffle])
def find_measurable_matvec_normal(fgraph, node):
    """Recognize ``A @ g`` of a Gaussian with non-square ``A`` as an ``MvNormal``."""
    if isinstance(node.op, MeasurableOp):
        return None
    if not _is_squeeze_last_axis(node.op, node.inputs[0]):
        return None
    mm = node.inputs[0]
    if mm.owner is None or mm.owner.op != _matmul:
        return None
    A, _ = mm.owner.inputs
    # Square (or statically unknown) maps are invertible; leave them to
    # MeasurableMatMul to avoid stepping on that path.
    if _square_or_unknown(A):
        return None
    moments = _affine_gaussian_moments(node.outputs[0])
    if moments is None:
        return None
    return _emit_mvnormal(*moments)


@node_rewriter(tracks=[Elemwise])
def find_measurable_sum_of_gaussians(fgraph, node):
    """Recognize a sum of >=2 independent Gaussians as an ``MvNormal``."""
    if isinstance(node.op, MeasurableOp):
        return None
    if not isinstance(node.op.scalar_op, ScalarAdd):
        return None
    moments = _moments_of_add(node, require_two=True)
    if moments is None:
        return None
    return _emit_mvnormal(*moments)


measurable_ir_rewrites_db.register(
    find_measurable_matvec_normal.__name__,
    find_measurable_matvec_normal,
    "basic",
    "gaussian",
)
measurable_ir_rewrites_db.register(
    find_measurable_sum_of_gaussians.__name__,
    find_measurable_sum_of_gaussians,
    "basic",
    "gaussian",
)
