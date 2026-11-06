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
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.tensor.type import tensor
from scipy import stats

import pymc as pm

from pymc.distributions import MatrixNormal, MvNormal, Normal
from pymc.logprob.basic import logp


@pytest.mark.parametrize("univariate", [True, False])
@pytest.mark.parametrize("batch_shape", [(), (3,)])
def test_matrix_vector_transform(univariate, batch_shape):
    rng = np.random.default_rng(755)

    μ = rng.normal(size=(*batch_shape, 2))
    if univariate:
        σ = np.abs(rng.normal(size=(*batch_shape, 2)))
        Σ = np.eye(2) * (σ**2)[..., None]
        x = Normal.dist(mu=μ, sigma=σ)
    else:
        A = rng.normal(size=(*batch_shape, 2, 2))
        Σ = np.swapaxes(A, -1, -2) @ A
        x = MvNormal.dist(mu=μ, cov=Σ)

    c = rng.normal(size=(*batch_shape, 2))
    B = rng.normal(size=(*batch_shape, 2, 2))
    y = c + (B @ x[..., None]).squeeze(-1)

    # An affine transformed MvNormal is still a MvNormal
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Affine_transformation
    ref_dist = MvNormal.dist(
        mu=c + (B @ μ[..., None]).squeeze(-1), cov=B @ Σ @ np.swapaxes(B, -1, -2)
    )
    test_y = rng.normal(size=(*batch_shape, 2))
    np.testing.assert_allclose(
        logp(y, test_y).eval(),
        logp(ref_dist, test_y).eval(),
    )


def test_matrix_matrix_transform():
    rng = np.random.default_rng(46)

    n, p = 2, 3
    M = rng.normal(size=(n, p))
    A = rng.normal(size=(n, n)) * 0.1
    # Keep covariances well-conditioned so the Jacobian is visible at this tolerance.
    U = A.T @ A + np.eye(n)
    B = rng.normal(size=(p, p)) * 0.1
    V = B.T @ B + np.eye(p)
    X = MatrixNormal.dist(mu=M, rowcov=U, colcov=V)

    D = rng.normal(size=(n, n))
    C = rng.normal(size=(p, p))
    Y = D @ X @ C

    # A linearly transformed MatrixNormal is still a MatrixNormal
    # https://en.wikipedia.org/wiki/Matrix_normal_distribution#Transformation
    ref_dist = MatrixNormal.dist(mu=D @ M @ C, rowcov=D @ U @ D.T, colcov=C.T @ V @ C)
    test_Y = rng.normal(size=(n, p))
    np.testing.assert_allclose(
        logp(Y, test_Y).eval(),
        logp(ref_dist, test_Y).eval(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_broadcasted_matmul_fails():
    x = Normal.dist(size=(3, 2))
    A = tensor("A", shape=(4, 3, 3))
    y = A @ x
    with pytest.raises(NotImplementedError):
        logp(y, y.type())


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "base_dist,batch_shape",
    [
        ("normal", ()),
        ("normal", (2,)),
        ("normal", (2, 3)),
        ("mvnormal", ()),
        ("mvnormal", (2,)),
        ("mvnormal", (2, 3)),
        ("matrixnormal", ()),
    ],
)
def test_matrix_matmul_jacobian(side, base_dist, batch_shape):
    rows, cols = (2, 3) if side == "left" else (3, 2)
    shape = (*batch_shape, rows, cols)
    mu = np.linspace(-0.3, 0.5, np.prod(shape)).reshape(shape)
    value = np.linspace(-1.1, 0.9, np.prod(shape)).reshape(shape)
    rowcov = np.eye(rows) + 0.2
    colcov = np.eye(cols) + 0.1
    if base_dist == "normal":
        x = Normal.dist(mu=mu, sigma=1.2)
        base_cov = np.eye(rows * cols) * 1.2**2
    elif base_dist == "mvnormal":
        x = MvNormal.dist(mu=mu, cov=colcov)
        base_cov = np.kron(np.eye(rows), colcov)
    else:
        x = MatrixNormal.dist(mu=mu, rowcov=rowcov, colcov=colcov)
        base_cov = np.kron(rowcov, colcov)

    # A non-diagonal map exercises matmul; a constant diagonal can rewrite to scaling.
    a = np.array([[2.0, 0.2], [0.0, -1.5]])
    if batch_shape:
        scales = np.linspace(1.0, 1.5, np.prod(batch_shape)).reshape(batch_shape)
        a = scales[..., None, None] * a
    transformed = a @ x if side == "left" else x @ a
    actual = logp(transformed, value).eval()
    assert actual.shape == batch_shape
    expected = np.empty(batch_shape)
    for index in np.ndindex(batch_shape):
        matrix = a[index] if batch_shape else a
        transform = (
            np.kron(matrix, np.eye(cols)) if side == "left" else np.kron(np.eye(rows), matrix.T)
        )
        expected[index] = stats.multivariate_normal.logpdf(
            value[index].ravel(),
            mean=transform @ mu[index].ravel(),
            cov=transform @ base_cov @ transform.T,
        )
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-10)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("shared", [False, True])
def test_matrix_matmul_diagonal_rewrites(side, shared):
    rows, cols = (2, 3) if side == "left" else (3, 2)
    matrix = np.diag([2.0, 3.0])
    a = pytensor.shared(matrix) if shared else matrix
    x = Normal.dist(mu=np.zeros((rows, cols)), sigma=1.2)
    transformed = a @ x if side == "left" else x @ a
    value = pt.matrix("value")
    # The constant diagonal rewrite retains scalar logp axes; compare joint densities.
    fn = pytensor.function([value], logp(transformed, value).sum())
    observed = np.linspace(-1.1, 0.9, rows * cols).reshape(rows, cols)
    matrices = [matrix, np.diag([-1.5, 2.0])] if shared else [matrix]
    for matrix in matrices:
        if shared:
            a.set_value(matrix)
        solved = (
            np.linalg.solve(matrix, observed)
            if side == "left"
            else np.linalg.solve(matrix.T, observed.T).T
        )
        expected = stats.norm.logpdf(solved, scale=1.2).sum()
        expected -= (cols if side == "left" else rows) * np.linalg.slogdet(matrix)[1]
        np.testing.assert_allclose(fn(observed), expected, atol=1e-9, rtol=1e-10)


@pytest.mark.parametrize("side", ["left", "right"])
def test_matrix_matmul_dynamic_shape(side):
    a, mu, value = (pt.matrix(name) for name in ("A", "mu", "value"))
    x = Normal.dist(mu=mu, sigma=1.2)
    transformed = a @ x if side == "left" else x @ a
    fn = pytensor.function([a, mu, value], logp(transformed, value))
    matrix = np.array([[2.0, 0.2], [0.0, 3.0]])
    for repeats in (3, 1):
        shape = (2, repeats) if side == "left" else (repeats, 2)
        means = np.linspace(-0.3, 0.5, np.prod(shape)).reshape(shape)
        observed = np.linspace(-1.1, 0.9, np.prod(shape)).reshape(shape)
        solved = (
            np.linalg.solve(matrix, observed)
            if side == "left"
            else np.linalg.solve(matrix.T, observed.T).T
        )
        expected = stats.norm.logpdf(solved, loc=means, scale=1.2).sum()
        expected -= repeats * np.linalg.slogdet(matrix)[1]
        np.testing.assert_allclose(fn(matrix, means, observed), expected, atol=1e-9, rtol=1e-10)


@pytest.mark.parametrize("side", ["left", "right"])
def test_matrix_matmul_model_logp_and_gradient(side):
    rows, cols = (3, 4) if side == "left" else (4, 3)
    observed = np.linspace(-1.1, 0.9, rows * cols).reshape(rows, cols)
    models = []
    for reference in (False, True):
        with pm.Model() as model:
            scale = pm.HalfNormal("scale", sigma=1.0)
            a = scale * np.array([[2.0, 0.2, 0.1], [0.1, 3.0, -0.2], [0.2, 0.1, 1.5]])
            if reference:
                rowcov = a @ a.T if side == "left" else np.eye(rows)
                colcov = np.eye(cols) if side == "left" else a.T @ a
                MatrixNormal(
                    "y",
                    mu=np.zeros((rows, cols)),
                    rowcov=rowcov,
                    colcov=colcov,
                    observed=observed,
                )
            else:
                x = Normal.dist(mu=np.zeros((rows, cols)), sigma=1.0)
                y = a @ x if side == "left" else x @ a
                # Condition on model parameters before deriving the transformed density.
                [y] = model.replace_rvs_by_values([y])
                pm.Potential("density", logp(y, observed))
        models.append(model)
    point = {"scale_log__": np.log(1.3)}
    np.testing.assert_allclose(
        models[0].compile_logp()(point),
        models[1].compile_logp()(point),
        atol=1e-9,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        models[0].compile_dlogp()(point),
        models[1].compile_dlogp()(point),
        atol=1e-9,
        rtol=1e-10,
    )
