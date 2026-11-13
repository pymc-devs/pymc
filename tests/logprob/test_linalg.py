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
import pytest

from pytensor.tensor.type import tensor

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
    U = A.T @ A
    B = rng.normal(size=(p, p)) * 0.1
    V = B.T @ B
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
        rtol=1e-5,
    )


def test_broadcasted_matmul_fails():
    x = Normal.dist(size=(3, 2))
    A = tensor("A", shape=(4, 3, 3))
    y = A @ x
    with pytest.raises(NotImplementedError):
        logp(y, y.type())


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("observed", [False, True])
def test_matmul_custom_dist_model_logp(side, observed):
    from scipy import stats

    import pymc as pm

    # Unit determinant keeps this independent of the matrix Jacobian
    # multiplicity issue. The map still changes the covariance.
    matrix = np.array([[1.0, 0.3], [0.0, 1.0]])
    value = np.array([[0.4, -0.2, 0.7], [1.1, 0.5, -0.3]])
    shape = value.shape if side == "left" else value.T.shape

    def dist(mu, size):
        x = Normal.dist(mu=mu, sigma=1.0, shape=shape)
        return matrix @ x if side == "left" else x @ matrix.T

    with pm.Model() as model:
        mu = Normal("mu", 0.0, 1.0)
        pm.CustomDist(
            "y",
            mu,
            dist=dist,
            observed=(value if side == "left" else value.T) if observed else None,
        )

    point = {"mu": 0.2}
    if not observed:
        point["y"] = value if side == "left" else value.T
    whitened = np.linalg.solve(matrix, value)
    expected = stats.norm.logpdf(point["mu"]) + stats.norm.logpdf(whitened, point["mu"]).sum()
    expected_grad = -point["mu"] + (whitened - point["mu"]).sum()
    for jacobian in (False, True):
        np.testing.assert_allclose(model.compile_logp(jacobian=jacobian)(point), expected)
        np.testing.assert_allclose(
            model.compile_dlogp(vars=[mu], jacobian=jacobian)(point), [expected_grad]
        )
