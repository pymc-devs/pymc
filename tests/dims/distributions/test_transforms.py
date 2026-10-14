#   Copyright 2026 - present The PyMC Developers
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

from pytensor.xtensor import as_xtensor

from pymc.dims.distributions.transforms import WeightedZeroSumTransform, ZeroSumTransform

WEIGHTS = [
    np.array([0.70, 0.20, 0.08, 0.02]),
    np.array([0.90, 0.05, 0.03, 0.02]),
    np.array([0.25, 0.25, 0.25, 0.25]),
    np.array([0.5, 0.5]),
]


@pytest.mark.parametrize("w", WEIGHTS)
def test_weighted_zerosum_roundtrip_and_constraint(w):
    rng = np.random.default_rng(2026)
    n = len(w)
    transform = WeightedZeroSumTransform(dim="a", weights=w)

    z_np = rng.normal(size=(11, n - 1))
    z = as_xtensor(z_np, dims=("batch", "a"))
    x = transform.backward(z)
    x_np = x.transpose("batch", "a").eval()

    u = w / np.linalg.norm(w)
    np.testing.assert_allclose(x_np @ u, 0.0, atol=1e-12)

    z_back = transform.forward(x).transpose("batch", "a").eval()
    np.testing.assert_allclose(z_back, z_np, atol=1e-12)

    # isometry => log_jac_det == 0
    np.testing.assert_allclose(
        np.linalg.norm(x_np, axis=-1), np.linalg.norm(z_np, axis=-1), atol=1e-12
    )
    ljd = transform.log_jac_det(x).eval()
    np.testing.assert_allclose(ljd, 0.0, atol=1e-12)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_weighted_zerosum_equal_weights_matches_zerosum(n):
    rng = np.random.default_rng(5)
    weighted = WeightedZeroSumTransform(dim="a", weights=np.full(n, 1.0 / n))
    uniform = ZeroSumTransform(dims=("a",))

    z = as_xtensor(rng.normal(size=(7, n - 1)), dims=("batch", "a"))
    x_w = weighted.backward(z).transpose("batch", "a").eval()
    x_u = uniform.backward(z).transpose("batch", "a").eval()
    np.testing.assert_allclose(x_w, x_u, atol=1e-12)

    x = as_xtensor(x_w, dims=("batch", "a"))
    np.testing.assert_allclose(
        weighted.forward(x).transpose("batch", "a").eval(),
        uniform.forward(x).transpose("batch", "a").eval(),
        atol=1e-12,
    )


def test_weighted_zerosum_invalid_weights():
    with pytest.raises(ValueError, match="strictly positive"):
        WeightedZeroSumTransform(dim="a", weights=np.array([0.5, 0.0, 0.5]))
    with pytest.raises(ValueError, match="1-d"):
        WeightedZeroSumTransform(dim="a", weights=np.ones((2, 2)))
