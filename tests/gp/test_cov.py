#   Copyright 2023 The PyMC Developers
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
import numpy.testing as npt
import pytensor
import pytensor.tensor as pt
import pytest

import pymc as pm

from pymc.math import cartesian, kronecker


class TestCovAdd:
    def test_symadd_cov(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.ExpQuad(1, 0.1)
            cov = cov1 + cov2
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightadd_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = 1
            cov = pm.gp.cov.ExpQuad(1, 0.1) + a
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 1.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftadd_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = 1
            cov = a + pm.gp.cov.ExpQuad(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 1.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightadd_matrix(self):
        X = np.linspace(0, 1, 10)[:, None]
        M = 2 * np.ones((10, 10))
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(1, 0.1) + M
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 2.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftadd_matrixt(self):
        X = np.linspace(0, 1, 10)[:, None]
        M = 2 * pt.ones((10, 10))
        with pm.Model() as model:
            cov = M + pm.gp.cov.ExpQuad(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 2.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_matrix(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with pm.Model() as model:
            cov = M + pm.gp.cov.ExpQuad(1, 0.1)
            cov_true = pm.gp.cov.ExpQuad(1, 0.1) + M
        K = cov(X).eval()
        K_true = cov_true(X).eval()
        assert np.allclose(K, K_true)

    def test_inv_rightadd(self):
        M = np.random.randn(2, 2, 2)
        with pytest.raises(ValueError, match=r"cannot combine"):
            cov = M + pm.gp.cov.ExpQuad(1, 1.0)

    def test_rightadd_whitenoise(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.WhiteNoise(sigma=1)
            cov = cov1 + cov2
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        npt.assert_allclose(K[0, 0], 2, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestCovProd:
    def test_symprod_cov(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.ExpQuad(1, 0.1)
            cov = cov1 * cov2
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.53940 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightprod_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = 2
            cov = pm.gp.cov.ExpQuad(1, 0.1) * a
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = 2
            cov = a * pm.gp.cov.ExpQuad(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightprod_matrix(self):
        X = np.linspace(0, 1, 10)[:, None]
        M = 2 * np.ones((10, 10))
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(1, 0.1) * M
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_matrix(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with pm.Model() as model:
            cov = M * pm.gp.cov.ExpQuad(1, 0.1)
            cov_true = pm.gp.cov.ExpQuad(1, 0.1) * M
        K = cov(X).eval()
        K_true = cov_true(X).eval()
        assert np.allclose(K, K_true)

    def test_multiops(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with pm.Model() as model:
            cov1 = (
                3
                + pm.gp.cov.ExpQuad(1, 0.1)
                + M * pm.gp.cov.ExpQuad(1, 0.1) * M * pm.gp.cov.ExpQuad(1, 0.1)
            )
            cov2 = (
                pm.gp.cov.ExpQuad(1, 0.1) * M * pm.gp.cov.ExpQuad(1, 0.1) * M
                + pm.gp.cov.ExpQuad(1, 0.1)
                + 3
            )
        K1 = cov1(X).eval()
        K2 = cov2(X).eval()
        assert np.allclose(K1, K2)
        # check diagonal
        K1d = cov1(X, diag=True).eval()
        K2d = cov2(X, diag=True).eval()
        npt.assert_allclose(np.diag(K1), K2d, atol=1e-5)
        npt.assert_allclose(np.diag(K2), K1d, atol=1e-5)

    def test_inv_rightprod(self):
        M = np.random.randn(2, 2, 2)
        with pytest.raises(ValueError, match=r"cannot combine"):
            cov = M + pm.gp.cov.ExpQuad(1, 1.0)


class TestCovPSD:
    def test_covpsd_add(self):
        L = 10.0
        omega = np.pi * np.arange(1, 101) / (2 * L)
        with pm.Model() as model:
            cov1 = 2 * pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = 5 * pm.gp.cov.ExpQuad(1, 1.0)
            cov = cov1 + cov2
        psd1 = cov1.power_spectral_density(omega[:, None]).eval()
        psd2 = cov2.power_spectral_density(omega[:, None]).eval()
        psd = cov.power_spectral_density(omega[:, None]).eval()
        npt.assert_allclose(psd, psd1 + psd2)

    def test_copsd_multiply(self):
        # This could be implemented via convolution
        L = 10.0
        omega = np.pi * np.arange(1, 101) / (2 * L)
        with pm.Model() as model:
            cov1 = 2 * pm.gp.cov.ExpQuad(1, ls=1)
            cov2 = pm.gp.cov.ExpQuad(1, ls=1)

        msg = "The power spectral density of products of covariance functions is not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            psd = (cov1 * cov2).power_spectral_density(omega[:, None]).eval()

    def test_covpsd_nonstationary1(self):
        L = 10.0
        omega = np.pi * np.arange(1, 101) / (2 * L)
        with pm.Model() as model:
            cov = 2 * pm.gp.cov.Linear(1, c=5)

        msg = "can only be calculated for `Stationary` covariance functions."
        with pytest.raises(ValueError, match=msg):
            psd = cov.power_spectral_density(omega[:, None]).eval()

    def test_covpsd_nonstationary2(self):
        L = 10.0
        omega = np.pi * np.arange(1, 101) / (2 * L)
        with pm.Model() as model:
            cov = 2 * pm.gp.cov.ExpQuad(1, ls=1) + 10.0

        # Even though this should error, this isnt the appropriate message.  The actual problem
        # is because the covariance function is non-stationary. The underlying bug is due to
        # `Constant` covariances not having an input_dim.
        msg = "All covariances must have the same `input_dim`."
        with pytest.raises(ValueError, match=msg):
            psd = cov.power_spectral_density(omega[:, None]).eval()

    def test_covpsd_notimplemented(self):
        class NewStationaryCov(pm.gp.cov.Stationary):
            pass

        L = 10.0
        omega = np.pi * np.arange(1, 101) / (2 * L)
        with pm.Model() as model:
            cov = 2 * NewStationaryCov(1, ls=1)

        msg = "No power spectral density method has been implemented"
        with pytest.raises(NotImplementedError, match=msg):
            psd = cov.power_spectral_density(omega[:, None]).eval()


class TestCovExponentiation:
    def test_symexp_cov(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, 0.1)
            cov = cov1**2
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.53940**2, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_covexp_numpy(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = np.array([[2]])
            cov = pm.gp.cov.ExpQuad(1, 0.1) ** a
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.53940**2, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_covexp_pytensor(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = pt.alloc(2.0, 1, 1)
            cov = pm.gp.cov.ExpQuad(1, 0.1) ** a
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.53940**2, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_covexp_shared(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = pytensor.shared(2.0)
            cov = pm.gp.cov.ExpQuad(1, 0.1) ** a
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.53940**2, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_invalid_covexp(self):
        with pytest.raises(
            ValueError, match=r"A covariance function can only be exponentiated by a scalar value"
        ):
            with pm.Model():
                a = np.array([[1.0, 2.0]])
                pm.gp.cov.ExpQuad(1, 0.1) ** a

    def test_invalid_covexp_noncov(self):
        with pytest.raises(
            TypeError,
            match=r"Can only exponentiate covariance functions which inherit from `Covariance`",
        ):
            with pm.Model():
                pm.gp.cov.Constant(2) ** 2


class TestCovKron:
    def test_symprod_cov(self):
        X1 = np.linspace(0, 1, 10)[:, None]
        X2 = np.linspace(0, 1, 10)[:, None]
        X = cartesian(X1, X2)
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.ExpQuad(1, 0.1)
            cov = pm.gp.cov.Kron([cov1, cov2])
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 1 * 0.53940, atol=1e-3)
        npt.assert_allclose(K[0, 11], 0.53940 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_multiops(self):
        X1 = np.linspace(0, 1, 3)[:, None]
        X21 = np.linspace(0, 1, 5)[:, None]
        X22 = np.linspace(0, 1, 4)[:, None]
        X2 = cartesian(X21, X22)
        X = cartesian(X1, X21, X22)
        with pm.Model() as model:
            cov1 = (
                3
                + pm.gp.cov.ExpQuad(1, 0.1)
                + pm.gp.cov.ExpQuad(1, 0.1) * pm.gp.cov.ExpQuad(1, 0.1)
            )
            cov2 = pm.gp.cov.ExpQuad(2, 0.1) * pm.gp.cov.ExpQuad(2, 0.1)
            cov = pm.gp.cov.Kron([cov1, cov2])
        K_true = kronecker(cov1(X1).eval(), cov2(X2).eval()).eval()
        K = cov(X).eval()
        npt.assert_allclose(K_true, K)


class TestCovSliceDim:
    def test_slice1(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(3, 0.1, active_dims=[0, 0, 1])
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.20084298, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_slice2(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(3, ls=[0.1, 0.1], active_dims=[1, 2])
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.34295549, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_slice3(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(3, ls=np.array([0.1, 0.1]), active_dims=[1, 2])
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.34295549, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_diffslice(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(3, ls=0.1, active_dims=[1, 0, 0]) + pm.gp.cov.ExpQuad(
                3, ls=[0.1, 0.2, 0.3]
            )
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.683572, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        lengthscales = 2.0
        with pytest.raises(ValueError):
            pm.gp.cov.ExpQuad(1, lengthscales, [True, False])
            pm.gp.cov.ExpQuad(2, lengthscales, [True])


class TestStability:
    def test_stable(self):
        X = np.random.uniform(low=320.0, high=400.0, size=[2000, 2])
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(2, 0.1)
        dists = cov.square_dist(X, X).eval()
        assert not np.any(dists < 0)


class TestExpQuad:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_2d(self):
        X = np.linspace(0, 1, 10).reshape(5, 2)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(2, 0.5)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.820754, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_2dard(self):
        X = np.linspace(0, 1, 10).reshape(5, 2)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(2, np.array([1, 2]))
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.969607, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_inv_lengthscale(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(1, ls_inv=10)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_psd(self):
        # compare to simple 1d formula
        X = np.linspace(0, 1, 10)[:, None]
        omega = np.linspace(0, 2, 50)
        ell = 2.0
        true_1d_psd = np.sqrt(2 * np.pi * np.square(ell)) * np.exp(-0.5 * np.square(ell * omega))
        test_1d_psd = (
            pm.gp.cov.ExpQuad(1, ls=ell).power_spectral_density(omega[:, None]).flatten().eval()
        )
        npt.assert_allclose(true_1d_psd, test_1d_psd, atol=1e-5)

    def test_euclidean_dist(self):
        X = np.arange(0, 3)[:, None]
        Xs = np.arange(1, 4)[:, None]
        with pm.Model():
            cov = pm.gp.cov.ExpQuad(1, ls=1)
        result = cov.euclidean_dist(X, Xs).eval()
        expected = np.array(
            [
                [1, 2, 3],
                [0, 1, 2],
                [1, 0, 1],
            ]
        )
        print(result, expected)
        npt.assert_allclose(result, expected, atol=1e-5)


class TestWhiteNoise:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.WhiteNoise(sigma=0.5)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.0, atol=1e-3)
        npt.assert_allclose(K[0, 0], 0.5**2, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)
        # check predict
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.0, atol=1e-3)
        # white noise predicting should return all zeros
        npt.assert_allclose(K[0, 0], 0.0, atol=1e-3)


class TestConstant:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Constant(2.5)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 2.5, atol=1e-3)
        npt.assert_allclose(K[0, 0], 2.5, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 2.5, atol=1e-3)
        npt.assert_allclose(K[0, 0], 2.5, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestRatQuad:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.RatQuad(1, ls=0.1, alpha=0.5)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.66896, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.66896, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestExponential:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Exponential(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.57375, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.57375, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestMatern52:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Matern52(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.46202, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.46202, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_psd(self):
        # compare to simple 1d formula
        X = np.linspace(0, 1, 10)[:, None]
        omega = np.linspace(0, 2, 50)
        ell = 2.0
        lamda = np.sqrt(5) / ell
        true_1d_psd = (16.0 / 3.0) * np.power(lamda, 5) * np.power(lamda**2 + omega**2, -3)
        test_1d_psd = (
            pm.gp.cov.Matern52(1, ls=ell).power_spectral_density(omega[:, None]).flatten().eval()
        )
        npt.assert_allclose(true_1d_psd, test_1d_psd, atol=1e-5)


class TestMatern32:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Matern32(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.42682, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.42682, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_psd(self):
        # compare to simple 1d formula
        X = np.linspace(0, 1, 10)[:, None]
        omega = np.linspace(0, 2, 50)
        ell = 2.0
        lamda = np.sqrt(3) / ell
        true_1d_psd = 4 * np.power(lamda, 3) * np.power(lamda**2 + omega**2, -2)
        test_1d_psd = (
            pm.gp.cov.Matern32(1, ls=ell).power_spectral_density(omega[:, None]).flatten().eval()
        )
        npt.assert_allclose(true_1d_psd, test_1d_psd, atol=1e-5)


class TestMatern12:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Matern12(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.32919, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.32919, atol=1e-3)
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestCosine:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Cosine(1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.766, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.766, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestPeriodic:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Periodic(1, 0.1, 0.1)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.00288, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.00288, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestLinear:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Linear(1, 0.5)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.19444, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.19444, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestPolynomial:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Polynomial(1, 0.5, 2, 0)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.03780, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.03780, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestWarpedInput:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]

        def warp_func(x, a, b, c):
            return x + (a * pt.tanh(b * (x - c)))

        with pm.Model() as model:
            cov_m52 = pm.gp.cov.Matern52(1, 0.2)
            cov = pm.gp.cov.WarpedInput(1, warp_func=warp_func, args=(1, 10, 1), cov_func=cov_m52)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 0.79593, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 0.79593, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        cov_m52 = pm.gp.cov.Matern52(1, 0.2)
        with pytest.raises(TypeError):
            pm.gp.cov.WarpedInput(1, cov_m52, "str is not callable")
        with pytest.raises(TypeError):
            pm.gp.cov.WarpedInput(1, "str is not Covariance object", lambda x: x)


class TestWrappedPeriodic:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model():
            cov1 = pm.gp.cov.Periodic(1, ls=0.2, period=1)
            cov2 = pm.gp.cov.WrappedPeriodic(
                cov_func=pm.gp.cov.ExpQuad(1, ls=0.2),
                period=1,
            )
        K1 = cov1(X).eval()
        K2 = cov2(X).eval()
        npt.assert_allclose(K1, K2, atol=1e-3)
        K1d = cov1(X, diag=True).eval()
        K2d = cov2(X, diag=True).eval()
        npt.assert_allclose(K1d, K2d, atol=1e-3)

    def test_raises(self):
        lin_cov = pm.gp.cov.Linear(1, c=1)
        with pytest.raises(TypeError, match="Must inherit from the Stationary class"):
            pm.gp.cov.WrappedPeriodic(lin_cov, period=1)


class TestGibbs:
    def test_1d(self):
        X = np.linspace(0, 2, 10)[:, None]

        def tanh_func(x, x1, x2, w, x0):
            return (x1 + x2) / 2.0 - (x1 - x2) / 2.0 * pt.tanh((x - x0) / w)

        with pm.Model() as model:
            cov = pm.gp.cov.Gibbs(1, tanh_func, args=(0.05, 0.6, 0.4, 1.0))
        K = cov(X).eval()
        npt.assert_allclose(K[2, 3], 0.136683, atol=1e-4)
        K = cov(X, X).eval()
        npt.assert_allclose(K[2, 3], 0.136683, atol=1e-4)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        with pytest.raises(TypeError):
            pm.gp.cov.Gibbs(1, "str is not callable")
        with pytest.raises(NotImplementedError):
            pm.gp.cov.Gibbs(2, lambda x: x)
        with pytest.raises(NotImplementedError):
            pm.gp.cov.Gibbs(3, lambda x: x, active_dims=[0, 1])


class TestScaledCov:
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]

        def scaling_func(x, a, b):
            return a + b * x

        with pm.Model() as model:
            cov_m52 = pm.gp.cov.Matern52(1, 0.2)
            cov = pm.gp.cov.ScaledCov(1, scaling_func=scaling_func, args=(2, -1), cov_func=cov_m52)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], 3.00686, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], 3.00686, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        cov_m52 = pm.gp.cov.Matern52(1, 0.2)
        with pytest.raises(TypeError):
            pm.gp.cov.ScaledCov(1, cov_m52, "str is not callable")
        with pytest.raises(TypeError):
            pm.gp.cov.ScaledCov(1, "str is not Covariance object", lambda x: x)


class TestCircular:
    def test_1d_tau1(self):
        X = np.linspace(0, 1, 10)[:, None]
        etalon = 0.600881
        with pm.Model():
            cov = pm.gp.cov.Circular(1, 1, tau=5)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], etalon, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], etalon, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_1d_tau2(self):
        X = np.linspace(0, 1, 10)[:, None]
        etalon = 0.691239
        with pm.Model():
            cov = pm.gp.cov.Circular(1, 1, tau=4)
        K = cov(X).eval()
        npt.assert_allclose(K[0, 1], etalon, atol=1e-3)
        K = cov(X, X).eval()
        npt.assert_allclose(K[0, 1], etalon, atol=1e-3)
        # check diagonal
        Kd = cov(X, diag=True).eval()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestHandleArgs:
    def test_handleargs(self):
        def func_noargs(x):
            return x

        def func_onearg(x, a):
            return x + a

        def func_twoarg(x, a, b):
            return x + a + b

        x = 100
        a = 2
        b = 3
        func_noargs2 = pm.gp.cov.handle_args(func_noargs)
        func_onearg2 = pm.gp.cov.handle_args(func_onearg)
        func_twoarg2 = pm.gp.cov.handle_args(func_twoarg)
        assert func_noargs(x) == func_noargs2(x, args=None)
        assert func_onearg(x, a) == func_onearg2(x, args=a)
        assert func_twoarg(x, a, b) == func_twoarg2(x, args=(a, b))


class TestCoregion:
    def setup_method(self):
        self.nrows = 6
        self.ncols = 3
        self.W = np.random.rand(self.nrows, self.ncols)
        self.kappa = np.random.rand(self.nrows)
        self.B = np.dot(self.W, self.W.T) + np.diag(self.kappa)
        self.rand_rows = np.random.randint(0, self.nrows, size=(20, 1))
        self.rand_cols = np.random.randint(0, self.ncols, size=(10, 1))
        self.X = np.concatenate((self.rand_rows, np.random.rand(20, 1)), axis=1)
        self.Xs = np.concatenate((self.rand_cols, np.random.rand(10, 1)), axis=1)

    def test_full(self):
        B_mat = self.B[self.rand_rows, self.rand_rows.T]
        with pm.Model() as model:
            B = pm.gp.cov.Coregion(2, W=self.W, kappa=self.kappa, active_dims=[0])
            npt.assert_allclose(B(np.array([[2, 1.5], [3, -42]])).eval(), self.B[2:4, 2:4])
            npt.assert_allclose(B(self.X).eval(), B_mat)

    def test_fullB(self):
        B_mat = self.B[self.rand_rows, self.rand_rows.T]
        with pm.Model() as model:
            B = pm.gp.cov.Coregion(1, B=self.B)
            npt.assert_allclose(B(np.array([[2], [3]])).eval(), self.B[2:4, 2:4])
            npt.assert_allclose(B(self.X).eval(), B_mat)

    def test_Xs(self):
        B_mat = self.B[self.rand_rows, self.rand_cols.T]
        with pm.Model() as model:
            B = pm.gp.cov.Coregion(2, W=self.W, kappa=self.kappa, active_dims=[0])
            npt.assert_allclose(B(np.array([[2, 1.5]]), np.array([[3, -42]])).eval(), self.B[2, 3])
            npt.assert_allclose(B(self.X, self.Xs).eval(), B_mat)

    def test_diag(self):
        B_diag = np.diag(self.B)[self.rand_rows.ravel()]
        with pm.Model() as model:
            B = pm.gp.cov.Coregion(2, W=self.W, kappa=self.kappa, active_dims=[0])
            npt.assert_allclose(B(np.array([[2, 1.5]]), diag=True).eval(), np.diag(self.B)[2])
            npt.assert_allclose(B(self.X, diag=True).eval(), B_diag)

    def test_raises(self):
        with pm.Model() as model:
            with pytest.raises(ValueError):
                B = pm.gp.cov.Coregion(2, W=self.W, kappa=self.kappa)

    def test_raises2(self):
        with pm.Model() as model:
            with pytest.raises(ValueError):
                B = pm.gp.cov.Coregion(1, W=self.W, kappa=self.kappa, B=self.B)

    def test_raises3(self):
        with pm.Model() as model:
            with pytest.raises(ValueError):
                B = pm.gp.cov.Coregion(1)
