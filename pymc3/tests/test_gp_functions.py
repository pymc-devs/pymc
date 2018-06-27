#  pylint:disable=unused-variable
from ..math import cartesian, kronecker
import pymc3 as pm
import theano
import theano.tensor as tt
import numpy as np
import numpy.testing as npt
import pytest

np.random.seed(101)


class TestZeroMean(object):
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            zero_mean = pm.gp.mean.Zero()
        M = theano.function([], zero_mean(X))()
        assert np.all(M==0)
        assert M.shape == (10, )


class TestConstantMean(object):
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            const_mean = pm.gp.mean.Constant(6)
        M = theano.function([], const_mean(X))()
        assert np.all(M==6)
        assert M.shape == (10, )


class TestLinearMean(object):
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            linear_mean = pm.gp.mean.Linear(2, 0.5)
        M = theano.function([], linear_mean(X))()
        npt.assert_allclose(M[1], 0.7222, atol=1e-3)
        assert M.shape == (10, )


class TestAddProdMean(object):
    def test_add(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            mean1 = pm.gp.mean.Linear(coeffs=2, intercept=0.5)
            mean2 = pm.gp.mean.Constant(2)
            mean = mean1 + mean2 + mean2
        M = theano.function([], mean(X))()
        npt.assert_allclose(M[1], 0.7222 + 2 + 2, atol=1e-3)

    def test_prod(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            mean1 = pm.gp.mean.Linear(coeffs=2, intercept=0.5)
            mean2 = pm.gp.mean.Constant(2)
            mean = mean1 * mean2 * mean2
        M = theano.function([], mean(X))()
        npt.assert_allclose(M[1], 0.7222 * 2 * 2, atol=1e-3)

    def test_add_multid(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        A = np.array([1, 2, 3])
        b = 10
        with pm.Model() as model:
            mean1 = pm.gp.mean.Linear(coeffs=A, intercept=b)
            mean2 = pm.gp.mean.Constant(2)
            mean = mean1 + mean2 + mean2
        M = theano.function([], mean(X))()
        npt.assert_allclose(M[1], 10.8965 + 2 + 2, atol=1e-3)

    def test_prod_multid(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        A = np.array([1, 2, 3])
        b = 10
        with pm.Model() as model:
            mean1 = pm.gp.mean.Linear(coeffs=A, intercept=b)
            mean2 = pm.gp.mean.Constant(2)
            mean = mean1 * mean2 * mean2
        M = theano.function([], mean(X))()
        npt.assert_allclose(M[1], 10.8965 * 2 * 2, atol=1e-3)


class TestCovAdd(object):
    def test_symadd_cov(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.ExpQuad(1, 0.1)
            cov = cov1 + cov2
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightadd_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = 1
            cov = pm.gp.cov.ExpQuad(1, 0.1) + a
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 1.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftadd_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = 1
            cov = a + pm.gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 1.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightadd_matrix(self):
        X = np.linspace(0, 1, 10)[:, None]
        M = 2 * np.ones((10, 10))
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(1, 0.1) + M
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftadd_matrixt(self):
        X = np.linspace(0, 1, 10)[:, None]
        M = 2 * tt.ones((10, 10))
        with pm.Model() as model:
            cov = M + pm.gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_matrix(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with pm.Model() as model:
            cov = M + pm.gp.cov.ExpQuad(1, 0.1)
            cov_true = pm.gp.cov.ExpQuad(1, 0.1) + M
        K = theano.function([], cov(X))()
        K_true = theano.function([], cov_true(X))()
        assert np.allclose(K, K_true)


class TestCovProd(object):
    def test_symprod_cov(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.ExpQuad(1, 0.1)
            cov = cov1 * cov2
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.53940 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightprod_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = 2
            cov = pm.gp.cov.ExpQuad(1, 0.1) * a
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            a = 2
            cov = a * pm.gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightprod_matrix(self):
        X = np.linspace(0, 1, 10)[:, None]
        M = 2 * np.ones((10, 10))
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(1, 0.1) * M
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_matrix(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with pm.Model() as model:
            cov = M * pm.gp.cov.ExpQuad(1, 0.1)
            cov_true = pm.gp.cov.ExpQuad(1, 0.1) * M
        K = theano.function([], cov(X))()
        K_true = theano.function([], cov_true(X))()
        assert np.allclose(K, K_true)

    def test_multiops(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with pm.Model() as model:
            cov1 = 3 + pm.gp.cov.ExpQuad(1, 0.1) + M * pm.gp.cov.ExpQuad(1, 0.1) * M * pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.ExpQuad(1, 0.1) * M * pm.gp.cov.ExpQuad(1, 0.1) * M + pm.gp.cov.ExpQuad(1, 0.1) + 3
        K1 = theano.function([], cov1(X))()
        K2 = theano.function([], cov2(X))()
        assert np.allclose(K1, K2)
        # check diagonal
        K1d = theano.function([], cov1(X, diag=True))()
        K2d = theano.function([], cov2(X, diag=True))()
        npt.assert_allclose(np.diag(K1), K2d, atol=1e-5)
        npt.assert_allclose(np.diag(K2), K1d, atol=1e-5)


class TestCovKron(object):
    def test_symprod_cov(self):
        X1 = np.linspace(0, 1, 10)[:, None]
        X2 = np.linspace(0, 1, 10)[:, None]
        X = cartesian(X1, X2)
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.ExpQuad(1, 0.1)
            cov = pm.gp.cov.Kron([cov1, cov2])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 1 * 0.53940, atol=1e-3)
        npt.assert_allclose(K[0, 11], 0.53940 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_multiops(self):
        X1 = np.linspace(0, 1, 3)[:, None]
        X21 = np.linspace(0, 1, 5)[:, None]
        X22 = np.linspace(0, 1, 4)[:, None]
        X2 = cartesian(X21, X22)
        X = cartesian(X1, X21, X22)
        with pm.Model() as model:
            cov1 = 3 + pm.gp.cov.ExpQuad(1, 0.1) + pm.gp.cov.ExpQuad(1, 0.1) * pm.gp.cov.ExpQuad(1, 0.1)
            cov2 = pm.gp.cov.ExpQuad(1, 0.1) * pm.gp.cov.ExpQuad(2, 0.1)
            cov = pm.gp.cov.Kron([cov1, cov2])
        K_true = kronecker(theano.function([], cov1(X1))(), theano.function([], cov2(X2))()).eval()
        K = theano.function([], cov(X))()
        npt.assert_allclose(K_true, K)


class TestCovSliceDim(object):
    def test_slice1(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(3, 0.1, active_dims=[0, 0, 1])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.20084298, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_slice2(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(3, ls=[0.1, 0.1], active_dims=[1,2])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.34295549, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_slice3(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(3, ls=np.array([0.1, 0.1]), active_dims=[1,2])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.34295549, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_diffslice(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(3, ls=0.1, active_dims=[1, 0, 0]) + pm.gp.cov.ExpQuad(3, ls=[0.1, 0.2, 0.3])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.683572, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        lengthscales = 2.0
        with pytest.raises(ValueError):
            pm.gp.cov.ExpQuad(1, lengthscales, [True, False])
            pm.gp.cov.ExpQuad(2, lengthscales, [True])


class TestStability(object):
    def test_stable(self):
        X = np.random.uniform(low=320., high=400., size=[2000, 2])
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(2, 0.1)
        dists = theano.function([], cov.square_dist(X, X))()
        assert not np.any(dists < 0)


class TestExpQuad(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_2d(self):
        X = np.linspace(0, 1, 10).reshape(5, 2)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(2, 0.5)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.820754, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_2dard(self):
        X = np.linspace(0, 1, 10).reshape(5, 2)
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(2, np.array([1, 2]))
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.969607, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_inv_lengthscale(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.ExpQuad(1, ls_inv=10)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestWhiteNoise(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.WhiteNoise(sigma=0.5)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.0, atol=1e-3)
        npt.assert_allclose(K[0, 0], 0.5**2, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)
        # check predict
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.0, atol=1e-3)
        # white noise predicting should return all zeros
        npt.assert_allclose(K[0, 0], 0.0, atol=1e-3)


class TestConstant(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Constant(2.5)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2.5, atol=1e-3)
        npt.assert_allclose(K[0, 0], 2.5, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 2.5, atol=1e-3)
        npt.assert_allclose(K[0, 0], 2.5, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestRatQuad(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.RatQuad(1, ls=0.1, alpha=0.5)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.66896, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.66896, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestExponential(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Exponential(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.57375, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.57375, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestMatern52(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Matern52(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.46202, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.46202, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestMatern32(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Matern32(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.42682, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.42682, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestCosine(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Cosine(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.766, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.766, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestPeriodic(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Periodic(1, 0.1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.00288, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.00288, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestLinear(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Linear(1, 0.5)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.19444, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.19444, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestPolynomial(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            cov = pm.gp.cov.Polynomial(1, 0.5, 2, 0)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.03780, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.03780, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestWarpedInput(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        def warp_func(x, a, b, c):
            return x + (a * tt.tanh(b * (x - c)))
        with pm.Model() as model:
            cov_m52 = pm.gp.cov.Matern52(1, 0.2)
            cov = pm.gp.cov.WarpedInput(1, warp_func=warp_func, args=(1, 10, 1), cov_func=cov_m52)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.79593, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.79593, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        cov_m52 = pm.gp.cov.Matern52(1, 0.2)
        with pytest.raises(TypeError):
            pm.gp.cov.WarpedInput(1, cov_m52, "str is not callable")
        with pytest.raises(TypeError):
            pm.gp.cov.WarpedInput(1, "str is not Covariance object", lambda x: x)


class TestGibbs(object):
    def test_1d(self):
        X = np.linspace(0, 2, 10)[:, None]
        def tanh_func(x, x1, x2, w, x0):
            return (x1 + x2) / 2.0 - (x1 - x2) / 2.0 * tt.tanh((x - x0) / w)
        with pm.Model() as model:
            cov = pm.gp.cov.Gibbs(1, tanh_func, args=(0.05, 0.6, 0.4, 1.0))
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[2, 3], 0.136683, atol=1e-4)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[2, 3], 0.136683, atol=1e-4)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        with pytest.raises(TypeError):
            pm.gp.cov.Gibbs(1, "str is not callable")
        with pytest.raises(NotImplementedError):
            pm.gp.cov.Gibbs(2, lambda x: x)
        with pytest.raises(NotImplementedError):
            pm.gp.cov.Gibbs(3, lambda x: x, active_dims=[0,1])


class TestScaledCov(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        def scaling_func(x, a, b):
            return a + b*x
        with pm.Model() as model:
            cov_m52 = pm.gp.cov.Matern52(1, 0.2)
            cov = pm.gp.cov.ScaledCov(1, scaling_func=scaling_func, args=(2, -1), cov_func=cov_m52)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 3.00686, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 3.00686, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        cov_m52 = pm.gp.cov.Matern52(1, 0.2)
        with pytest.raises(TypeError):
            pm.gp.cov.ScaledCov(1, cov_m52, "str is not callable")
        with pytest.raises(TypeError):
            pm.gp.cov.ScaledCov(1, "str is not Covariance object", lambda x: x)


class TestHandleArgs(object):
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
        func_noargs2 = pm.gp.cov.handle_args(func_noargs, None)
        func_onearg2 = pm.gp.cov.handle_args(func_onearg, a)
        func_twoarg2 = pm.gp.cov.handle_args(func_twoarg, args=(a, b))
        assert func_noargs(x) == func_noargs2(x, args=None)
        assert func_onearg(x, a) == func_onearg2(x, args=a)
        assert func_twoarg(x, a, b) == func_twoarg2(x, args=(a, b))


class TestCoregion(object):
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
            npt.assert_allclose(
                B(np.array([[2, 1.5], [3, -42]])).eval(),
                self.B[2:4, 2:4]
                )
            npt.assert_allclose(B(self.X).eval(), B_mat)

    def test_fullB(self):
        B_mat = self.B[self.rand_rows, self.rand_rows.T]
        with pm.Model() as model:
            B = pm.gp.cov.Coregion(1, B=self.B)
            npt.assert_allclose(
                B(np.array([[2], [3]])).eval(),
                self.B[2:4, 2:4]
                )
            npt.assert_allclose(B(self.X).eval(), B_mat)

    def test_Xs(self):
        B_mat = self.B[self.rand_rows, self.rand_cols.T]
        with pm.Model() as model:
            B = pm.gp.cov.Coregion(2, W=self.W, kappa=self.kappa, active_dims=[0])
            npt.assert_allclose(
                B(np.array([[2, 1.5]]), np.array([[3, -42]])).eval(),
                self.B[2, 3]
                )
            npt.assert_allclose(B(self.X, self.Xs).eval(), B_mat)

    def test_diag(self):
        B_diag = np.diag(self.B)[self.rand_rows.ravel()]
        with pm.Model() as model:
            B = pm.gp.cov.Coregion(2, W=self.W, kappa=self.kappa, active_dims=[0])
            npt.assert_allclose(
                B(np.array([[2, 1.5]]), diag=True).eval(),
                np.diag(self.B)[2]
                )
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


