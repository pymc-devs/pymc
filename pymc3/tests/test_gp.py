#  pylint:disable=unused-variable
from functools import reduce
from operator import add
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


class TestMarginalVsLatent(object):
    R"""
    Compare the logp of models Marginal, noise=0 and Latent.
    """
    def setup_method(self):
        X = np.random.randn(50,3)
        y = np.random.randn(50)*0.01
        Xnew = np.random.randn(60, 3)
        pnew = np.random.randn(60)*0.01
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Marginal(mean_func, cov_func)
            f = gp.marginal_likelihood("f", X, y, noise=0.0, is_observed=False, observed=y)
            p = gp.conditional("p", Xnew)
        self.logp = model.logp({"p": pnew})
        self.X = X
        self.Xnew = Xnew
        self.y = y
        self.pnew = pnew

    def testLatent1(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Latent(mean_func, cov_func)
            f = gp.prior("f", self.X, reparameterize=False)
            p = gp.conditional("p", self.Xnew)
        latent_logp = model.logp({"f": self.y, "p": self.pnew})
        npt.assert_allclose(latent_logp, self.logp, atol=0, rtol=1e-2)

    def testLatent2(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Latent(mean_func, cov_func)
            f = gp.prior("f", self.X, reparameterize=True)
            p = gp.conditional("p", self.Xnew)
        chol = np.linalg.cholesky(cov_func(self.X).eval())
        y_rotated = np.linalg.solve(chol, self.y - 0.5)
        latent_logp = model.logp({"f_rotated_": y_rotated, "p": self.pnew})
        npt.assert_allclose(latent_logp, self.logp, atol=0, rtol=1e-2)


class TestMarginalVsMarginalSparse(object):
    R"""
    Compare logp of models Marginal and MarginalSparse.
    Should be nearly equal when inducing points are same as inputs.
    """
    def setup_method(self):
        X = np.random.randn(50,3)
        y = np.random.randn(50)*0.01
        Xnew = np.random.randn(60, 3)
        pnew = np.random.randn(60)*0.01
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Marginal(mean_func, cov_func)
            sigma = 0.1
            f = gp.marginal_likelihood("f", X, y, noise=sigma)
            p = gp.conditional("p", Xnew)
        self.logp = model.logp({"p": pnew})
        self.X = X
        self.Xnew = Xnew
        self.y = y
        self.sigma = sigma
        self.pnew = pnew
        self.gp = gp

    @pytest.mark.parametrize('approx', ['FITC', 'VFE', 'DTC'])
    def testApproximations(self, approx):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.MarginalSparse(mean_func, cov_func, approx=approx)
            f = gp.marginal_likelihood("f", self.X, self.X, self.y, self.sigma)
            p = gp.conditional("p", self.Xnew)
        approx_logp = model.logp({"f": self.y, "p": self.pnew})
        npt.assert_allclose(approx_logp, self.logp, atol=0, rtol=1e-2)

    @pytest.mark.parametrize('approx', ['FITC', 'VFE', 'DTC'])
    def testPredictVar(self, approx):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.MarginalSparse(mean_func, cov_func, approx=approx)
            f = gp.marginal_likelihood("f", self.X, self.X, self.y, self.sigma)
        mu1, var1 = self.gp.predict(self.Xnew, diag=True)
        mu2, var2 = gp.predict(self.Xnew, diag=True)
        npt.assert_allclose(mu1, mu2, atol=0, rtol=1e-3)
        npt.assert_allclose(var1, var2, atol=0, rtol=1e-3)

    def testPredictCov(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.MarginalSparse(mean_func, cov_func, approx="DTC")
            f = gp.marginal_likelihood("f", self.X, self.X, self.y, self.sigma, is_observed=False)
        mu1, cov1 = self.gp.predict(self.Xnew, pred_noise=True)
        mu2, cov2 = gp.predict(self.Xnew, pred_noise=True)
        npt.assert_allclose(mu1, mu2, atol=0, rtol=1e-3)
        npt.assert_allclose(cov1, cov2, atol=0, rtol=1e-3)


class TestGPAdditive(object):
    def setup_method(self):
        self.X = np.random.randn(50,3)
        self.y = np.random.randn(50)*0.01
        self.Xnew = np.random.randn(60, 3)
        self.noise = pm.gp.cov.WhiteNoise(0.1)
        self.covs = (pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]),
                     pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]),
                     pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]))
        self.means = (pm.gp.mean.Constant(0.5),
                      pm.gp.mean.Constant(0.5),
                      pm.gp.mean.Constant(0.5))

    def testAdditiveMarginal(self):
        with pm.Model() as model1:
            gp1 = pm.gp.Marginal(self.means[0], self.covs[0])
            gp2 = pm.gp.Marginal(self.means[1], self.covs[1])
            gp3 = pm.gp.Marginal(self.means[2], self.covs[2])

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.marginal_likelihood("f", self.X, self.y, noise=self.noise)
            model1_logp = model1.logp({"fsum": self.y})

        with pm.Model() as model2:
            gptot = pm.gp.Marginal(reduce(add, self.means), reduce(add, self.covs))
            fsum = gptot.marginal_likelihood("f", self.X, self.y, noise=self.noise)
            model2_logp = model2.logp({"fsum": self.y})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional("fp1", self.Xnew, given={"X": self.X, "y": self.y,
                                                            "noise": self.noise, "gp": gpsum})
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])
        npt.assert_allclose(fp1.logp({"fp1": fp}), fp2.logp({"fp2": fp}), atol=0, rtol=1e-2)

    @pytest.mark.parametrize('approx', ['FITC', 'VFE', 'DTC'])
    def testAdditiveMarginalSparse(self, approx):
        Xu = np.random.randn(10, 3)
        sigma = 0.1
        with pm.Model() as model1:
            gp1 = pm.gp.MarginalSparse(self.means[0], self.covs[0], approx=approx)
            gp2 = pm.gp.MarginalSparse(self.means[1], self.covs[1], approx=approx)
            gp3 = pm.gp.MarginalSparse(self.means[2], self.covs[2], approx=approx)

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.marginal_likelihood("f", self.X, Xu, self.y, sigma=sigma)
            model1_logp = model1.logp({"fsum": self.y})

        with pm.Model() as model2:
            gptot = pm.gp.MarginalSparse(reduce(add, self.means), reduce(add, self.covs), approx=approx)
            fsum = gptot.marginal_likelihood("f", self.X, Xu, self.y, sigma=sigma)
            model2_logp = model2.logp({"fsum": self.y})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional("fp1", self.Xnew, given={"X": self.X, "Xu": Xu, "y": self.y,
                                                            "sigma": sigma, "gp": gpsum})
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])
        npt.assert_allclose(fp1.logp({"fp1": fp}), fp2.logp({"fp2": fp}), atol=0, rtol=1e-2)

    def testAdditiveLatent(self):
        with pm.Model() as model1:
            gp1 = pm.gp.Latent(self.means[0], self.covs[0])
            gp2 = pm.gp.Latent(self.means[1], self.covs[1])
            gp3 = pm.gp.Latent(self.means[2], self.covs[2])

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.prior("fsum", self.X, reparameterize=False)
            model1_logp = model1.logp({"fsum": self.y})

        with pm.Model() as model2:
            gptot = pm.gp.Latent(reduce(add, self.means), reduce(add, self.covs))
            fsum = gptot.prior("fsum", self.X, reparameterize=False)
            model2_logp = model2.logp({"fsum": self.y})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional("fp1", self.Xnew, given={"X": self.X, "f": self.y, "gp": gpsum})
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])
        npt.assert_allclose(fp1.logp({"fsum": self.y, "fp1": fp}),
                            fp2.logp({"fsum": self.y, "fp2": fp}), atol=0, rtol=1e-2)

    def testAdditiveSparseRaises(self):
        # cant add different approximations
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.MarginalSparse(cov_func=cov_func, approx="DTC")
            gp2 = pm.gp.MarginalSparse(cov_func=cov_func, approx="FITC")
            with pytest.raises(Exception) as e_info:
                gp1 + gp2

    def testAdditiveTypeRaises1(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.MarginalSparse(cov_func=cov_func, approx="DTC")
            gp2 = pm.gp.Marginal(cov_func=cov_func)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2

    def testAdditiveTypeRaises2(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.Latent(cov_func=cov_func)
            gp2 = pm.gp.Marginal(cov_func=cov_func)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2


class TestTP(object):
    R"""
    Compare TP with high degress of freedom to GP
    """
    def setup_method(self):
        X = np.random.randn(20,3)
        y = np.random.randn(20)*0.01
        Xnew = np.random.randn(50, 3)
        pnew = np.random.randn(50)*0.01
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp = pm.gp.Latent(cov_func=cov_func)
            f = gp.prior("f", X, reparameterize=False)
            p = gp.conditional("p", Xnew)
        self.X = X
        self.y = y
        self.Xnew = Xnew
        self.pnew = pnew
        self.latent_logp = model.logp({"f": y, "p": pnew})
        self.plogp = p.logp({"f": y, "p": pnew})

    def testTPvsLatent(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            tp = pm.gp.TP(cov_func=cov_func, nu=10000)
            f = tp.prior("f", self.X, reparameterize=False)
            p = tp.conditional("p", self.Xnew)
        tp_logp = model.logp({"f": self.y, "p": self.pnew})
        npt.assert_allclose(self.latent_logp, tp_logp, atol=0, rtol=1e-2)

    def testTPvsLatentReparameterized(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            tp = pm.gp.TP(cov_func=cov_func, nu=10000)
            f = tp.prior("f", self.X, reparameterize=True)
            p = tp.conditional("p", self.Xnew)
        chol = np.linalg.cholesky(cov_func(self.X).eval())
        y_rotated = np.linalg.solve(chol, self.y)
        # testing full model logp unreliable due to introduction of chi2__log__
        plogp = p.logp({"f_rotated_": y_rotated, "p": self.pnew, "chi2__log__": np.log(1e20)})
        npt.assert_allclose(self.plogp, plogp, atol=0, rtol=1e-2)

    def testAdditiveTPRaises(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.TP(cov_func=cov_func, nu=10)
            gp2 = pm.gp.TP(cov_func=cov_func, nu=10)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2












