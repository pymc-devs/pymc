#  pylint:disable=unused-variable
from .helpers import SeededTest
from pymc3 import Model, gp, sample, Uniform
from pymc3.gp.gp import GP, GPFullNonConjugate, GPFullConjugate, GPSparseConjugate
from pymc3 import Normal
import theano
import theano.tensor as tt
import numpy as np
from scipy.linalg import cholesky as sp_cholesky
from scipy.linalg import solve_triangular as sp_solve_triangular
import itertools
import numpy.testing as npt
import pytest


class TestZeroMean(object):
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            zero_mean = gp.mean.Zero()
        M = theano.function([], zero_mean(X))()
        assert np.all(M==0)
        assert M.shape == (10, )


class TestConstantMean(object):
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            const_mean = gp.mean.Constant(6)
        M = theano.function([], const_mean(X))()
        assert np.all(M==6)
        assert M.shape == (10, )


class TestLinearMean(object):
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            linear_mean = gp.mean.Linear(2, 0.5)
        M = theano.function([], linear_mean(X))()
        npt.assert_allclose(M[1], 0.7222, atol=1e-3)
        assert M.shape == (10, )


class TestAddProdMean(object):
    def test_add(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            mean1 = gp.mean.Linear(coeffs=2, intercept=0.5)
            mean2 = gp.mean.Constant(2)
            mean = mean1 + mean2 + mean2
        M = theano.function([], mean(X))()
        npt.assert_allclose(M[1], 0.7222 + 2 + 2, atol=1e-3)

    def test_prod(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            mean1 = gp.mean.Linear(coeffs=2, intercept=0.5)
            mean2 = gp.mean.Constant(2)
            mean = mean1 * mean2 * mean2
        M = theano.function([], mean(X))()
        npt.assert_allclose(M[1], 0.7222 * 2 * 2, atol=1e-3)

    def test_add_multid(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        A = np.array([1, 2, 3])
        b = 10
        with Model() as model:
            mean1 = gp.mean.Linear(coeffs=A, intercept=b)
            mean2 = gp.mean.Constant(2)
            mean = mean1 + mean2 + mean2
        M = theano.function([], mean(X))()
        npt.assert_allclose(M[1], 10.8965 + 2 + 2, atol=1e-3)

    def test_prod_multid(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        A = np.array([1, 2, 3])
        b = 10
        with Model() as model:
            mean1 = gp.mean.Linear(coeffs=A, intercept=b)
            mean2 = gp.mean.Constant(2)
            mean = mean1 * mean2 * mean2
        M = theano.function([], mean(X))()
        npt.assert_allclose(M[1], 10.8965 * 2 * 2, atol=1e-3)


class TestCovAdd(object):
    def test_symadd_cov(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            cov1 = gp.cov.ExpQuad(1, 0.1)
            cov2 = gp.cov.ExpQuad(1, 0.1)
            cov = cov1 + cov2
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightadd_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            a = 1
            cov = gp.cov.ExpQuad(1, 0.1) + a
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 1.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftadd_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            a = 1
            cov = a + gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 1.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightadd_matrix(self):
        X = np.linspace(0, 1, 10)[:, None]
        M = 2 * np.ones((10, 10))
        with Model() as model:
            cov = gp.cov.ExpQuad(1, 0.1) + M
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_matrix(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with Model() as model:
            cov = M + gp.cov.ExpQuad(1, 0.1)
            cov_true = gp.cov.ExpQuad(1, 0.1) + M
        K = theano.function([], cov(X))()
        K_true = theano.function([], cov_true(X))()
        assert np.allclose(K, K_true)


class TestCovProd(object):
    def test_symprod_cov(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            cov1 = gp.cov.ExpQuad(1, 0.1)
            cov2 = gp.cov.ExpQuad(1, 0.1)
            cov = cov1 * cov2
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.53940 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightprod_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            a = 2
            cov = gp.cov.ExpQuad(1, 0.1) * a
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_scalar(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            a = 2
            cov = a * gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_rightprod_matrix(self):
        X = np.linspace(0, 1, 10)[:, None]
        M = 2 * np.ones((10, 10))
        with Model() as model:
            cov = gp.cov.ExpQuad(1, 0.1) * M
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 2 * 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_leftprod_matrix(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with Model() as model:
            cov = M * gp.cov.ExpQuad(1, 0.1)
            cov_true = gp.cov.ExpQuad(1, 0.1) * M
        K = theano.function([], cov(X))()
        K_true = theano.function([], cov_true(X))()
        assert np.allclose(K, K_true)

    def test_multiops(self):
        X = np.linspace(0, 1, 3)[:, None]
        M = np.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]])
        with Model() as model:
            cov1 = 3 + gp.cov.ExpQuad(1, 0.1) + M * gp.cov.ExpQuad(1, 0.1) * M * gp.cov.ExpQuad(1, 0.1)
            cov2 = gp.cov.ExpQuad(1, 0.1) * M * gp.cov.ExpQuad(1, 0.1) * M + gp.cov.ExpQuad(1, 0.1) + 3
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
        with Model() as model:
            cov = gp.cov.ExpQuad(3, 0.1, active_dims=[0, 0, 1])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.20084298, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_slice2(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with Model() as model:
            cov = gp.cov.ExpQuad(3, [0.1, 0.1], active_dims=[False, True, True])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.34295549, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_slice3(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with Model() as model:
            cov = gp.cov.ExpQuad(3, np.array([0.1, 0.1]), active_dims=[False, True, True])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.34295549, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_diffslice(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        with Model() as model:
            cov = gp.cov.ExpQuad(3, 0.1, [1, 0, 0]) + gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.683572, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        lengthscales = 2.0
        with pytest.raises(ValueError):
            gp.cov.ExpQuad(1, lengthscales, [True, False])
            gp.cov.ExpQuad(2, lengthscales, [True])


class TestStability(object):
    def test_stable(self):
        X = np.random.uniform(low=320., high=400., size=[2000, 2])
        with Model() as model:
            cov = gp.cov.ExpQuad(2, 0.1)
        dists = theano.function([], cov.square_dist(X, X))()
        assert not np.any(dists < 0)


class TestExpQuad(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            cov = gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.53940, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_2d(self):
        X = np.linspace(0, 1, 10).reshape(5, 2)
        with Model() as model:
            cov = gp.cov.ExpQuad(2, 0.5)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.820754, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_2dard(self):
        X = np.linspace(0, 1, 10).reshape(5, 2)
        with Model() as model:
            cov = gp.cov.ExpQuad(2, np.array([1, 2]))
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.969607, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestRatQuad(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            cov = gp.cov.RatQuad(1, 0.1, 0.5)
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
        with Model() as model:
            cov = gp.cov.Exponential(1, 0.1)
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
        with Model() as model:
            cov = gp.cov.Matern52(1, 0.1)
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
        with Model() as model:
            cov = gp.cov.Matern32(1, 0.1)
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
        with Model() as model:
            cov = gp.cov.Cosine(1, 0.1)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], -0.93969, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], -0.93969, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)


class TestLinear(object):
    def test_1d(self):
        X = np.linspace(0, 1, 10)[:, None]
        with Model() as model:
            cov = gp.cov.Linear(1, 0.5)
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
        with Model() as model:
            cov = gp.cov.Polynomial(1, 0.5, 2, 0)
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
        with Model() as model:
            cov_m52 = gp.cov.Matern52(1, 0.2)
            cov = gp.cov.WarpedInput(1, warp_func=warp_func, args=(1, 10, 1), cov_func=cov_m52)
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[0, 1], 0.79593, atol=1e-3)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[0, 1], 0.79593, atol=1e-3)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        cov_m52 = gp.cov.Matern52(1, 0.2)
        with pytest.raises(TypeError):
            gp.cov.WarpedInput(1, cov_m52, "str is not callable")
        with pytest.raises(TypeError):
            gp.cov.WarpedInput(1, "str is not Covariance object", lambda x: x)


class TestGibbs(object):
    def test_1d(self):
        X = np.linspace(0, 2, 10)[:, None]
        def tanh_func(x, x1, x2, w, x0):
            return (x1 + x2) / 2.0 - (x1 - x2) / 2.0 * tt.tanh((x - x0) / w)
        with Model() as model:
            cov = gp.cov.Gibbs(1, tanh_func, args=(0.05, 0.6, 0.4, 1.0))
        K = theano.function([], cov(X))()
        npt.assert_allclose(K[2, 3], 0.136683, atol=1e-4)
        K = theano.function([], cov(X, X))()
        npt.assert_allclose(K[2, 3], 0.136683, atol=1e-4)
        # check diagonal
        Kd = theano.function([], cov(X, diag=True))()
        npt.assert_allclose(np.diag(K), Kd, atol=1e-5)

    def test_raises(self):
        with pytest.raises(TypeError):
            gp.cov.Gibbs(1, "str is not callable")
        with pytest.raises(NotImplementedError):
            gp.cov.Gibbs(2, lambda x: x)
        with pytest.raises(NotImplementedError):
            gp.cov.Gibbs(3, lambda x: x, active_dims=[True, True, False])


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
        func_noargs2 = gp.cov.handle_args(func_noargs, None)
        func_onearg2 = gp.cov.handle_args(func_onearg, a)
        func_twoarg2 = gp.cov.handle_args(func_twoarg, args=(a, b))
        assert func_noargs(x) == func_noargs2(x, args=None)
        assert func_onearg(x, a) == func_onearg2(x, args=a)
        assert func_twoarg(x, a, b) == func_twoarg2(x, args=(a, b))


"""The following set of tests use the conjugate gp without approx
as a baseline  for the other gp models.  Fixtures for three
different data sets are fed into each GP, and the key computations
performed by each GP* class (logp, prior, conditional) are tested
on each data set."""
@pytest.fixture(scope='module', params=["1D", "2D", "3D"])
def data(request):
    # generate data sets with different noise levels and dimension
    np.random.seed(100)
    if request.param == "1D":
        d, sigma = 1, 0.1
    if request.param == "2D":
        d, sigma = 2, 0.3
    if request.param == "3D":
        d, sigma = 3, 0.2
    # X, Xs, y, mu, cov, cov_n, sigma
    n_data = 100
    return (np.random.rand(n_data, d), np.random.rand(n_data, d),
            np.random.randn(n_data), gp.mean.Zero(),
            gp.cov.ExpQuad(d, 0.1), gp.cov.WhiteNoise(d, sigma), sigma)


@pytest.fixture(scope='module')
def nonconj_full(data):
    X, Xs, y, mu, cov, cov_n, sigma = data
    # add white noise covariance so cov is equivalent to the conjugate
    #   gp, which includes the variance from the normal likelihood.
    with Model() as model:
        input_dim = X.shape[0]
        gp_cls = GPFullNonConjugate("f", X, mu, cov + cov_n)
        f = gp_cls.RV
    return f


@pytest.fixture(scope='module')
def conj_full(data):
    X, Xs, y, mu, cov, cov_n, sigma = data
    with Model() as model:
        f = GPFullConjugate("f", X, mu, cov, cov_n, observed=y)
    return f


@pytest.fixture(scope='module')
def conj_fitc(data):
    # use inputs for inducing points so results should nearly match full conj gp
    X, Xs, y, mu, cov, cov_n, sigma = data
    with Model() as model:
        f = GPSparseConjugate("f", X, mu, cov, sigma, "FITC", X, observed=y)
    return f


@pytest.fixture(scope='module')
def conj_vfe(data):
    # use inputs for inducing points so results should nearly match full conj gp
    X, Xs, y, mu, cov, cov_n, sigma = data
    with Model() as model:
        f = GPSparseConjugate("f", X, mu, cov, sigma, "VFE", X, observed=y)
    return f


# the non-approximated conjugate GP is used as a baseline for comparison
class TestConjFITC(object):
    """ Test that the conjugate GP with the FITC approx is similar to the
    full conjugate GP.  Inducing points are placed at same location as the inputs,
    so the results should be approximately the same.  As the number of inducing points
    decreases, the approximation worsens.
    """
    def test_logp(self, data, conj_full, conj_fitc):
        X, Xs, y, mu, cov, cov_n, sigma = data
        np.testing.assert_allclose(conj_full.distribution.logp(y=y).eval(),
                                   conj_fitc.distribution.logp(y=y).eval(), atol=0, rtol=1e-3)

    def test_conditional_with_obsnoise(self, data, conj_full, conj_fitc):
        X, Xs, y, mu, cov, cov_n, sigma = data
        c_mu1, c_cov1 = conj_full.distribution.conditional(Xs, y, obs_noise=True)
        c_mu2, c_cov2 = conj_fitc.distribution.conditional(Xs, y, obs_noise=True)
        np.testing.assert_allclose(c_mu1.eval(), c_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(c_cov1.eval(), c_cov2.eval(), atol=0.1, rtol=1e-3)

    def test_conditional_without_obsnoise(self, data, conj_full, conj_fitc):
        X, Xs, y, mu, cov, cov_n, sigma = data
        c_mu1, c_cov1 = conj_full.distribution.conditional(Xs, y, obs_noise=False)
        c_mu2, c_cov2 = conj_fitc.distribution.conditional(Xs, y, obs_noise=False)
        np.testing.assert_allclose(c_mu1.eval(), c_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(c_cov1.eval(), c_cov2.eval(), atol=0.1, rtol=1e-3)

    def test_prior_with_obsnoise(self, data, conj_full, conj_fitc):
        p_mu1, p_cov1 = conj_full.distribution.prior()
        p_mu2, p_cov2 = conj_fitc.distribution.prior()
        np.testing.assert_allclose(p_mu1.eval(), p_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(p_cov1.eval(), p_cov2.eval(), atol=0.1, rtol=1e-3)

    def test_prior_without_obsnoise(self, data, conj_full, conj_fitc):
        p_mu1, p_cov1 = conj_full.distribution.prior(obs_noise=True)
        p_mu2, p_cov2 = conj_fitc.distribution.prior(obs_noise=True)
        np.testing.assert_allclose(p_mu1.eval(), p_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(p_cov1.eval(), p_cov2.eval(), atol=0.1, rtol=1e-3)


class TestConjVFE(object):
    """ Test that the conjugate GP with the VFE approx is similar to the
    full conjugate GP.  Inducing points are placed at same location as the inputs,
    so the results should be approximately the same.  As the number of inducing points
    decreases, the approximation worsens.
    """
    def test_logp(self, data, conj_full, conj_vfe):
        X, Xs, y, mu, cov, cov_n, sigma = data
        np.testing.assert_allclose(conj_full.distribution.logp(y=y).eval(),
                                   conj_vfe.distribution.logp(y=y).eval(), atol=0, rtol=1e-3)

    def test_conditional_with_obsnoise(self, data, conj_full, conj_vfe):
        X, Xs, y, mu, cov, cov_n, sigma = data
        c_mu1, c_cov1 = conj_full.distribution.conditional(Xs, y, obs_noise=True)
        c_mu2, c_cov2 = conj_vfe.distribution.conditional(Xs, y, obs_noise=True)
        np.testing.assert_allclose(c_mu1.eval(), c_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(c_cov1.eval(), c_cov2.eval(), atol=0.1, rtol=1e-3)

    def test_conditional_without_obsnoise(self, data, conj_full, conj_vfe):
        X, Xs, y, mu, cov, cov_n, sigma = data
        c_mu1, c_cov1 = conj_full.distribution.conditional(Xs, y, obs_noise=False)
        c_mu2, c_cov2 = conj_vfe.distribution.conditional(Xs, y, obs_noise=False)
        np.testing.assert_allclose(c_mu1.eval(), c_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(c_cov1.eval(), c_cov2.eval(), atol=0.1, rtol=1e-3)

    def test_prior_with_obsnoise(self, data, conj_full, conj_vfe):
        p_mu1, p_cov1 = conj_full.distribution.prior()
        p_mu2, p_cov2 = conj_vfe.distribution.prior()
        np.testing.assert_allclose(p_mu1.eval(), p_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(p_cov1.eval(), p_cov2.eval(), atol=0.1, rtol=1e-3)

    def test_prior_with_obsnoise(self, data, conj_full, conj_vfe):
        p_mu1, p_cov1 = conj_full.distribution.prior(obs_noise=True)
        p_mu2, p_cov2 = conj_vfe.distribution.prior(obs_noise=True)
        np.testing.assert_allclose(p_mu1.eval(), p_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(p_cov1.eval(), p_cov2.eval(), atol=0.1, rtol=1e-3)


class TestNonConjvsConj(object):
    """ Test that the non conjugate GP is identical to the conjugate GP
    (no approx) noise has been applied to the kernel of the non
    conjugate GP so that it matches the conjugate GP fixture.  In order
    for nonconjugate GP's `conditional` method to be comparable to the
    conjugate GP's `conditional` method, the WhiteNoise covariance is
    included in the nonconjugate GP model that is specified in the
    tests.  There aren't special tests for when the obs_noise flag is
    set to True because observation noise doesn't make sense in the
    nonconjugate GP model.
    """
    def test_logp(self, data, conj_full, nonconj_full):
        X, Xs, y, mu, cov, cov_n, sigma = data
        # nonconj_full gp is deterministic, so logp comes from GPBase, which
        # defaults to zero.
        assert nonconj_full.distribution.logp(y=y) == 0.0

    def test_conditional(self, data, conj_full, nonconj_full):
        X, Xs, y, mu, cov, cov_n, sigma = data
        # replace self.v in nonconj_full with rotated version of y for testing
        K = (cov(X) + cov_n(X)).eval()
        L = sp_cholesky(K, lower=True)
        v = sp_solve_triangular(L, y, lower=True)

        c_mu1, c_cov1 = conj_full.distribution.conditional(Xs, y, obs_noise=False)
        c_mu2, c_cov2 = nonconj_full.distribution.conditional(Xs, v=v)
        np.testing.assert_allclose(c_mu1.eval(), c_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(c_cov1.eval(), c_cov2.eval(), atol=0.1, rtol=1e-3)

    def test_prior(self, data, conj_full, nonconj_full):
        p_mu1, p_cov1 = conj_full.distribution.prior(obs_noise=True)
        p_mu2, p_cov2 = nonconj_full.distribution.prior()
        np.testing.assert_allclose(p_mu1.eval(), p_mu2.eval(), atol=0.1, rtol=1e-3)
        np.testing.assert_allclose(p_cov1.eval(), p_cov2.eval(), atol=0.1, rtol=1e-3)


