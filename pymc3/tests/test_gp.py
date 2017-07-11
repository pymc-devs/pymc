#  pylint:disable=unused-variable
from .helpers import SeededTest
from pymc3 import Model, gp, sample, Uniform
from pymc3.gp.gp import GP, GPFullNonConjugate, GPFullConjugate, GPSparseConjugate
from pymc3 import Normal
import theano
import theano.tensor as tt
import numpy as np
from scipy.linalg import cholesky as spcholesky, solve_triangular
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


"""
X_ = (np.linspace(0,1,10)[:,None], np.random.rand(10,2), None)
mean_func_ = (gp.mean.Zero(), None)
cov_func_ = (gp.cov.ExpQuad(1, 0.1), gp.cov.ExpQuad(2, 0.1), None)
cov_func_noise_ = (gp.cov.Matern32(1, 0.1), gp.cov.Matern32(2, 0.1), None)
sigma_ = (0.2, None)
approx_ = ("vfe", "FITC", "fitc", None, "whatever")
n_inducing_ = (10, None)
inducing_points_ = (np.linspace(0,1,10)[:,None], np.random.rand(10,2), None)
observed_ = (np.random.rand(10), None)
chol_const_ = (True, None)

@pytest.mark.parametrize('X,mean_func,cov_func,cov_func_noise,sigma,approx,n_inducing,inducing_points,observed,chol_const',
    itertools.product(X_,mean_func_,cov_func_,cov_func_noise_,sigma_,approx_,
                      n_inducing_,inducing_points_,observed_,chol_const_)
)
#def test_gp_constructor(inputs):
def test_gp_constructor(X,mean_func,cov_func,cov_func_noise,sigma,approx,n_inducing,
                        inducing_points,observed,chol_const):
    with Model() as model:
        if n_inducing is None and inducing_points is None:

        f = GP("f", X,mean_func,cov_func,cov_func_noise,sigma,approx,n_inducing,
                inducing_points,observed,chol_const,model)
    pass

"""


def normdiff(a, b):
    return np.linalg.norm(a - b)

""" data sets for testing GP models """
@pytest.fixture(scope='module')
def md1():
    # one dimensional GP with 50 data points
    #   (md = "model data")
    np.random.seed(100)
    return {"X": np.linspace(0, 1, 50)[:, None],
            "Z": np.linspace(0, 1, 100)[:, None],
            "y": np.random.randn(50),
            "mu": gp.mean.Zero(),
            "cov": gp.cov.ExpQuad(1, 0.1),
            "sigma": 0.1}

@pytest.fixture(scope='module')
def md2():
    # 2 dimensional GP with 20 data points
    np.random.seed(100)
    return {"X": np.random.rand(20,2),
            "Z": np.random.rand(40,2),
            "y": np.random.randn(20),
            "mu": gp.mean.Zero(),
            "cov": gp.cov.ExpQuad(2, 0.1),
            "sigma": 0.1}

@pytest.fixture(scope='module')
def md3():
    # one dimensional GP with 50 data points, tiny observation noise
    np.random.seed(100)
    return {"X": np.linspace(0, 1, 50)[:, None],
            "Z": np.linspace(0, 1, 100)[:, None],
            "y": np.random.randn(50),
            "mu": gp.mean.Zero(),
            "cov": gp.cov.ExpQuad(1, 0.1),
            "sigma": 1e-6}

## 4 GP models, 3 data sets -> 12 models that should be precompiled before comparisons

def conj_full(md):
    with Model() as model:
        f = GPFullConjugate("f", X=md["X"], mean_func=md["mu"], cov_func=md["cov"],
                            cov_func_noise=lambda X: tt.square(md["sigma"]) * tt.eye(X.shape[0]),
                            observed=md["y"])
        ####### HERE > better way to organize output??  these evals dont need to occur in the model block
        n_cond_mu, n_cond_cov = [x.eval() for x in f.distribution.conditional(md["Z"], md["y"], obs_noise=True)]
        cond_mu, cond_cov = [x.eval() for x in f.distribution.conditional(md["Z"], md["y"], obs_noise=False)]
        return {"logp": f.distribution.logp(md["y"]).eval(),
                "cond_n": [x.eval() for x in f.distribution.conditional(md["Z"], md["y"], obs_noise=True)],
                "cond": [x.eval() for x in f.distribution.conditional(md["Z"], md["y"], obs_noise=False)],
                "prior_n": [x.eval() for x in f.distribution.prior(True)],
                "prior": [x.eval() for x in f.distribution.prior(False)]}

def nonconj_full(md):
    # rotate y into v using cholesky decomp of K
    K = md["cov"](md["X"]).eval() + 1e-6*np.eye(md["X"].shape[0])
    L = spcholesky(K, lower=True)
    v = solve_triangular(L, md["y"], lower=True)
    with Model() as model:
        gp = GPFullNonConjugate("f", X=md["X"], mean_func=md["mu"], cov_func=md["cov"])
        f = gp.RV
        return {"logp": f.distribution.logp(None),
                "cond": [x.eval() for x in f.distribution.conditional(md["Z"], v=v)],
                "prior": [x.eval() for x in f.distribution.prior()]}


def conj_fitc(md):
    with Model() as model:
        f  = GPSparseConjugate("f", approx="FITC", X=md["X"], mean_func=md["mu"], cov_func=md["cov"],
                               sigma=md["sigma"], inducing_points=md["X"], observed=md["y"])
        return {"logp": f.distribution.logp(md["y"]).eval(),
                "cond_n": [x.eval() for x in f.distribution.conditional(md["Z"], md["y"], obs_noise=True)],
                "cond": [x.eval() for x in f.distribution.conditional(md["Z"], md["y"], obs_noise=False)],
                "prior_n": [x.eval() for x in f.distribution.prior(True)],
                "prior": [x.eval() for x in f.distribution.prior(False)]}

def conj_vfe(md):
    with Model() as model:
        f  = GPSparseConjugate("f", approx="VFE", X=md["X"], mean_func=md["mu"], cov_func=md["cov"],
                               sigma=md["sigma"], inducing_points=md["X"], observed=md["y"])
        return {"logp": f.distribution.logp(md["y"]).eval(),
                "cond_n": [x.eval() for x in f.distribution.conditional(md["Z"], md["y"], obs_noise=True)],
                "cond": [x.eval() for x in f.distribution.conditional(md["Z"], md["y"], obs_noise=False)],
                "prior_n": [x.eval() for x in f.distribution.prior(True)],
                "prior": [x.eval() for x in f.distribution.prior(False)]}

### HERE Parameterize these tests better??
def test_nonconj():
    assert nonconj


# test non conjugate GP model
def test_nonconj_logp(md):
    assert nonconj_results["logp"] == 0.0

def test_nonconj_cond(nonconj_results, conj_full_results):
    # should match nonconjugate case without observation noise
    mu1, cov1 = nonconj_results["cond"]
    mu2, cov2 = conj_full_results["cond"]
    #assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3

def test_nonconj_prior(nonconj_results, md1):
    mu1 = md1["mu"](md1["X"]).eval()
    cov1 = md1["cov"](md1["X"]).eval()
    mu2, cov2 = nonconj_results["prior"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3

# compare full, conjugate model to fitc with inducing points at inputs
def test_fitc_logp(conj_full_results, conj_fitc_results):
    np.testing.assert_almost_equal(conj_full_results["logp"],
                                   conj_fitc_results["logp"], 0)

def test_fitc_cond(conj_full_results, conj_fitc_results):
    # without the noise in observations
    mu1, cov1 = conj_full_results["cond"]
    mu2, cov2 = conj_fitc_results["cond"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3

    # with the noise in observations
    mu1, cov1 = conj_full_results["cond_n"]
    mu2, cov2 = conj_fitc_results["cond_n"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3

def test_fitc_prior(conj_full_results, conj_fitc_results):
    # without the noise in observations
    mu1, cov1 = conj_full_results["prior"]
    mu2, cov2 = conj_fitc_results["prior"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3

    # with the noise in observations
    mu1, cov1 = conj_full_results["prior_n"]
    mu2, cov2 = conj_fitc_results["prior_n"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3


# compare full, conjugate model to vfe with inducing points at inputs
def test_vfe_logp(conj_full_results, conj_vfe_results):
    np.testing.assert_almost_equal(conj_full_results["logp"],
                                   conj_vfe_results["logp"], 0)

def test_vfe_cond(conj_full_results, conj_vfe_results):
    # without the noise in observations
    mu1, cov1 = conj_full_results["cond"]
    mu2, cov2 = conj_vfe_results["cond"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3

    # with the noise in observations
    mu1, cov1 = conj_full_results["cond_n"]
    mu2, cov2 = conj_vfe_results["cond_n"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3

def test_vfe_prior(conj_full_results, conj_vfe_results):
    # without the noise in observations
    mu1, cov1 = conj_full_results["prior"]
    mu2, cov2 = conj_vfe_results["prior"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3

    # with the noise in observations
    mu1, cov1 = conj_full_results["prior_n"]
    mu2, cov2 = conj_vfe_results["prior_n"]
    assert normdiff(mu1, mu2) < 1e-3
    assert normdiff(cov1, cov2) < 1e-3




