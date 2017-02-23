#  pylint:disable=unused-variable
from .helpers import SeededTest
import unittest
from pymc3 import Model, gp, sample, Uniform
import theano
import theano.tensor as tt
import numpy as np

class ZeroTest(unittest.TestCase):
    def test_value(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            zero_mean = gp.mean.Zero()
        M = theano.function([], zero_mean(X))()
        self.assertTrue(np.all(M==0))
        self.assertSequenceEqual(M.shape, (10,1))

class ConstantTest(unittest.TestCase):
    def test_value(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            const_mean = gp.mean.Constant(6)
        M = theano.function([], const_mean(X))()
        self.assertTrue(np.all(M==6))
        self.assertSequenceEqual(M.shape, (10,1))

class LinearMeanTest(unittest.TestCase):
    def test_value(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            linear_mean = gp.mean.Linear(2, 0.5)
        M = theano.function([], linear_mean(X))()
        self.assertAlmostEqual(M[1,0], 0.7222, 3)

class ExpQuadTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.53940, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.ExpQuad(2, 0.5)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.820754, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.ExpQuad(2, np.array([1, 2]))
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.969607, 3)


class RatQuadTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.RatQuad(1, 0.1, 0.5)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.66896, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.RatQuad(2, 0.5, 0.5)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.84664, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.RatQuad(2, np.array([1, 2]), 0.5)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.97049, 3)


class ExponentialTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Exponential(1, 0.1)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.57375, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Exponential(2, 0.5)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.73032, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Exponential(2, np.array([1, 2]))
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.88318, 3)


class Matern52Test(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Matern52(1, 0.1)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.46202, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Matern52(2, 0.5)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.75143, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Matern52(2, np.array([1, 2]))
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.95153, 3)


class Matern32Test(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Matern32(1, 0.1)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.42682, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Matern32(2, 0.5)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.70318, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Matern32(2, np.array([1, 2]))
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.930135, 3)


class LinearTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Linear(1, 0.5)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.19444, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Linear(2, 0.2)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], -0.01629, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Linear(2, np.array([0.2, -0.1]))
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.08703, 3)


class PolynomialTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Polynomial(1, 0.5, 2, 0)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.03780, 4)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Polynomial(2, 0.2, 2, 0)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.00026, 4)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Polynomial(2, np.array([0.2, -0.1]), 2, 0)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.00757, 4)


class WarpedInputTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        def warp_func(x, a, b, c):
            return x + (a * tt.tanh(b * (x - c)))
        with Model() as model:
            cov_m52 = gp.cov.Matern52(1, 0.2)
            cov = gp.cov.WarpedInput(1, warp_func=warp_func, args=(1,10,1), cov_func=cov_m52)
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[0,1], 0.79593, 4)

class GibbsTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0, 2, 10)[:,None]
        def tanh_func(x, x1, x2, w, x0):
            return (x1 + x2) / 2.0 - (x1 - x2) / 2.0 * tt.tanh((x - x0) / w)
        with Model() as model:
            cov = gp.cov.Gibbs(1, tanh_func, args=(0.05, 0.6, 0.4, 1.0))
        K = theano.function([], cov(X))()
        self.assertAlmostEqual(K[2,3], 0.136683, 4)

class GPTest(SeededTest):
    def test_func_args(self):
        X = np.linspace(0,1,10)[:,None]
        Y = np.random.randn(10,1)
        with Model() as model:
            # make a Gaussian model
            with self.assertRaises(ValueError):
                random_test = gp.GP('random_test', cov_func=gp.mean.Zero(), observed={'X':X, 'Y':Y})
            with self.assertRaises(ValueError):
                random_test = gp.GP('random_test', mean_func=gp.cov.Matern32(1, 1),
                                        cov_func=gp.cov.Matern32(1, 1), observed={'X':X, 'Y':Y})

    def test_sample(self):
        X = np.linspace(0,1,100)[:,None]
        Y = np.random.randn(100,1)
        with Model() as model:
            M = gp.mean.Zero()
            l = Uniform('l', 0, 5)
            K = gp.cov.Matern32(1, l)
            sigma = Uniform('sigma', 0, 10)

            # make a Gaussian model
            random_test = gp.GP('random_test', mean_func=M, cov_func=K, sigma=sigma, observed={'X':X, 'Y':Y})

            tr = sample(500, init=None, progressbar=False, random_seed=self.random_seed)
