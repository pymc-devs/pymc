import unittest
from pymc3 import Model, gp
import theano
import theano.tensor as tt
import numpy as np

class ExpQuadTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.ExpQuad(1, 0.1)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.53940, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.ExpQuad(2, 0.5)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.820754, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.ExpQuad(2, np.array([1, 2]))
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.969607, 3)

    def test_dimerrors(self):
        def err1():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.ExpQuad(1, 0.1)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err1)

        def err2():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.ExpQuad(2, np.array([1,2,3]))
            K = theano.function([], cov.K(X))()
        self.assertRaises(err2)


class RatQuadTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.RatQuad(1, 0.1, 0.5)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.66896, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.RatQuad(2, 0.5, 0.5)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.84664, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.RatQuad(2, np.array([1, 2]), 0.5)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.97049, 3)

    def test_dimerrors(self):
        def err1():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.RatQuad(1, 0.1, 0.5)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err1)

        def err2():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.RatQuad(2, np.array([1,2,3]), 0.5)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err2)


class ExponentialTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Exponential(1, 0.1)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.57375, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Exponential(2, 0.5)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.73032, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Exponential(2, np.array([1, 2]))
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.88318, 3)

    def test_dimerrors(self):
        def err1():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Exponential(1, 0.1)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err1)

        def err2():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Exponential(2, np.array([1,2,3]))
            K = theano.function([], cov.K(X))()
        self.assertRaises(err2)


class Matern52Test(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Matern52(1, 0.1)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.46202, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Matern52(2, 0.5)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.75143, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Matern52(2, np.array([1, 2]))
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.95153, 3)

    def test_dimerrors(self):
        def err1():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Matern52(1, 0.1)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err1)

        def err2():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Matern52(2, np.array([1,2,3]))
            K = theano.function([], cov.K(X))()
        self.assertRaises(err2)


class Matern32Test(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Matern32(1, 0.1)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.42682, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Matern32(2, 0.5)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.70318, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Matern32(2, np.array([1, 2]))
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.930135, 3)

    def test_dimerrors(self):
        def err1():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Matern32(1, 0.1)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err1)

        def err2():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Matern32(2, np.array([1,2,3]))
            K = theano.function([], cov.K(X))()
        self.assertRaises(err2)


class LinearTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Linear(1, 0.5)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.19444, 3)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Linear(2, 0.2)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], -0.01629, 3)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Linear(2, np.array([0.2, -0.1]))
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.08703, 3)

    def test_dimerrors(self):
        def err1():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Linear(1, 0.1)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err1)

        def err2():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Linear(2, np.array([1,2,3]))
            K = theano.function([], cov.K(X))()
        self.assertRaises(err2)


class PolynomialTest(unittest.TestCase):
    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov = gp.cov.Polynomial(1, 0.5, 2, 0)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.03780, 4)

    def test_2d(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Polynomial(2, 0.2, 2, 0)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.00026, 4)

    def test_2dard(self):
        X = np.linspace(0,1,10).reshape(5,2)
        with Model() as model:
            cov = gp.cov.Polynomial(2, np.array([0.2, -0.1]), 2, 0)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[0,1], 0.00757, 4)

    def test_dimerrors(self):
        def err1():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Polynomial(1, 0.1, 2, 0)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err1)

        def err2():
            X = np.random.randn(10,2)
            with Model() as model:
                cov = gp.cov.Polynomial(2, np.array([1,2,3]), 2, 0)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err2)


class WarpedInputTest(unittest.TestCase):
    def warp_func(self, x, a, b, c):
        return x + (a * tt.tanh(b * (x - c)))

    def test_1d(self):
        X = np.linspace(0,1,10)[:,None]
        with Model() as model:
            cov_m52 = gp.cov.Matern52(1, 0.2)
            cov = gp.cov.WarpedInput(1, warp_func=self.warp_func, args=(1,10,1), cov_func=cov_m52)
        K = theano.function([], cov.K(X))()
        self.assertAlmostEqual(K[4,5], 0.795196, 4)

    def test_dimerrors(self):
        def err():
            X = np.random.randn(10,2)
            with Model() as model:
                cov_m52 = gp.cov.Matern52(1, 0.2)
                cov = gp.cov.WarpedInput(1, warp_func=warp_func, args=(1,10,1), cov_func=cov_m52)
            K = theano.function([], cov.K(X))()
        self.assertRaises(err)


