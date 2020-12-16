#   Copyright 2020 The PyMC Developers
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

import itertools
import pickle

import numpy as np
import pytest
import theano

from scipy import stats as stats
from theano import tensor as tt

import pymc3 as pm

from pymc3 import GeneratorAdapter, Normal, floatX, generator, tt_rng
from pymc3.tests.helpers import select_by_precision
from pymc3.theanof import GeneratorOp


class _DataSampler:
    """
    Not for users
    """

    def __init__(self, data, batchsize=50, random_seed=42, dtype="floatX"):
        self.dtype = theano.config.floatX if dtype == "floatX" else dtype
        self.rng = np.random.RandomState(random_seed)
        self.data = data
        self.n = batchsize

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.rng.uniform(size=self.n, low=0.0, high=self.data.shape[0] - 1e-16).astype(
            "int64"
        )
        return np.asarray(self.data[idx], self.dtype)

    next = __next__


@pytest.fixture(scope="module")
def datagen():
    return _DataSampler(np.random.uniform(size=(1000, 10)))


def integers():
    i = 0
    while True:
        yield pm.floatX(i)
        i += 1


def integers_ndim(ndim):
    i = 0
    while True:
        yield np.ones((2,) * ndim) * i
        i += 1


@pytest.mark.usefixtures("strict_float32")
class TestGenerator:
    def test_basic(self):
        generator = GeneratorAdapter(integers())
        gop = GeneratorOp(generator)()
        assert gop.tag.test_value == np.float32(0)
        f = theano.function([], gop)
        assert f() == np.float32(0)
        assert f() == np.float32(1)
        for _ in range(2, 100):
            f()
        assert f() == np.float32(100)

    def test_ndim(self):
        for ndim in range(10):
            res = list(itertools.islice(integers_ndim(ndim), 0, 2))
            generator = GeneratorAdapter(integers_ndim(ndim))
            gop = GeneratorOp(generator)()
            f = theano.function([], gop)
            assert ndim == res[0].ndim
            np.testing.assert_equal(f(), res[0])
            np.testing.assert_equal(f(), res[1])

    def test_cloning_available(self):
        gop = generator(integers())
        res = gop ** 2
        shared = theano.shared(floatX(10))
        res1 = theano.clone(res, {gop: shared})
        f = theano.function([], res1)
        assert f() == np.float32(100)

    def test_default_value(self):
        def gen():
            for i in range(2):
                yield floatX(np.ones((10, 10)) * i)

        gop = generator(gen(), np.ones((10, 10)) * 10)
        f = theano.function([], gop)
        np.testing.assert_equal(np.ones((10, 10)) * 0, f())
        np.testing.assert_equal(np.ones((10, 10)) * 1, f())
        np.testing.assert_equal(np.ones((10, 10)) * 10, f())
        with pytest.raises(ValueError):
            gop.set_default(1)

    def test_set_gen_and_exc(self):
        def gen():
            for i in range(2):
                yield floatX(np.ones((10, 10)) * i)

        gop = generator(gen())
        f = theano.function([], gop)
        np.testing.assert_equal(np.ones((10, 10)) * 0, f())
        np.testing.assert_equal(np.ones((10, 10)) * 1, f())
        with pytest.raises(StopIteration):
            f()
        gop.set_gen(gen())
        np.testing.assert_equal(np.ones((10, 10)) * 0, f())
        np.testing.assert_equal(np.ones((10, 10)) * 1, f())

    def test_pickling(self, datagen):
        gen = generator(datagen)
        pickle.loads(pickle.dumps(gen))
        bad_gen = generator(integers())
        with pytest.raises(Exception):
            pickle.dumps(bad_gen)

    def test_gen_cloning_with_shape_change(self, datagen):
        gen = generator(datagen)
        gen_r = tt_rng().normal(size=gen.shape).T
        X = gen.dot(gen_r)
        res, _ = theano.scan(lambda x: x.sum(), X, n_steps=X.shape[0])
        assert res.eval().shape == (50,)
        shared = theano.shared(datagen.data.astype(gen.dtype))
        res2 = theano.clone(res, {gen: shared ** 2})
        assert res2.eval().shape == (1000,)


def gen1():
    i = 0
    while True:
        yield np.ones((10, 100)) * i
        i += 1


def gen2():
    i = 0
    while True:
        yield np.ones((20, 100)) * i
        i += 1


class TestScaling:
    """
    Related to minibatch training
    """

    def test_density_scaling(self):
        with pm.Model() as model1:
            Normal("n", observed=[[1]], total_size=1)
            p1 = theano.function([], model1.logpt)

        with pm.Model() as model2:
            Normal("n", observed=[[1]], total_size=2)
            p2 = theano.function([], model2.logpt)
        assert p1() * 2 == p2()

    def test_density_scaling_with_genarator(self):
        # We have different size generators

        def true_dens():
            g = gen1()
            for i, point in enumerate(g):
                yield stats.norm.logpdf(point).sum() * 10

        t = true_dens()
        # We have same size models
        with pm.Model() as model1:
            Normal("n", observed=gen1(), total_size=100)
            p1 = theano.function([], model1.logpt)

        with pm.Model() as model2:
            gen_var = generator(gen2())
            Normal("n", observed=gen_var, total_size=100)
            p2 = theano.function([], model2.logpt)

        for i in range(10):
            _1, _2, _t = p1(), p2(), next(t)
            decimals = select_by_precision(float64=7, float32=2)
            np.testing.assert_almost_equal(_1, _t, decimal=decimals)  # Value O(-50,000)
            np.testing.assert_almost_equal(_1, _2)
        # Done

    def test_gradient_with_scaling(self):
        with pm.Model() as model1:
            genvar = generator(gen1())
            m = Normal("m")
            Normal("n", observed=genvar, total_size=1000)
            grad1 = theano.function([m], tt.grad(model1.logpt, m))
        with pm.Model() as model2:
            m = Normal("m")
            shavar = theano.shared(np.ones((1000, 100)))
            Normal("n", observed=shavar)
            grad2 = theano.function([m], tt.grad(model2.logpt, m))

        for i in range(10):
            shavar.set_value(np.ones((100, 100)) * i)
            g1 = grad1(1)
            g2 = grad2(1)
            np.testing.assert_almost_equal(g1, g2)

    def test_multidim_scaling(self):
        with pm.Model() as model0:
            Normal("n", observed=[[1, 1], [1, 1]], total_size=[])
            p0 = theano.function([], model0.logpt)

        with pm.Model() as model1:
            Normal("n", observed=[[1, 1], [1, 1]], total_size=[2, 2])
            p1 = theano.function([], model1.logpt)

        with pm.Model() as model2:
            Normal("n", observed=[[1], [1]], total_size=[2, 2])
            p2 = theano.function([], model2.logpt)

        with pm.Model() as model3:
            Normal("n", observed=[[1, 1]], total_size=[2, 2])
            p3 = theano.function([], model3.logpt)

        with pm.Model() as model4:
            Normal("n", observed=[[1]], total_size=[2, 2])
            p4 = theano.function([], model4.logpt)

        with pm.Model() as model5:
            Normal("n", observed=[[1]], total_size=[2, Ellipsis, 2])
            p5 = theano.function([], model5.logpt)
        _p0 = p0()
        assert (
            np.allclose(_p0, p1())
            and np.allclose(_p0, p2())
            and np.allclose(_p0, p3())
            and np.allclose(_p0, p4())
            and np.allclose(_p0, p5())
        )

    def test_common_errors(self):
        with pm.Model():
            with pytest.raises(ValueError) as e:
                Normal("n", observed=[[1]], total_size=[2, Ellipsis, 2, 2])
            assert "Length of" in str(e.value)
            with pytest.raises(ValueError) as e:
                Normal("n", observed=[[1]], total_size=[2, 2, 2])
            assert "Length of" in str(e.value)
            with pytest.raises(TypeError) as e:
                Normal("n", observed=[[1]], total_size="foo")
            assert "Unrecognized" in str(e.value)
            with pytest.raises(TypeError) as e:
                Normal("n", observed=[[1]], total_size=["foo"])
            assert "Unrecognized" in str(e.value)
            with pytest.raises(ValueError) as e:
                Normal("n", observed=[[1]], total_size=[Ellipsis, Ellipsis])
            assert "Double Ellipsis" in str(e.value)

    def test_mixed1(self):
        with pm.Model():
            data = np.random.rand(10, 20, 30, 40, 50)
            mb = pm.Minibatch(data, [2, None, 20, Ellipsis, 10])
            Normal("n", observed=mb, total_size=(10, None, 30, Ellipsis, 50))

    def test_mixed2(self):
        with pm.Model():
            data = np.random.rand(10, 20, 30, 40, 50)
            mb = pm.Minibatch(data, [2, None, 20])
            Normal("n", observed=mb, total_size=(10, None, 30))

    def test_free_rv(self):
        with pm.Model() as model4:
            Normal("n", observed=[[1, 1], [1, 1]], total_size=[2, 2])
            p4 = theano.function([], model4.logpt)

        with pm.Model() as model5:
            Normal("n", total_size=[2, Ellipsis, 2], shape=(1, 1), broadcastable=(False, False))
            p5 = theano.function([model5.n], model5.logpt)
        assert p4() == p5(pm.floatX([[1]]))
        assert p4() == p5(pm.floatX([[1, 1], [1, 1]]))


@pytest.mark.usefixtures("strict_float32")
class TestMinibatch:
    data = np.random.rand(30, 10, 40, 10, 50)

    def test_1d(self):
        mb = pm.Minibatch(self.data, 20)
        assert mb.eval().shape == (20, 10, 40, 10, 50)

    def test_2d(self):
        mb = pm.Minibatch(self.data, [(10, 42), (4, 42)])
        assert mb.eval().shape == (10, 4, 40, 10, 50)

    def test_special1(self):
        mb = pm.Minibatch(self.data, [(10, 42), None, (4, 42)])
        assert mb.eval().shape == (10, 10, 4, 10, 50)

    def test_special2(self):
        mb = pm.Minibatch(self.data, [(10, 42), Ellipsis, (4, 42)])
        assert mb.eval().shape == (10, 10, 40, 10, 4)

    def test_special3(self):
        mb = pm.Minibatch(self.data, [(10, 42), None, Ellipsis, (4, 42)])
        assert mb.eval().shape == (10, 10, 40, 10, 4)

    def test_special4(self):
        mb = pm.Minibatch(self.data, [10, None, Ellipsis, (4, 42)])
        assert mb.eval().shape == (10, 10, 40, 10, 4)

    def test_cloning_available(self):
        gop = pm.Minibatch(np.arange(100), 1)
        res = gop ** 2
        shared = theano.shared(np.array([10]))
        res1 = theano.clone(res, {gop: shared})
        f = theano.function([], res1)
        assert f() == np.array([100])

    def test_align(self):
        m = pm.Minibatch(np.arange(1000), 1, random_seed=1)
        n = pm.Minibatch(np.arange(1000), 1, random_seed=1)
        f = theano.function([], [m, n])
        n.eval()  # not aligned
        a, b = zip(*(f() for _ in range(1000)))
        assert a != b
        pm.align_minibatches()
        a, b = zip(*(f() for _ in range(1000)))
        assert a == b
        n.eval()  # not aligned
        pm.align_minibatches([m])
        a, b = zip(*(f() for _ in range(1000)))
        assert a != b
        pm.align_minibatches([m, n])
        a, b = zip(*(f() for _ in range(1000)))
        assert a == b
