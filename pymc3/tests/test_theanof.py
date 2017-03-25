import pickle
import itertools
import numpy as np
from theano import theano
from pymc3.theanof import GeneratorOp, generator, tt_rng
from pymc3.data import DataSampler, GeneratorAdapter
import pytest


def integers():
    i = 0
    while True:
        yield np.float32(i)
        i += 1


def integers_ndim(ndim):
    i = 0
    while True:
        yield np.ones((2,) * ndim) * i
        i += 1


class TestGenerator(object):
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
        shared = theano.shared(np.float32(10))
        res1 = theano.clone(res, {gop: shared})
        f = theano.function([], res1)
        assert f() == np.float32(100)

    def test_default_value(self):
        def gen():
            for i in range(2):
                yield np.ones((10, 10)) * i

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
                yield np.ones((10, 10)) * i

        gop = generator(gen())
        f = theano.function([], gop)
        np.testing.assert_equal(np.ones((10, 10)) * 0, f())
        np.testing.assert_equal(np.ones((10, 10)) * 1, f())
        with pytest.raises(StopIteration):
            f()
        gop.set_gen(gen())
        np.testing.assert_equal(np.ones((10, 10)) * 0, f())
        np.testing.assert_equal(np.ones((10, 10)) * 1, f())

    def test_pickling(self):
        data = np.random.uniform(size=(1000, 10))
        minibatches = DataSampler(data, batchsize=50)
        gen = generator(minibatches)
        pickle.loads(pickle.dumps(gen))
        bad_gen = generator(integers())
        with pytest.raises(Exception):
            pickle.dumps(bad_gen)

    def test_gen_cloning_with_shape_change(self):
        data = np.random.uniform(size=(1000, 10))
        minibatches = DataSampler(data, batchsize=50)
        gen = generator(minibatches)
        gen_r = tt_rng().normal(size=gen.shape).T
        X = gen.dot(gen_r)
        res, _ = theano.scan(lambda x: x.sum(), X, n_steps=X.shape[0])
        assert res.eval().shape == (50,)
        shared = theano.shared(data)
        res2 = theano.clone(res, {gen: shared**2})
        assert res2.eval().shape == (1000,)
