import pymc3 as pm
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, text
import pytest
import theano


class TestTextSampling:
    name = 'text-db'

    def test_supports_sampler_stats(self):
        with pm.Model():
            pm.Normal("mu", mu=0, sigma=1, shape=2)
            db = text.Text(self.name)
            pm.sample(20, tune=10, init=None, trace=db, cores=2)

    def teardown_method(self):
        bf.remove_file_or_directory(self.name)


class TestText0dSampling(bf.SamplingTestCase):
    backend = text.Text
    name = 'text-db'
    shape = ()


class TestText1dSampling(bf.SamplingTestCase):
    backend = text.Text
    name = 'text-db'
    shape = 2


class TestText2dSampling(bf.SamplingTestCase):
    backend = text.Text
    name = 'text-db'
    shape = (2, 3)


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestText0dSelection(bf.SelectionTestCase):
    backend = text.Text
    name = 'text-db'
    shape = ()


class TestText1dSelection(bf.SelectionTestCase):
    backend = text.Text
    name = 'text-db'
    shape = 2


class TestText2dSelection(bf.SelectionTestCase):
    backend = text.Text
    name = 'text-db'
    shape = (2, 3)


class TestTextDumpLoad(bf.DumpLoadTestCase):
    backend = text.Text
    load_func = staticmethod(text.load)
    name = 'text-db'
    shape = (2, 3)


@pytest.mark.xfail(condition=(theano.config.floatX == "float32"), reason="Fails on float32")
class TestTextDumpFunction(bf.BackendEqualityTestCase):
    backend0 = backend1 = ndarray.NDArray
    name0 = None
    name1 = 'text-db'
    shape = (2, 3)

    @classmethod
    def setup_class(cls):
        super().setup_class()
        text.dump(cls.name1, cls.mtrace1)
        with cls.model:
            cls.mtrace1 = text.load(cls.name1)


class TestNDArrayTextEquality(bf.BackendEqualityTestCase):
    backend0 = ndarray.NDArray
    name0 = None
    backend1 = text.Text
    name1 = 'text-db'
    shape = (2, 3)
