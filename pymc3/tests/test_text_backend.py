from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, text
import pytest
import theano


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
        super(TestTextDumpFunction, cls).setup_class()
        text.dump(cls.name1, cls.mtrace1)
        with cls.model:
            cls.mtrace1 = text.load(cls.name1)


class TestNDArrayTextEquality(bf.BackendEqualityTestCase):
    backend0 = ndarray.NDArray
    name0 = None
    backend1 = text.Text
    name1 = 'text-db'
    shape = (2, 3)
