import numpy.testing as npt
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, text


class TestTextDumpLoad(bf.DumpLoadTestCase):
    backend = text.Text
    load_func = staticmethod(text.load)
    name = 'text-db'
    shape = (2, 3)


class TestTextDumpFunction(bf.BackendEqualityTestCase):
    backend0 = backend1 = ndarray.NDArray
    name0 = None
    name1 = 'text-db'
    shape = (2, 3)

    @classmethod
    def setUpClass(cls):
        super(TestTextDumpFunction, cls).setUpClass()
        text.dump(cls.name1, cls.mtrace1)
        with cls.model:
            cls.mtrace1 = text.load(cls.name1)


def test__chain_dir_to_chain():
    assert text._chain_dir_to_chain('/path/to/chain-0') == 0
    assert text._chain_dir_to_chain('chain-0') == 0
