import numpy.testing as npt
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, text


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


class TestText0dSelection(bf.SelectionNoSliceTestCase):
    backend = text.Text
    name = 'text-db'
    shape = ()


class TestText1dSelection(bf.SelectionNoSliceTestCase):
    backend = text.Text
    name = 'text-db'
    shape = 2


class TestText2dSelection(bf.SelectionNoSliceTestCase):
    backend = text.Text
    name = 'text-db'
    shape = (2, 3)


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


class TestTraceToDf(bf.ModelBackendSampledTestCase):
    backend = ndarray.NDArray
    name = 'text-db'
    shape = (2, 3)

    def test_trace_to_df(self):
        mtrace = self.mtrace
        df = text._trace_to_df(mtrace._traces[0])
        self.assertEqual(len(mtrace), df.shape[0])

        checked = False
        for varname in self.test_point.keys():
            vararr = mtrace.get_values(varname, chains=0)
            ## With `shape` above, only one variable has to have that
            ## `shape`.
            if vararr.shape[1:] != self.shape:
                continue
            npt.assert_equal(vararr[:, 0, 0], df[varname + '__0_0'].values)
            npt.assert_equal(vararr[:, 1, 0], df[varname + '__1_0'].values)
            npt.assert_equal(vararr[:, 1, 2], df[varname + '__1_2'].values)
            checked = True
        self.assertTrue(checked)

class TestNDArrayTextEquality(bf.BackendEqualityTestCase):
    backend0 = ndarray.NDArray
    name0 = None
    backend1 = text.Text
    name1 = 'text-db'
    shape = (2, 3)


def test_create_flat_names_0d():
    shape = ()
    result = text._create_flat_names('x', shape)
    expected = ['x']
    assert result == expected
    assert text._create_shape(result) == shape


def test_create_flat_names_1d():
    shape = 2,
    result = text._create_flat_names('x', shape)
    expected = ['x__0', 'x__1']
    assert result == expected
    assert text._create_shape(result) == shape


def test_create_flat_names_2d():
    shape = 2, 3
    result = text._create_flat_names('x', shape)
    expected = ['x__0_0', 'x__0_1', 'x__0_2',
                'x__1_0', 'x__1_1', 'x__1_2']
    assert result == expected
    assert text._create_shape(result) == shape


def test_create_flat_names_3d():
    shape = 2, 3, 4
    assert text._create_shape(text._create_flat_names('x', shape)) == shape
