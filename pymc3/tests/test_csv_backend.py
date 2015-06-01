import numpy.testing as npt
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, csv


class TestCSV0dSampling(bf.SamplingTestCase):
    backend = csv.CSV
    name = 'csv-db'
    shape = ()


class TestCSV1dSampling(bf.SamplingTestCase):
    backend = csv.CSV
    name = 'csv-db'
    shape = 2


class TestCSV2dSampling(bf.SamplingTestCase):
    backend = csv.CSV
    name = 'csv-db'
    shape = (2, 3)


class TestCSV0dSelection(bf.SelectionNoSliceTestCase):
    backend = csv.CSV
    name = 'csv-db'
    shape = ()


class TestCSV1dSelection(bf.SelectionNoSliceTestCase):
    backend = csv.CSV
    name = 'csv-db'
    shape = 2


class TestCSV2dSelection(bf.SelectionNoSliceTestCase):
    backend = csv.CSV
    name = 'csv-db'
    shape = (2, 3)


class TestCSVDumpLoad(bf.DumpLoadTestCase):
    backend = csv.CSV
    load_func = staticmethod(csv.load)
    name = 'csv-db'
    shape = (2, 3)


class TestCSVDumpFunction(bf.BackendEqualityTestCase):
    backend0 = backend1 = ndarray.NDArray
    name0 = None
    name1 = 'csv-db'
    shape = (2, 3)

    @classmethod
    def setUpClass(cls):
        super(TestCSVDumpFunction, cls).setUpClass()
        csv.dump(cls.name1, cls.mtrace1)
        with cls.model:
            cls.mtrace1 = csv.load(cls.name1)


class TestTraceToDf(bf.ModelBackendSampledTestCase):
    backend = ndarray.NDArray
    name = 'csv-db'
    shape = (2, 3)

    def test_trace_to_df(self):
        mtrace = self.mtrace
        df = csv._trace_to_df(mtrace._traces[0])
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

class TestNDArrayCSVEquality(bf.BackendEqualityTestCase):
    backend0 = ndarray.NDArray
    name0 = None
    backend1 = csv.CSV
    name1 = 'csv-db'
    shape = (2, 3)


def test_create_flat_names_0d():
    shape = ()
    result = csv._create_flat_names('x', shape)
    expected = ['x']
    assert result == expected
    assert csv._create_shape(result) == shape


def test_create_flat_names_1d():
    shape = 2,
    result = csv._create_flat_names('x', shape)
    expected = ['x__0', 'x__1']
    assert result == expected
    assert csv._create_shape(result) == shape


def test_create_flat_names_2d():
    shape = 2, 3
    result = csv._create_flat_names('x', shape)
    expected = ['x__0_0', 'x__0_1', 'x__0_2',
                'x__1_0', 'x__1_1', 'x__1_2']
    assert result == expected
    assert csv._create_shape(result) == shape


def test_create_flat_names_3d():
    shape = 2, 3, 4
    assert csv._create_shape(csv._create_flat_names('x', shape)) == shape
