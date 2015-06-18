import numpy.testing as npt
import os
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, sqlite
import tempfile

DBNAME = os.path.join(tempfile.gettempdir(), 'test.db')


class TestSQlite0dSampling(bf.SamplingTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = ()


class TestSQlite1dSampling(bf.SamplingTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = 2


class TestSQlite2dSampling(bf.SamplingTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = (2, 3)


class TestSQLite0dSelection(bf.SelectionTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = ()


class TestSQLite1dSelection(bf.SelectionTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = 2


class TestSQLite2dSelection(bf.SelectionTestCase):
    backend = sqlite.SQLite
    name = DBNAME
    shape = (2, 3)


class TestSQLiteDumpLoad(bf.DumpLoadTestCase):
    backend = sqlite.SQLite
    load_func = staticmethod(sqlite.load)
    name = DBNAME
    shape = (2, 3)


class TestNDArraySqliteEquality(bf.BackendEqualityTestCase):
    backend0 = ndarray.NDArray
    name0 = None
    backend1 = sqlite.SQLite
    name1 = DBNAME
    shape = (2, 3)


def test_create_column_0d():
    shape = ()
    result = sqlite._create_colnames(shape)
    expected = ['v1']
    assert result == expected
    assert sqlite._create_shape(result) == shape


def test_create_column_1d():
    shape = 2,
    result = sqlite._create_colnames(shape)
    expected = ['v1', 'v2']
    assert result == expected
    assert sqlite._create_shape(result) == shape


def test_create_column_2d():
    shape = 2, 3
    result = sqlite._create_colnames(shape)
    expected = ['v1_1', 'v1_2', 'v1_3',
                'v2_1', 'v2_2', 'v2_3']
    assert result == expected
    assert sqlite._create_shape(result) == shape


def test_create_column_3d():
    shape = 2, 3, 4
    assert sqlite._create_shape(sqlite._create_colnames(shape)) == shape
