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
