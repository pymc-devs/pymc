import numpy as np
import numpy.testing as npt
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest
import warnings

from pymc.backends import base, sqlite


class SQLiteTestCase(unittest.TestCase):

    def setUp(self):
        self.variables = ['x', 'y']
        self.model = mock.Mock()
        self.model.unobserved_RVs = self.variables
        self.model.fastfn = mock.MagicMock()

        db_patch = mock.patch('pymc.backends.sqlite._SQLiteDB')
        self.addCleanup(db_patch.stop)
        self.db = db_patch.start()

        with mock.patch('pymc.backends.base.modelcontext') as context:
            context.return_value = self.model
            self.trace = sqlite.SQLite('test.db')

        self.draws = 5

        self.trace.var_shapes = {'x': (), 'y': (3,)}

        self.trace._chains = [0]
        self.trace._len = self.draws


class TestSQLiteSample(SQLiteTestCase):

    def test_setup_trace(self):
        self.trace.setup(self.draws, chain=0)
        assert self.trace.db.connect.called

    def test_setup_scalar(self):
        trace = self.trace
        trace.setup(draws=3, chain=0)
        tbl_expected = ('CREATE TABLE IF NOT EXISTS [x] '
                        '(recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
                        'draw INTEGER, '
                        'chain INT(5), v1 FLOAT)')
        trace.db.cursor.execute.assert_any_call(tbl_expected)

        trace_expected = ('INSERT INTO [x] (recid, draw, chain, v1) '
                          'VALUES (NULL, ?, ?, ?)')
        self.assertEqual(trace.var_inserts['x'], trace_expected)

    def test_setup_1d(self):
        trace = self.trace
        trace.setup(draws=3, chain=0)
        trace._chains = []

        tbl_expected = ('CREATE TABLE IF NOT EXISTS [y] '
                        '(recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
                        'draw INTEGER, '
                        'chain INT(5), v1 FLOAT, v2 FLOAT, v3 FLOAT)')
        trace.db.cursor.execute.assert_any_call(tbl_expected)

        trace_expected = ('INSERT INTO [y] (recid, draw, chain, v1, v2, v3) '
                          'VALUES (NULL, ?, ?, ?, ?, ?)')
        self.assertEqual(trace.var_inserts['y'], trace_expected)

    def test_setup_2d(self):
        trace = self.trace
        trace.var_shapes = {'x': (2, 3)}
        trace.setup(draws=3, chain=0)
        tbl_expected = ('CREATE TABLE IF NOT EXISTS [x] '
                        '(recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
                        'draw INTEGER, '
                        'chain INT(5), '
                        'v1_1 FLOAT, v1_2 FLOAT, v1_3 FLOAT, '
                        'v2_1 FLOAT, v2_2 FLOAT, v2_3 FLOAT)')

        trace.db.cursor.execute.assert_any_call(tbl_expected)
        trace_expected = ('INSERT INTO [x] (recid, draw, chain, '
                          'v1_1, v1_2, v1_3, '
                          'v2_1, v2_2, v2_3) '
                          'VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)')
        self.assertEqual(trace.var_inserts['x'], trace_expected)

    def test_record_scalar(self):
        trace = self.trace
        trace.setup(draws=3, chain=0)
        varname = 'x'
        trace.varnames = ['x']

        trace.draw_idx = 0
        trace.fn = mock.Mock(return_value=iter([3.]))
        trace.record({'x': None})
        expected = (0, 0, 3.)
        self.assertTrue(expected in self.trace._queue['x'])

    def test_record_1d(self):
        trace = self.trace
        trace.setup(draws=3, chain=0)
        varname = 'x'
        trace.varnames = ['x']

        trace.draw_idx = 0
        trace.fn = mock.Mock(return_value=iter([[3., 3.]]))
        trace.record({'x': None})
        expected = (0, 0, 3., 3.)
        self.assertTrue(expected in self.trace._queue['x'])


class TestSQLiteSelection(SQLiteTestCase):

    def setUp(self):
        super(TestSQLiteSelection, self).setUp()
        self.trace.var_shapes = {'x': (), 'y': (4,)}
        self.trace.setup(self.draws, chain=0)

        ndarray_patch = mock.patch('pymc.backends.sqlite._rows_to_ndarray')
        self.addCleanup(ndarray_patch.stop)
        ndarray_patch.start()

        self.draws = 5

    def test_get_values_default_keywords(self):
        self.trace.get_values('x')
        statement = sqlite.TEMPLATES['select'].format(table='x')
        expected = (statement, {'chain': 0})
        self.trace.db.cursor.execute.assert_called_with(*expected)

    def test_get_values_burn_arg(self):
        self.trace.get_values('x', burn=2).format(table='x')
        statement = sqlite.TEMPLATES['select_burn'].format(table='x')
        expected = (statement, {'chain': 0, 'burn': 1})
        self.trace.db.cursor.execute.assert_called_with(*expected)

    def test_get_values_thin_arg(self):
        self.trace.get_values('x', thin=2)
        statement = sqlite.TEMPLATES['select_thin'].format(table='x')
        expected = (statement, {'chain': 0, 'thin': 2})
        self.trace.db.cursor.execute.assert_called_with(*expected)

    def test_get_values_burn_thin_arg(self):
        self.trace.get_values('x', thin=2, burn=1)
        statement = sqlite.TEMPLATES['select_burn_thin'].format(table='x')
        expected = (statement, {'chain': 0, 'burn': 0, 'thin': 2})
        self.trace.db.cursor.execute.assert_called_with(*expected)

    def test_point(self):
        idx = 2

        point = self.trace.point(idx)
        statement = sqlite.TEMPLATES['select_point'].format(table='x')
        statement_args = {'chain': 0, 'draw': idx}
        expected = {'x': (statement, statement_args),
                    'y': (statement, statement_args)}

        for varname, value in expected.items():
            self.trace.db.cursor.execute.assert_any_call(*value)

    def test_slice(self):
        with warnings.catch_warnings(record=True) as wrn:
            self.trace[:10]
            self.assertEqual(len(wrn), 1)
            self.assertEqual(str(wrn[0].message),
                             'Slice for SQLite backend has no effect.')


class TestSQLiteSelectionMultipleChains(SQLiteTestCase):

    def setUp(self):
        self.variables = ['x', 'y']
        self.model = mock.Mock()
        self.model.unobserved_RVs = self.variables
        self.model.fastfn = mock.MagicMock()

        db_patch = mock.patch('pymc.backends.sqlite._SQLiteDB')
        self.addCleanup(db_patch.stop)
        self.db = db_patch.start()

        with mock.patch('pymc.backends.base.modelcontext') as context:
            context.return_value = self.model
            self.trace0 = sqlite.SQLite('test.db')
            self.trace1 = sqlite.SQLite('test.db')

        self.draws = 5

        self.trace0.var_shapes = {'x': (), 'y': (3,)}
        self.trace0.chain = 0
        self.trace0._len = self.draws

        self.trace1.var_shapes = {'x': (), 'y': (3,)}
        self.trace1.chain = 1
        self.trace1._len = self.draws

        self.mtrace = base.MultiTrace([self.trace0, self.trace1])

        ndarray_patch = mock.patch('pymc.backends.sqlite._rows_to_ndarray')
        self.addCleanup(ndarray_patch.stop)
        ndarray_patch.start()

        self.draws = 5

    def test_get_values_default_keywords(self):
        self.mtrace.get_values('x')

        db = self.mtrace._traces[0].db
        self.assertEqual(db.cursor.execute.call_count, 2)

        statement = 'SELECT * FROM [x] WHERE (chain = :chain)'
        expected = [mock.call(statement, {'chain': chain})
                    for chain in (0, 1)]
        db.cursor.execute.assert_has_calls(expected)

    def test_get_values_chains_one_given(self):
        self.mtrace.get_values('x', chains=[0])
        ## If 0 chain is last call, 1 was not called.
        statement = sqlite.TEMPLATES['select'].format(table='x')
        expected = (statement, {'chain': 0})
        trace = self.mtrace._traces[0]
        trace.db.cursor.execute.assert_called_with(*expected)

    def test_get_values_chains_one_chain_arg(self):
        self.mtrace.get_values('x', chains=[0])
        ## If 0 chain is last call, 1 was not called.
        statement = sqlite.TEMPLATES['select'].format(table='x')
        expected = (statement, {'chain': 0})
        trace = self.mtrace._traces[0]
        trace.db.cursor.execute.assert_called_with(*expected)


class TestSQLiteLoad(unittest.TestCase):

    def setUp(self):
        db_patch = mock.patch('pymc.backends.sqlite._SQLiteDB')
        self.addCleanup(db_patch.stop)
        self.db = db_patch.start()

        table_list_patch = mock.patch('pymc.backends.sqlite._get_table_list')
        self.addCleanup(table_list_patch.stop)
        self.table_list = table_list_patch.start()
        self.table_list.return_value = ['x', 'y']

    def test_load(self):
        trace = sqlite.load('test.db')
        assert self.table_list.called
        assert self.db.called


def test_create_column_empty():
    result = sqlite._create_colnames(())
    expected = ['v1']
    assert result == expected


def test_create_column_1d():
    result = sqlite._create_colnames((2, ))
    expected = ['v1', 'v2']
    assert result == expected


def test_create_column_2d():
    result = sqlite._create_colnames((2, 2))
    expected = ['v1_1', 'v1_2', 'v2_1', 'v2_2']
    assert result == expected
