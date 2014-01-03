import numpy as np
import numpy.testing as npt
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest
import warnings

from pymc.backends import sqlite


class SQLiteTestCase(unittest.TestCase):

    def setUp(self):
        self.variables = ['x', 'y']
        self.model = mock.Mock()
        self.model.unobserved_RVs = self.variables
        self.model.fastfn = mock.MagicMock()

        with mock.patch('pymc.backends.base.modelcontext') as context:
            context.return_value = self.model
            self.db = sqlite.SQLite('test.db')
        self.db.cursor = mock.Mock()

        connect_patch = mock.patch('pymc.backends.sqlite.SQLite.connect')
        self.addCleanup(connect_patch.stop)
        self.connect = connect_patch.start()
        self.draws = 5


class TestSQLiteSample(SQLiteTestCase):

    def test_setup_trace(self):
        self.db.setup_samples(self.draws, chain=0)
        self.connect.assert_called_once_with()

    def test__create_trace_scalar(self):
        db = self.db
        var_trace = db._create_trace(chain=0, var_name='x',
                                     shape=(self.draws,))

        tbl_expected = ('CREATE TABLE IF NOT EXISTS [x] '
                        '(recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
                        'draw INTEGER, '
                        'chain INT(5), v1 FLOAT)')
        db.cursor.execute.assert_called_once_with(tbl_expected)

        trace_expected = ('INSERT INTO [x] (recid, draw, chain, v1) '
                          'VALUES (NULL, {draw}, 0, {value})')
        self.assertEqual(var_trace, trace_expected)

    def test__create_trace_1d(self):
        db = self.db
        var_trace = db._create_trace(chain=0, var_name='x',
                                     shape=(self.draws, 2))
        tbl_expected = ('CREATE TABLE IF NOT EXISTS [x] '
                        '(recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
                        'draw INTEGER, '
                        'chain INT(5), v1 FLOAT, v2 FLOAT)')
        db.cursor.execute.assert_called_once_with(tbl_expected)

        trace_expected = ('INSERT INTO [x] (recid, draw, chain, v1, v2) '
                          'VALUES (NULL, {draw}, 0, {value})')
        self.assertEqual(var_trace, trace_expected)

    def test__create_trace_2d(self):
        db = self.db
        var_trace = db._create_trace(chain=0, var_name='x',
                                     shape=(self.draws, 2, 3))
        tbl_expected = ('CREATE TABLE IF NOT EXISTS [x] '
                        '(recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
                        'draw INTEGER, '
                        'chain INT(5), '
                        'v1_1 FLOAT, v1_2 FLOAT, v1_3 FLOAT, '
                        'v2_1 FLOAT, v2_2 FLOAT, v2_3 FLOAT)')
        db.cursor.execute.assert_called_once_with(tbl_expected)

        trace_expected = ('INSERT INTO [x] (recid, draw, chain, '
                          'v1_1, v1_2, v1_3, '
                          'v2_1, v2_2, v2_3) '
                          'VALUES (NULL, {draw}, 0, {value})')
        self.assertEqual(var_trace, trace_expected)

    def test__store_value_scalar(self):
        db = self.db
        db.setup_samples(draws=3, chain=0)
        var_name = 'x'
        query = sqlite.QUERIES['insert'].format(table=var_name,
                                                value_cols='v1',
                                                chain=0)
        db.trace.samples[0] = {'x': query}
        db._store_value(draw=0, var_trace=db.trace.samples[0][var_name],
                        value=3.)
        expected = ('INSERT INTO [x] (recid, draw, chain, v1) '
                    'VALUES (NULL, 0, 0, 3.0)')
        db.cursor.execute.assert_called_once_with(expected)

    def test__store_value_1d(self):
        db = self.db
        db.setup_samples(draws=3, chain=0)
        var_name = 'x'
        query = sqlite.QUERIES['insert'].format(table=var_name,
                                                value_cols='v1, v2',
                                                chain=0)
        db.trace.samples[0] = {'x': query}
        print(db)
        db._store_value(draw=0, var_trace=db.trace.samples[0][var_name],
                        value=[3., 3.])
        expected = ('INSERT INTO [x] (recid, draw, chain, v1, v2) '
                    'VALUES (NULL, 0, 0, 3.0, 3.0)')
        db.cursor.execute.assert_called_once_with(expected)


class SQLiteSelectionTestCase(SQLiteTestCase):

    def setUp(self):
        super(SQLiteSelectionTestCase, self).setUp()
        self.db.var_shapes = {'x': (), 'y': (4,)}
        self.db.setup_samples(self.draws, chain=0)

        ndarray_patch = mock.patch('pymc.backends.sqlite._rows_to_ndarray')
        self.addCleanup(ndarray_patch.stop)
        ndarray_patch.start()

        self.draws = 5


class TestSQLiteSelection(SQLiteSelectionTestCase):

    def test_get_values_default_keywords(self):
        self.db.trace.get_values('x')
        expected = 'SELECT * FROM [x] WHERE (chain = 0)'
        self.db.cursor.execute.assert_called_with(expected)

    def test_get_values_burn_arg(self):
        self.db.trace.get_values('x', burn=2)
        expected = 'SELECT * FROM [x] WHERE (chain = 0) AND (draw > 1)'
        self.db.cursor.execute.assert_called_with(expected)

    def test_get_values_thin_arg(self):
        self.db.trace.get_values('x', thin=2)
        expected = ('SELECT * FROM [x] '
                    'WHERE (chain = 0) AND '
                    '(draw - (SELECT draw FROM [x] '
                    'WHERE chain = 0 '
                    'ORDER BY draw LIMIT 1)) % 2 = 0')
        self.db.cursor.execute.assert_called_with(expected)

    def test_get_values_burn_thin_arg(self):
        self.db.trace.get_values('x', thin=2, burn=1)
        expected = ('SELECT * FROM [x] '
                    'WHERE (chain = 0) AND (draw > 0) '
                    'AND (draw - (SELECT draw FROM [x] '
                    'WHERE (chain = 0) AND (draw > 0) '
                    'ORDER BY draw LIMIT 1)) % 2 = 0')
        self.db.cursor.execute.assert_called_with(expected)

    def test_point(self):
        idx = 2

        point = self.db.trace.point(idx)
        expected = {'x':
                    'SELECT * FROM [x] WHERE (chain=0) AND (draw=2)',
                    'y':
                    'SELECT * FROM [y] WHERE (chain=0) AND (draw=2)'}

        for var_name, value in expected.items():
            self.db.cursor.execute.assert_any_call(value)

    def test_slice(self):
        with warnings.catch_warnings(record=True) as wrn:
            self.db.trace[:10]
            self.assertEqual(len(wrn), 1)
            self.assertEqual(str(wrn[0].message),
                             'Slice for SQLite backend has no effect.')


class TestSQLiteSelectionMultipleChains(SQLiteSelectionTestCase):

    def setUp(self):
        super(TestSQLiteSelectionMultipleChains, self).setUp()
        self.db.trace.samples[1] = self.db.trace.samples[0]

    def test_get_values_default_keywords(self):
        self.db.trace.get_values('x')
        expected = ['SELECT * FROM [x] WHERE (chain = 0)',
                    'SELECT * FROM [x] WHERE (chain = 1)']
        for value in expected:
            self.db.cursor.execute.assert_any_call(value)

    def test_get_values_chains_one_given(self):
        self.db.trace.get_values('x', chains=[0])
        expected = 'SELECT * FROM [x] WHERE (chain = 0)'
        ## If 0 chain is last call, 1 was not called
        self.db.cursor.execute.assert_called_with(expected)


class TestSQLiteLoad(unittest.TestCase):

    def setUp(self):
        db_patch = mock.patch('pymc.backends.sqlite.SQLite')
        self.addCleanup(db_patch.stop)
        self.db = db_patch.start()

        table_list_patch = mock.patch('pymc.backends.sqlite._get_table_list')
        self.addCleanup(table_list_patch.stop)
        self.table_list = table_list_patch.start()
        self.table_list.return_value = ['x', 'y']

        var_strs_list_patch = mock.patch('pymc.backends.sqlite._get_var_strs')
        self.addCleanup(var_strs_list_patch.stop)
        self.var_strs_list = var_strs_list_patch.start()
        self.var_strs_list.return_value = ['v1', 'v2']

        chain_list_patch = mock.patch('pymc.backends.sqlite._get_chain_list')
        self.addCleanup(chain_list_patch.stop)
        self.chain_list = chain_list_patch.start()
        self.chain_list.return_value = [0, 1]

    def test_load(self):
        trace = sqlite.load('test.db')
        self.assertEqual(len(trace.samples), 2)

        self.assertTrue('x' in trace.samples[0])
        self.assertTrue('y' in trace.samples[0])

        expected = ('INSERT INTO [{}] '
                    '(recid, draw, chain, v1, v2) '
                    'VALUES (NULL, {{draw}}, {}, {{value}})')
        for chain in [0, 1]:
            for var_name in ['x', 'y']:
                self.assertEqual(trace.samples[chain][var_name],
                                 expected.format(var_name, chain))


def test_create_column_empty():
    result = sqlite.create_colnames(())
    expected = ['v1']
    assert result == expected


def test_create_column_1d():
    result = sqlite.create_colnames((2, ))
    expected = ['v1', 'v2']
    assert result == expected


def test_create_column_2d():
    result = sqlite.create_colnames((2, 2))
    expected = ['v1_1', 'v1_2', 'v2_1', 'v2_2']
    assert result == expected
