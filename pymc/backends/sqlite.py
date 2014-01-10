"""SQLite trace backend

Store sampling values in SQLite database file.

Database format
---------------
For each variable, a table is created with the following format:

 recid (INT), draw (INT), chain (INT),  v1 (FLOAT), v2 (FLOAT), v3 (FLOAT) ...

The variable column names are extended to reflect addition dimensions.
For example, a variable with the shape (2, 2) would be stored as

 key (INT), draw (INT), chain (INT),  v1_1 (FLOAT), v1_2 (FLOAT), v2_1 (FLOAT) ...

The key is autoincremented each time a new row is added to the table.
The chain column denotes the chain index, and starts at 0.
"""
import numpy as np
import sqlite3
import warnings

from pymc.backends import base

QUERIES = {
    'table':            ('CREATE TABLE IF NOT EXISTS [{table}] '
                         '(recid INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
                         'draw INTEGER, chain INT(5), '
                         '{value_cols})'),
    'insert':           ('INSERT INTO [{table}] '
                         '(recid, draw, chain, {value_cols}) '
                         'VALUES (NULL, {{draw}}, {chain}, {{value}})'),
    'select':           ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain})'),
    'select_burn':      ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain}) AND (draw > {burn})'),
    'select_thin':      ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain}) AND '
                         '(draw - (SELECT draw FROM [{table}] '
                         'WHERE chain = {chain} '
                         'ORDER BY draw LIMIT 1)) % {thin} = 0'),
    'select_burn_thin': ('SELECT * FROM [{table}] '
                         'WHERE (chain = {chain}) AND (draw > {burn}) '
                         'AND (draw - (SELECT draw FROM [{table}] '
                         'WHERE (chain = {chain}) AND (draw > {burn}) '
                         'ORDER BY draw LIMIT 1)) % {thin} = 0'),
    'select_point':     ('SELECT * FROM [{{table}}] '
                         'WHERE (chain={chain}) AND (draw={draw})'),
    'max_draw':         ('SELECT MAX(draw) FROM [{table}] '
                         'WHERE chain={chain}'),
}


class SQLite(base.Backend):
    """SQLite storage

    Parameters
    ----------
    name : str
        Name of database file
    model : Model
        If None, the model is taken from the `with` context.
    variables : list of variable objects
        Sampling values will be stored for these variables
    """

    def __init__(self, name, model=None, variables=None):
        super(SQLite, self).__init__(name, model, variables)

        self.trace = Trace(self.var_names, self)

        ## These are set in `setup` to avoid sqlite3.OperationalError
        ## (Base Cursor.__init__ not called) when performing parallel
        ## sampling
        self.conn = None
        self.cursor = None
        self.connected = False

        self.var_inserts = {}  # var_name -> insert query
        self.draw_idx = 0

    def setup(self, draws, chain):
        """Perform chain-specific setup

        draws : int
            Expected number of draws
        chain : int
            chain number
        """
        self.connect()
        table = QUERIES['table']
        insert = QUERIES['insert']
        for var_name, shape in self.var_shapes.items():
            var_cols = _create_colnames(shape)
            var_float = ', '.join([v + ' FLOAT' for v in var_cols])
            ## Create table
            self.cursor.execute(table.format(table=var_name,
                                             value_cols=var_float))
            ## Create insert query for each variable
            var_str = ', '.join(var_cols)
            self.var_inserts[var_name] = insert.format(table=var_name,
                                                       value_cols=var_str,
                                                       chain=chain)

    def connect(self):
        if self.connected:
            return

        self.conn = sqlite3.connect(self.name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.connected = True

    def record(self, point):
        """Record results of a sampling iteration

        point : dict
            Values mappled to variable names
        """
        for var_name, value in zip(self.var_names, self.fn(point)):
            val_str = ', '.join(['{}'.format(val) for val in np.ravel(value)])
            query = self.var_inserts[var_name].format(draw=self.draw_idx,
                                                      value=val_str)
            self.cursor.execute(query)
        self.draw_idx += 1

        if not self.draw_idx % 1000:
            self.conn.commit()

    def close(self):
        if not self.connected:
            return

        self.cursor.close()
        self.conn.commit()
        self.conn.close()
        self.connected = False


class Trace(base.Trace):

    __doc__ = 'SQLite trace\n' + base.Trace.__doc__

    def __init__(self, var_names, backend=None):
        super(Trace, self).__init__(var_names, backend)
        self._len = None
        self._chains = None

    def __len__(self):
        if self._len is None:
            query = QUERIES['max_draw'].format(table=self.var_names[0],
                                               chain=self.default_chain)
            self.backend.connect()
            self._len = self.backend.cursor.execute(query).fetchall()[0][0] + 1
        return self._len

    @property
    def chains(self):
        """All chains in trace"""
        if self._chains is None:
            self.backend.connect()
            var_name = self.var_names[0]  # any variable should do
            self._chains = _get_chain_list(self.backend.cursor, var_name)
        return self._chains

    def get_values(self, var_name, burn=0, thin=1, combine=False, chains=None,
                   squeeze=True):
        """Get values from samples

        Parameters
        ----------
        var_name : str
        burn : int
        thin : int
        combine : bool
            If True, results from all chains will be concatenated.
        chains : list
            Chains to retrieve. If None, `active_chains` is used.
        squeeze : bool
            If `combine` is False, return a single array element if the
            resulting list of values only has one element (even if
            `combine` is True).

        Returns
        -------
        A list of NumPy array of values
        """
        if burn < 0:
            raise ValueError('Negative burn values not supported '
                             'in SQLite backend.')
        if thin < 1:
            raise ValueError('Only positive thin values are supported '
                             'in SQLite backend.')
        if chains is None:
            chains = self.active_chains

        var_name = str(var_name)

        query_args = {}
        if burn == 0 and thin == 1:
            query = 'select'
        elif thin == 1:
            query = 'select_burn'
            query_args = {'burn': burn - 1}
        elif burn == 0:
            query = 'select_thin'
            query_args = {'thin': thin}
        else:
            query = 'select_burn_thin'
            query_args = {'burn': burn - 1, 'thin': thin}

        self.backend.connect()
        results = []
        for chain in chains:
            call = QUERIES[query].format(table=var_name, chain=chain,
                                         **query_args)
            self.backend.cursor.execute(call)
            results.append(_rows_to_ndarray(self.backend.cursor))
        return base._squeeze_cat(results, combine, squeeze)

    def _slice(self, idx):
        warnings.warn('Slice for SQLite backend has no effect.')

    def point(self, idx, chain=None):
        """Return dictionary of point values at `idx` for current chain
        with variables names as keys.

        If `chain` is not specified, `default_chain` is used.
        """
        if idx < 0:
            raise ValueError('Negtive indexing is not supported '
                             'in SQLite backend.')
        if chain is None:
            chain = self.default_chain

        query = QUERIES['select_point'].format(chain=chain,
                                               draw=idx)
        self.backend.connect()
        var_values = {}
        for var_name in self.var_names:
            self.backend.cursor.execute(query.format(table=var_name))
            var_values[var_name] = np.squeeze(
                _rows_to_ndarray(self.backend.cursor))
        return var_values

    def merge_chains(self, traces):
        pass


def _create_colnames(shape):
    """Return column names based on `shape`

    Examples
    --------
    >>> create_colnames((5,))
    ['v1', 'v2', 'v3', 'v4', 'v5']

    >>> create_colnames((2,2))
    ['v1_1', 'v1_2', 'v2_1', 'v2_2']
    """
    if not shape:
        return ['v1']

    size = np.prod(shape)
    indices = (np.indices(shape) + 1).reshape(-1, size)
    return ['v' + '_'.join(map(str, i)) for i in zip(*indices)]


def load(name, model=None):
    """Load SQLite database from file name

    Parameters
    ----------
    name : str
        Path to SQLite database file
    model : Model
        If None, the model is taken from the `with` context.

    Returns
    -------
    SQLite backend instance
    """
    db = SQLite(name, model=model)
    db.connect()
    var_names = _get_table_list(db.cursor)
    return Trace(var_names, db)


def _get_table_list(cursor):
    """Return a list of table names in the current database"""
    ## Modified from Django. Skips the sqlite_sequence system table used
    ## for autoincrement key generation.
    cursor.execute("SELECT name FROM sqlite_master "
                   "WHERE type='table' AND NOT name='sqlite_sequence' "
                   "ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


def _get_var_strs(cursor, var_name):
    cursor.execute('SELECT * FROM [{}]'.format(var_name))
    col_names = (col_descr[0] for col_descr in cursor.description)
    return [name for name in col_names if name.startswith('v')]


def _get_chain_list(cursor, var_name):
    """Return a list of sorted chains for `var_name`"""
    cursor.execute('SELECT DISTINCT chain FROM [{}]'.format(var_name))
    chains = [chain[0] for chain in cursor.fetchall()]
    chains.sort()
    return chains


def _rows_to_ndarray(cursor):
    """Convert SQL row to NDArray"""
    return np.array([row[3:] for row in cursor.fetchall()])
