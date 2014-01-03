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

    def __init__(self, name, model=None, variables=None):
        super(SQLite, self).__init__(name, model, variables)
        ## initialized by _connect
        self.conn = None
        self.cursor = None
        self.connected = False

    def _initialize_trace(self):
        return Trace(self.var_names, self)

    def connect(self):
        if self.connected:
            return

        self.conn = sqlite3.connect(self.name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.connected = True

    ## sampling methods

    def setup_samples(self, draws, chain):
        ## make connection here (versus __init__) to handle parallel
        ## chains
        self.connect()
        super(SQLite, self).setup_samples(draws, chain)

    def _create_trace(self, chain, var_name, shape):
        ## first element of trace is number of draws
        var_cols = create_colnames(shape[1:])
        var_float = ', '.join([v + ' FLOAT' for v in var_cols])
        self.cursor.execute(QUERIES['table'].format(table=var_name,
                                                    value_cols=var_float))
        return QUERIES['insert'].format(table=var_name,
                                        value_cols=', '.join(var_cols),
                                        chain=chain)

    def _store_value(self, draw, var_trace, value):
        val_str = ', '.join(['{}'.format(val) for val in np.ravel(value)])
        query = var_trace.format(draw=draw, value=val_str)
        self.cursor.execute(query)

    def commit(self):
        self.conn.commit()

    def close(self):
        if not self.connected:
            return

        self.cursor.close()
        self.commit()
        self.conn.close()
        self.connected = False


class Trace(base.Trace):

    __doc__ = 'SQLite trace\n' + base.Trace.__doc__

    def __len__(self):
        try:
            return super(Trace, self).__len__()
        except KeyError:  # draws dictionary not set up
            query = QUERIES['max_draw'].format(table=self.var_names[0],
                                               chain=self.default_chain)
            self.backend.connect()
            draws = self.backend.cursor.execute(query).fetchall()[0][0] + 1
            self._draws[self.default_chain] = draws
            return draws

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
        """Slice trace object
        """
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


def create_colnames(shape):
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
        If None, the model is taken from the `with` context. The trace
        can be loaded without connecting by passing False (although
        connecting to the original model is recommended).

    Returns
    -------
    SQLite backend instance
    """
    db = SQLite(name, model=model)
    db.connect()

    var_names = _get_table_list(db.cursor)
    trace = Trace(var_names, db)
    var_cols = {var_name: ', '.join(_get_var_strs(db.cursor, var_name))
                for var_name in var_names}

    ## Use first var_names element to get chain list. Chains should be
    ## the same for all.
    chains = _get_chain_list(db.cursor, var_names[0])

    query = QUERIES['insert']
    for chain in chains:
        samples = {}
        for var_name in var_names:
            samples[var_name] = query.format(table=var_name,
                                             value_cols=var_cols[var_name],
                                             chain=chain)
        trace.samples[chain] = samples
    return trace


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
