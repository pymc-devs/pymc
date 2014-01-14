import sys
import numpy as np
import numpy.testing as npt
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest
if sys.version_info[0] == 2:
    from StringIO import StringIO
else:
    from io import StringIO
import json

from pymc.backends import text


class TextTestCase(unittest.TestCase):

    def setUp(self):
        self.variables = ['x', 'y']
        self.model = mock.Mock()
        self.model.unobserved_RVs = self.variables
        self.model.fastfn = mock.MagicMock()

        shape_fh_patch = mock.patch('pymc.backends.text._get_shape_fh')
        self.addCleanup(shape_fh_patch.stop)
        self.shape_fh = shape_fh_patch.start()

        mkdir_patch = mock.patch('pymc.backends.text.os.mkdir')
        self.addCleanup(mkdir_patch.stop)
        mkdir_patch.start()


class TestTextWrite(TextTestCase):

    def setUp(self):
        super(TestTextWrite, self).setUp()

        with mock.patch('pymc.backends.base.modelcontext') as context:
            context.return_value = self.model
            self.db = text.Text('textdb')

        self.draws = 5
        self.db.var_shapes = {'x': (), 'y': (4,)}
        self.db.setup(self.draws, chain=0)
        self.db.draw_idx = self.draws

        savetxt_patch = mock.patch('pymc.backends.text.np.savetxt')
        self.addCleanup(savetxt_patch.stop)
        self.savetxt = savetxt_patch.start()

    def test_close_args(self):
        db = self.db

        db.close()

        self.assertEqual(self.savetxt.call_count, 2)

        for call, var_name in enumerate(db.var_names):
            fname, data = self.savetxt.call_args_list[call][0]
            self.assertEqual(fname, 'textdb/chain-0/{}.txt'.format(var_name))
            npt.assert_equal(data, db.trace[var_name].reshape(-1, data.size))

    def test_close_shape(self):
        db = self.db

        fh = StringIO()
        self.shape_fh.return_value.__enter__.return_value = fh
        db.close()
        self.shape_fh.assert_called_with('textdb/chain-0', 'w')

        shape_result = fh.getvalue()
        expected = {var_name: [self.draws] + list(var_shape)
                    for var_name, var_shape in db.var_shapes.items()}
        self.assertEqual(json.loads(shape_result), expected)


def test__chain_dir_to_chain():
    assert text._chain_dir_to_chain('/path/to/chain-0') == 0
    assert text._chain_dir_to_chain('chain-0') == 0


class TestTextLoad(TextTestCase):

    def setUp(self):
        super(TestTextLoad, self).setUp()

        data = {'chain-1/x.txt': np.zeros(4), 'chain-1/y.txt': np.ones(2)}
        loadtxt_patch = mock.patch('pymc.backends.text.np.loadtxt')
        self.addCleanup(loadtxt_patch.stop)
        self.loadtxt = loadtxt_patch.start()

        chain_patch = mock.patch('pymc.backends.text._get_chain_dirs')
        self.addCleanup(chain_patch.stop)
        self._get_chain_dirs = chain_patch.start()

    def test_load_model_supplied_scalar(self):
        draws = 4
        self._get_chain_dirs.return_value = {0: 'chain-0'}
        fh = StringIO(json.dumps({'x': (draws,)}))
        self.shape_fh.return_value.__enter__.return_value = fh

        data = np.zeros(draws)
        self.loadtxt.return_value = data

        trace = text.load('textdb', model=self.model)
        npt.assert_equal(trace.samples[0]['x'], data)

    def test_load_model_supplied_1d(self):
        draws = 4
        var_shape = (2,)
        self._get_chain_dirs.return_value = {0: 'chain-0'}
        fh = StringIO(json.dumps({'x': (draws,) + var_shape}))
        self.shape_fh.return_value.__enter__.return_value = fh

        data = np.zeros((draws,) + var_shape)
        self.loadtxt.return_value = data.reshape(-1, data.size)

        db = text.load('textdb', model=self.model)
        npt.assert_equal(db['x'], data)

    def test_load_model_supplied_2d(self):
        draws = 4
        var_shape = (2, 3)
        self._get_chain_dirs.return_value = {0: 'chain-0'}
        fh = StringIO(json.dumps({'x': (draws,) + var_shape}))
        self.shape_fh.return_value.__enter__.return_value = fh

        data = np.zeros((draws,) + var_shape)
        self.loadtxt.return_value = data.reshape(-1, data.size)

        db = text.load('textdb', model=self.model)
        npt.assert_equal(db['x'], data)

    def test_load_model_supplied_multichain_chains(self):
        draws = 4
        self._get_chain_dirs.return_value = {0: 'chain-0', 1: 'chain-1'}

        def chain_fhs():
            for chain in [0, 1]:
                yield StringIO(json.dumps({'x': (draws,)}))
        fhs = chain_fhs()

        self.shape_fh.return_value.__enter__ = lambda x: next(fhs)

        data = np.zeros(draws)
        self.loadtxt.return_value = data

        trace = text.load('textdb', model=self.model)

        self.assertEqual(trace.chains, [0, 1])

    def test_load_model_supplied_multichain_chains_select_one(self):
        draws = 4
        self._get_chain_dirs.return_value = {0: 'chain-0', 1: 'chain-1'}

        def chain_fhs():
            for chain in [0, 1]:
                yield StringIO(json.dumps({'x': (draws,)}))
        fhs = chain_fhs()

        self.shape_fh.return_value.__enter__ = lambda x: next(fhs)

        data = np.zeros(draws)
        self.loadtxt.return_value = data

        trace = text.load('textdb', model=self.model, chains=[1])

        self.assertEqual(trace.chains, [1])
