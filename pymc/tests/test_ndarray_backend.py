import numpy as np
import numpy.testing as npt
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest

from pymc.backends import ndarray


class TestNDArraySampling(unittest.TestCase):

    def setUp(self):
        self.variables = ['x', 'y']
        self.model = mock.Mock()
        self.model.unobserved_RVs = self.variables
        self.model.fastfn = mock.MagicMock()

        with mock.patch('pymc.backends.base.modelcontext') as context:
            context.return_value = self.model
            self.db = ndarray.NDArray()

    def test_setup_scalar(self):
        db = self.db
        db.var_shapes = {'x': ()}
        draws, chain = 3, 0
        db.setup(draws, chain)
        npt.assert_equal(db.trace.samples[chain]['x'], np.zeros(draws))

    def test_setup_1d(self):
        db = self.db
        shape = (2,)
        db.var_shapes = {'x': shape}
        draws, chain = 3, 0
        db.setup(draws, chain)
        npt.assert_equal(db.trace.samples[chain]['x'], np.zeros((draws,) + shape))

    def test_record(self):
        db = self.db
        draws = 3

        db.var_shapes = {'x': (), 'y': (4,)}
        db.setup(draws, chain=0)

        def just_ones(*args):
            while True:
                yield 1.

        db.fn = just_ones
        db.draw_idx = 0

        db.record(point=None)
        npt.assert_equal(1., db.trace.get_values('x')[0])
        npt.assert_equal(np.ones(4), db.trace['y'][0])

    def test_clean_interrupt(self):
        db = self.db
        db.setup(draws=10, chain=0)
        db.trace.samples = {0: {'x': np.zeros(10), 'y': np.zeros((10, 5))}}
        db.draw_idx = 3
        db.close()
        npt.assert_equal(np.zeros(3), db.trace['x'])
        npt.assert_equal(np.zeros((3, 5)), db.trace['y'])


class TestNDArraySelection(unittest.TestCase):

    def setUp(self):
        var_names = ['x', 'y']
        var_shapes = {'x': (), 'y': (2,)}
        draws = 3
        self.trace = ndarray.Trace(var_names)
        self.trace.samples = {0:
                              {'x': np.zeros(draws),
                               'y': np.zeros((draws, 2))}}
        self.draws = draws
        self.var_names = var_names
        self.var_shapes = var_shapes

    def test_chains_single_chain(self):
        self.trace.chains == [0]

    def test_nchains_single_chain(self):
        self.trace.nchains == 1

    def test_get_values_default(self):
        base_shape = (self.draws,)
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        xsample = self.trace.get_values('x')
        npt.assert_equal(np.zeros(base_shape + xshape), xsample)

        ysample = self.trace.get_values('y')
        npt.assert_equal(np.zeros(base_shape + yshape), ysample)

    def test_get_values_burn_keyword(self):
        base_shape = (self.draws,)
        burn = 2
        chain = 0

        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        ## Make traces distinguishable
        self.trace.samples[chain]['x'][:burn] = np.ones((burn,) + xshape)
        self.trace.samples[chain]['y'][:burn] = np.ones((burn,) + yshape)

        xsample = self.trace.get_values('x', burn=burn)
        npt.assert_equal(np.zeros(base_shape + xshape)[burn:], xsample)

        ysample = self.trace.get_values('y', burn=burn)
        npt.assert_equal(np.zeros(base_shape + yshape)[burn:], ysample)

    def test_get_values_thin_keyword(self):
        base_shape = (self.draws,)
        thin = 2
        chain = 0
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        ## Make traces distinguishable
        xthin = np.ones((self.draws,) + xshape)[::thin]
        ythin = np.ones((self.draws,) + yshape)[::thin]
        self.trace.samples[chain]['x'][::thin] = xthin
        self.trace.samples[chain]['y'][::thin] = ythin

        xsample = self.trace.get_values('x', thin=thin)
        npt.assert_equal(xthin, xsample)

        ysample = self.trace.get_values('y', thin=thin)
        npt.assert_equal(ythin, ysample)

    def test_point(self):
        idx = 2
        chain = 0
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        ## Make traces distinguishable
        self.trace.samples[chain]['x'][idx] = 1.
        self.trace.samples[chain]['y'][idx] = 1.

        point = self.trace.point(idx)
        expected = {'x': np.squeeze(np.ones(xshape)),
                    'y': np.squeeze(np.ones(yshape))}

        for var_name, value in expected.items():
            npt.assert_equal(value, point[var_name])

    def test_slice(self):
        base_shape = (self.draws,)
        burn = 2
        chain = 0

        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        ## Make traces distinguishable
        self.trace.samples[chain]['x'][:burn] = np.ones((burn,) + xshape)
        self.trace.samples[chain]['y'][:burn] = np.ones((burn,) + yshape)

        sliced = self.trace[burn:]

        expected = {'x': np.zeros(base_shape + xshape)[burn:],
                    'y': np.zeros(base_shape + yshape)[burn:]}

        for var_name, var_shape in self.var_shapes.items():
            npt.assert_equal(sliced.samples[chain][var_name],
                             expected[var_name])


class TestNDArrayMultipleChains(unittest.TestCase):

    def setUp(self):
        var_names = ['x', 'y']
        var_shapes = {'x': (), 'y': (2,)}
        draws = 3
        self.trace = ndarray.Trace(var_names)
        self.trace.samples = {0:
                              {'x': np.zeros(draws),
                               'y': np.zeros((draws, 2))},
                              1:
                              {'x': np.ones(draws),
                               'y': np.ones((draws, 2))}}
        self.draws = draws
        self.var_names = var_names
        self.var_shapes = var_shapes
        self.total_draws = 2 * draws

    def test_chains_multichain(self):
        self.trace.chains == [0, 1]

    def test_nchains_multichain(self):
        self.trace.nchains == [0, 1]

    def test_get_values_multi_default(self):
        sample = self.trace.get_values('x')
        xshape = self.var_shapes['x']

        expected = [np.zeros((self.draws,) + xshape),
                    np.ones((self.draws,) + xshape)]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_chains_only_one_element(self):
        sample = self.trace.get_values('x', chains=[0])
        xshape = self.var_shapes['x']

        expected = np.zeros((self.draws,) + xshape)

        npt.assert_equal(sample, expected)

    def test_get_values_multi_chains_two_element_reversed(self):
        sample = self.trace.get_values('x', chains=[1, 0])
        xshape = self.var_shapes['x']

        expected = [np.ones((self.draws,) + xshape),
                    np.zeros((self.draws,) + xshape)]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_combine(self):
        sample = self.trace.get_values('x', combine=True)
        xshape = self.var_shapes['x']

        expected = np.concatenate([np.zeros((self.draws,) + xshape),
                                   np.ones((self.draws,) + xshape)])
        npt.assert_equal(sample, expected)

    def test_get_values_multi_burn(self):
        sample = self.trace.get_values('x', burn=2)
        xshape = self.var_shapes['x']

        expected = [np.zeros((self.draws,) + xshape)[2:],
                    np.ones((self.draws,) + xshape)[2:]]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_burn_combine(self):
        sample = self.trace.get_values('x', burn=2, combine=True)
        xshape = self.var_shapes['x']

        expected = np.concatenate([np.zeros((self.draws,) + xshape)[2:],
                                   np.ones((self.draws,) + xshape)[2:]])
        npt.assert_equal(sample, expected)

    def test_get_values_multi_burn_one_active_chain(self):
        self.trace.active_chains = 0
        sample = self.trace.get_values('x', burn=2)
        xshape = self.var_shapes['x']

        expected = np.zeros((self.draws,) + xshape)[2:]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_thin(self):
        sample = self.trace.get_values('x', thin=2)
        xshape = self.var_shapes['x']

        expected = [np.zeros((self.draws,) + xshape)[::2],
                    np.ones((self.draws,) + xshape)[::2]]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_thin_combine(self):
        sample = self.trace.get_values('x', thin=2, combine=True)
        xshape = self.var_shapes['x']

        expected = np.concatenate([np.zeros((self.draws,) + xshape)[::2],
                                   np.ones((self.draws,) + xshape)[::2]])
        npt.assert_equal(sample, expected)

    def test_multichain_point(self):
        idx = 2
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        point = self.trace.point(idx)
        expected = {'x': np.squeeze(np.ones(xshape)),
                    'y': np.squeeze(np.ones(yshape))}

        for var_name, value in expected.items():
            npt.assert_equal(value, point[var_name])

    def test_multichain_point_chain_arg(self):
        idx = 2
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        point = self.trace.point(idx, chain=0)
        expected = {'x': np.squeeze(np.zeros(xshape)),
                    'y': np.squeeze(np.zeros(yshape))}

        for var_name, value in expected.items():
            npt.assert_equal(value, point[var_name])

    def test_multichain_point_change_default_chain(self):
        idx = 2
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        self.trace.default_chain = 0

        point = self.trace.point(idx)
        expected = {'x': np.squeeze(np.zeros(xshape)),
                    'y': np.squeeze(np.zeros(yshape))}

        for var_name, value in expected.items():
            npt.assert_equal(value, point[var_name])

    def test_multichain_slice(self):
        base_shapes = [(self.draws,)] * 2
        burn = 2

        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        expected = {0:
                    {'x': np.zeros((self.draws, ) + xshape)[burn:],
                     'y': np.zeros((self.draws, ) + yshape)[burn:]},
                    1:
                    {'x': np.ones((self.draws, ) + xshape)[burn:],
                     'y': np.ones((self.draws, ) + yshape)[burn:]}}

        sliced = self.trace[burn:]

        for chain in self.trace.chains:
            for var_name, var_shape in self.var_shapes.items():
                npt.assert_equal(sliced.samples[chain][var_name],
                                 expected[chain][var_name])

class TestMergeChains(unittest.TestCase):

    def setUp(self):
        var_names = ['x', 'y']
        var_shapes = {'x': (), 'y': (2,)}
        draws = 3
        self.trace1 = ndarray.Trace(var_names)
        self.trace1.samples = {0:
                              {'x': np.zeros(draws),
                               'y': np.zeros((draws, 2))}}

        self.trace2 = ndarray.Trace(var_names)
        self.trace2.samples = {1:
                               {'x': np.ones(draws),
                                'y': np.ones((draws, 2))}}
        self.draws = draws
        self.var_names = var_names
        self.var_shapes = var_shapes
        self.total_draws = 2 * draws

    def test_merge_chains_two_traces(self):
        self.trace1.merge_chains([self.trace2])
        self.assertEqual(self.trace1.samples[1], self.trace2.samples[1])

    def test_merge_chains_two_traces_same_slot(self):
        self.trace2.samples = self.trace1.samples

        with self.assertRaises(ValueError):
            self.trace1.merge_chains([self.trace2])
