import numpy as np
import numpy.testing as npt
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest

from pymc.backends import base, ndarray


class NDArrayTestCase(unittest.TestCase):
    def setUp(self):
        self.varnames = ['x', 'y']
        self.model = mock.Mock()
        self.model.unobserved_RVs = self.varnames
        self.model.fastfn = mock.MagicMock()

        with mock.patch('pymc.backends.base.modelcontext') as context:
            context.return_value = self.model
            self.trace = ndarray.NDArray()


class TestNDArraySampling(NDArrayTestCase):

    def test_setup_scalar(self):
        trace = self.trace
        trace.var_shapes = {'x': ()}
        draws, chain = 3, 0
        trace.setup(draws, chain)
        npt.assert_equal(trace.samples['x'], np.zeros(draws))

    def test_setup_1d(self):
        trace = self.trace
        shape = (2,)
        trace.var_shapes = {'x': shape}
        draws, chain = 3, 0
        trace.setup(draws, chain)
        npt.assert_equal(trace.samples['x'], np.zeros((draws,) + shape))

    def test_record(self):
        trace = self.trace
        draws = 3

        trace.var_shapes = {'x': (), 'y': (4,)}
        trace.setup(draws, chain=0)

        def just_ones(*args):
            while True:
                yield 1.

        trace.fn = just_ones
        trace.draw_idx = 0

        trace.record(point=None)
        npt.assert_equal(1., trace.get_values('x')[0])
        npt.assert_equal(np.ones(4), trace['y'][0])

    def test_clean_interrupt(self):
        trace = self.trace
        trace.setup(draws=10, chain=0)
        trace.samples = {'x': np.zeros(10), 'y': np.zeros((10, 5))}
        trace.draw_idx = 3
        trace.close()
        npt.assert_equal(np.zeros(3), trace['x'])
        npt.assert_equal(np.zeros((3, 5)), trace['y'])

    def test_standard_close(self):
        trace = self.trace
        trace.setup(draws=10, chain=0)
        trace.samples = {'x': np.zeros(10), 'y': np.zeros((10, 5))}
        trace.draw_idx = 10
        trace.close()
        npt.assert_equal(np.zeros(10), trace['x'])
        npt.assert_equal(np.zeros((10, 5)), trace['y'])


class TestNDArraySelection(NDArrayTestCase):

    def setUp(self):
        super(TestNDArraySelection, self).setUp()
        draws = 3
        self.trace.samples = {'x': np.zeros(draws),
                              'y': np.zeros((draws, 2))}
        self.draws = draws
        var_shapes = {'x': (), 'y': (2,)}
        self.var_shapes = var_shapes
        self.trace.var_shapes = var_shapes

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
        self.trace.samples['x'][:burn] = np.ones((burn,) + xshape)
        self.trace.samples['y'][:burn] = np.ones((burn,) + yshape)

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
        self.trace.samples['x'][::thin] = xthin
        self.trace.samples['y'][::thin] = ythin

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
        self.trace.samples['x'][idx] = 1.
        self.trace.samples['y'][idx] = 1.

        point = self.trace.point(idx)
        expected = {'x': np.squeeze(np.ones(xshape)),
                    'y': np.squeeze(np.ones(yshape))}

        for varname, value in expected.items():
            npt.assert_equal(value, point[varname])

    def test_slice(self):
        base_shape = (self.draws,)
        burn = 2
        chain = 0

        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        ## Make traces distinguishable
        self.trace.samples['x'][:burn] = np.ones((burn,) + xshape)
        self.trace.samples['y'][:burn] = np.ones((burn,) + yshape)

        sliced = self.trace[burn:]

        expected = {'x': np.zeros(base_shape + xshape)[burn:],
                    'y': np.zeros(base_shape + yshape)[burn:]}

        for varname, var_shape in self.var_shapes.items():
            npt.assert_equal(sliced.samples[varname],
                             expected[varname])


class TestNDArrayMultipleChains(unittest.TestCase):

    def setUp(self):
        varnames = ['x', 'y']
        var_shapes = {'x': (), 'y': (2,)}
        draws = 3

        self.varnames = varnames
        self.var_shapes = var_shapes
        self.draws = draws
        self.total_draws = 2 * draws

        self.model = mock.Mock()
        self.model.unobserved_RVs = varnames
        self.model.fastfn = mock.MagicMock()
        with mock.patch('pymc.backends.base.modelcontext') as context:
            context.return_value = self.model
            trace0 = ndarray.NDArray(varnames)
            trace0.samples = {'x': np.zeros(draws),
                              'y': np.zeros((draws, 2))}
            trace0.chain = 0

            trace1 = ndarray.NDArray(varnames)
            trace1.samples = {'x': np.ones(draws),
                              'y': np.ones((draws, 2))}
            trace1.chain = 1

        self.mtrace = base.MultiTrace([trace0, trace1])

    def test_chains_multichain(self):
        self.mtrace.chains == [0, 1]

    def test_nchains_multichain(self):
        self.mtrace.nchains == 1

    def test_get_values_multi_default(self):
        sample = self.mtrace.get_values('x')
        xshape = self.var_shapes['x']

        expected = [np.zeros((self.draws,) + xshape),
                    np.ones((self.draws,) + xshape)]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_chains_one_chain_list_arg(self):
        sample = self.mtrace.get_values('x', chains=[0])
        xshape = self.var_shapes['x']
        expected = np.zeros((self.draws,) + xshape)
        npt.assert_equal(sample, expected)

    def test_get_values_multi_chains_one_chain_int_arg(self):
        npt.assert_equal(self.mtrace.get_values('x', chains=[0]),
                         self.mtrace.get_values('x', chains=0))

    def test_get_values_multi_chains_two_element_reversed(self):
        sample = self.mtrace.get_values('x', chains=[1, 0])
        xshape = self.var_shapes['x']

        expected = [np.ones((self.draws,) + xshape),
                    np.zeros((self.draws,) + xshape)]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_combine(self):
        sample = self.mtrace.get_values('x', combine=True)
        xshape = self.var_shapes['x']

        expected = np.concatenate([np.zeros((self.draws,) + xshape),
                                   np.ones((self.draws,) + xshape)])
        npt.assert_equal(sample, expected)

    def test_get_values_multi_burn(self):
        sample = self.mtrace.get_values('x', burn=2)
        xshape = self.var_shapes['x']

        expected = [np.zeros((self.draws,) + xshape)[2:],
                    np.ones((self.draws,) + xshape)[2:]]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_burn_combine(self):
        sample = self.mtrace.get_values('x', burn=2, combine=True)
        xshape = self.var_shapes['x']

        expected = np.concatenate([np.zeros((self.draws,) + xshape)[2:],
                                   np.ones((self.draws,) + xshape)[2:]])
        npt.assert_equal(sample, expected)

    def test_get_values_multi_thin(self):
        sample = self.mtrace.get_values('x', thin=2)
        xshape = self.var_shapes['x']

        expected = [np.zeros((self.draws,) + xshape)[::2],
                    np.ones((self.draws,) + xshape)[::2]]
        npt.assert_equal(sample, expected)

    def test_get_values_multi_thin_combine(self):
        sample = self.mtrace.get_values('x', thin=2, combine=True)
        xshape = self.var_shapes['x']

        expected = np.concatenate([np.zeros((self.draws,) + xshape)[::2],
                                   np.ones((self.draws,) + xshape)[::2]])
        npt.assert_equal(sample, expected)

    def test_multichain_point(self):
        idx = 2
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        point = self.mtrace.point(idx)
        expected = {'x': np.squeeze(np.ones(xshape)),
                    'y': np.squeeze(np.ones(yshape))}

        for varname, value in expected.items():
            npt.assert_equal(value, point[varname])

    def test_multichain_point_chain_arg(self):
        idx = 2
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        point = self.mtrace.point(idx, chain=0)
        expected = {'x': np.squeeze(np.zeros(xshape)),
                    'y': np.squeeze(np.zeros(yshape))}

        for varname, value in expected.items():
            npt.assert_equal(value, point[varname])

    def test_multichain_slice(self):
        burn = 2
        xshape = self.var_shapes['x']
        yshape = self.var_shapes['y']

        expected = {0:
                    {'x': np.zeros((self.draws, ) + xshape)[burn:],
                     'y': np.zeros((self.draws, ) + yshape)[burn:]},
                    1:
                    {'x': np.ones((self.draws, ) + xshape)[burn:],
                     'y': np.ones((self.draws, ) + yshape)[burn:]}}

        sliced = self.mtrace[burn:]

        for chain in self.mtrace.chains:
            for varname, var_shape in self.var_shapes.items():
                npt.assert_equal(sliced.get_values(varname, chains=[0]),
                                 expected[0][varname])
                npt.assert_equal(sliced.get_values(varname, chains=[1]),
                                 expected[1][varname])
