import numpy as np
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest
import nose

from pymc.backends import base


class TestBaseInit(unittest.TestCase):

    def setUp(self):
        self.variables = ['x', 'y']
        self.model = mock.Mock()
        self.model.unobserved_RVs = self.variables
        self.model.fastfn = mock.MagicMock()

    def test_base_init_just_name(self):
        with mock.patch('pymc.backends.base.modelcontext') as context:
            variables = self.variables
            context.return_value = self.model

            db = base.Backend('name')

            context.assert_called_once_with(None)
            self.assertEqual(db.variables, variables)
            self.assertEqual(db.var_names, variables)
            self.model.fastfn.assert_called_once_with(variables)

    def test_base_init_model_supplied(self):
        db = base.Backend('name', model=self.model)

        self.assertEqual(db.variables, self.variables)
        self.assertEqual(db.var_names, self.variables)
        self.model.fastfn.assert_called_once_with(self.variables)

    def test_base_init_variables_supplied(self):
        with mock.patch('pymc.backends.base.modelcontext') as context:
            variables = ['a', 'b']
            context.return_value = self.model

            db = base.Backend('name', variables=variables)

            context.assert_called_once_with(None)
            self.assertEqual(db.variables, variables)
            self.assertEqual(db.var_names, variables)
            self.model.fastfn.assert_called_once_with(variables)

    def test_base_setup_samples_default_chain(self):
        with mock.patch('pymc.backends.base.modelcontext') as context:
            variables = ['a', 'b']
            context.return_value = self.model

            db = base.Backend('name', variables=variables)

        db._create_trace = mock.Mock()
        db.var_shapes = {'x': (), 'y': (10,)}
        draws = 3

        patch = mock.patch('pymc.backends.base.Backend._initialize_trace')
        with patch as init_trace:
            db.setup_samples(draws, 0)

        init_trace.assert_called_with()
        db._create_trace.assert_any_call(0, 'x', [draws])
        db._create_trace.assert_any_call(0, 'y', [draws, 10])


class TestBaseTrace(unittest.TestCase):

    def setUp(self):
        var_names = ['x']
        self.trace = base.Trace(var_names)
        self.trace.samples = {0: {'x': None}}

    def test_nchains(self):

        self.assertEqual(self.trace.nchains, 1)

        self.trace.samples[1] = {'y': None}
        self.assertEqual(self.trace.nchains, 2)

    def test_chains(self):
        self.assertEqual(self.trace.chains, [0])

        self.trace.samples[1] = {'y': None}
        self.assertEqual(self.trace.chains, [0, 1])

    def test_chains_not_sequential(self):
        self.trace.samples[4] = {'y': None}
        self.assertEqual(self.trace.chains, [0, 4])

    def test_default_chain_one_chain(self):
        self.assertEqual(self.trace.default_chain, 0)

    def test_default_chain_multiple_chain(self):
        self.trace.samples[1] = {'y': None}
        self.assertEqual(self.trace.default_chain, 1)

    def test_default_chain_multiple_chains_set(self):
        self.trace.samples[1] = {'y': None}
        self.trace.default_chain = 0
        self.assertEqual(self.trace.default_chain, 0)

    def test_active_chains(self):
        self.assertEqual(self.trace.chains, self.trace.active_chains)
        self.trace.samples[1] = {'y': None}
        self.assertEqual(self.trace.chains, self.trace.active_chains)

    def test_active_chains_set_with_int(self):
        self.trace.samples[1] = {'y': None}
        self.trace.active_chains = 0
        self.assertEqual(self.trace.active_chains, [0])


class TestMergeChains(unittest.TestCase):

    def test_merge_chains_one_trace(self):
        trace = mock.Mock()
        trace.samples = {0: {'x': 0, 'y': 1}}
        merged = base.merge_chains([trace])
        self.assertEqual(trace.samples, merged.samples)

    def test_merge_chains_two_traces(self):
        trace1 = mock.Mock()
        trace1.samples = {0: {'x': 0, 'y': 1}}
        trace1.chains = [0]

        trace2 = mock.Mock()
        trace2.samples = {1: {'x': 3, 'y': 4}}
        trace2.chains = [1]

        merged = base.merge_chains([trace1, trace2])
        self.assertEqual(trace1.samples[0], merged.samples[0])
        self.assertEqual(trace2.samples[1], merged.samples[1])

    def test_merge_chains_two_traces_same_slot(self):
        trace1 = mock.Mock()
        trace1.samples = {0: {'x': 0, 'y': 1}}
        trace1.chains = [0]

        trace2 = mock.Mock()
        trace2.samples = {0: {'x': 3, 'y': 4}}
        trace2.chains = [0]

        with self.assertRaises(ValueError):
            base.merge_chains([trace1, trace2])
