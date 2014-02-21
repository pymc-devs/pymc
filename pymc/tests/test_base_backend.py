import numpy as np
try:
    import unittest.mock as mock  # py3
except ImportError:
    import mock
import unittest

from pymc.backends import base


class TestMultiTrace(unittest.TestCase):

    def test_multitrace_init_unique_chains(self):
        trace0 = mock.Mock()
        trace0.chain = 0
        trace1 = mock.Mock()
        trace1.chain = 1
        mtrace = base.MultiTrace([trace0, trace1])
        self.assertEqual(mtrace._traces[0], trace0)
        self.assertEqual(mtrace._traces[1], trace1)

    def test_multitrace_init_nonunique_chains(self):
        trace0 = mock.Mock()
        trace0.chain = 0
        trace1 = mock.Mock()
        trace1.chain = 0
        self.assertRaises(ValueError,
                          base.MultiTrace, [trace0, trace1])


class TestMergeChains(unittest.TestCase):

    def test_merge_traces_unique_chains(self):
            trace0 = mock.Mock()
            trace0.chain = 0
            mtrace0 = base.MultiTrace([trace0])

            trace1 = mock.Mock()
            trace1.chain = 1
            mtrace1 = base.MultiTrace([trace1])

            merged = base.merge_traces([mtrace0, mtrace1])
            self.assertEqual(merged._traces[0], trace0)
            self.assertEqual(merged._traces[1], trace1)

    def test_merge_traces_nonunique_chains(self):
            trace0 = mock.Mock()
            trace0.chain = 0
            mtrace0 = base.MultiTrace([trace0])

            trace1 = mock.Mock()
            trace1.chain = 0
            mtrace1 = base.MultiTrace([trace1])

            self.assertRaises(ValueError,
                              base.merge_traces, [mtrace0, mtrace1])
