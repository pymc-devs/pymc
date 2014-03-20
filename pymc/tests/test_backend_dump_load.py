import os
import shutil
import numpy as np
import numpy.testing as npt
import unittest

import pymc as pm

## Set to False to keep effect of cea5659. Should this be set to True?
TEST_PARALLEL = False


def remove_file_or_directory(name):
    try:
        os.remove(name)
    except OSError:
        shutil.rmtree(name, ignore_errors=True)


class DumpLoadTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if TEST_PARALLEL:
            njobs = 2
        else:
            njobs = 1

        data = np.random.normal(size=(2, 20))
        model = pm.Model()
        with model:
            x = pm.Normal('x', mu=.5, tau=2. ** -2, shape=(2, 1))
            z = pm.Beta('z', alpha=10, beta=5.5)
            d = pm.Normal('data', mu=x, tau=.75 ** -2, observed=data)
            data = np.random.normal(size=(3, 20))
            n = 1

        draws = 5
        cls.draws = draws

        with model:
            try:
                cls.trace = pm.sample(n, step=pm.Metropolis(),
                                      trace=cls.backend(cls.db),
                                      njobs=2)
                cls.dumped = cls.load_func(cls.db)
            except:
                remove_file_or_directory(cls.db)
                raise

    @classmethod
    def tearDownClass(cls):
        remove_file_or_directory(cls.db)


class TestTextDumpLoad(DumpLoadTestCase):

    backend = pm.backends.Text
    load_func = staticmethod(pm.backends.text.load)
    db = 'text-db'

    def test_nchains(self):
        self.assertEqual(self.trace.nchains, self.dumped.nchains)

    def test_varnames(self):
        trace_names = list(sorted(self.trace.varnames))
        dumped_names = list(sorted(self.dumped.varnames))
        self.assertEqual(trace_names, dumped_names)

    def test_values(self):
        trace = self.trace
        dumped = self.dumped
        for chain in trace.chains:
            for varname in trace.varnames:
                data = trace.get_values(varname, chains=[chain])
                dumped_data = dumped.get_values(varname, chains=[chain])
                npt.assert_equal(data, dumped_data)


class TestSQLiteDumpLoad(DumpLoadTestCase):

    backend = pm.backends.SQLite
    load_func = staticmethod(pm.backends.sqlite.load)
    db = 'test.db'

    def test_nchains(self):
        self.assertEqual(self.trace.nchains, self.dumped.nchains)

    def test_varnames(self):
        trace_names = list(sorted(self.trace.varnames))
        dumped_names = list(sorted(self.dumped.varnames))
        self.assertEqual(trace_names, dumped_names)

    def test_values(self):
        trace = self.trace
        dumped = self.dumped
        for chain in trace.chains:
            for varname in trace.varnames:
                data = trace.get_values(varname, chains=[chain])
                dumped_data = dumped.get_values(varname, chains=[chain])
                npt.assert_equal(data, dumped_data)
