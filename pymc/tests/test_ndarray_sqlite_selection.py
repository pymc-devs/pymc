import os
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


class TestCompareNDArraySQLite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if TEST_PARALLEL:
            njobs = 2
        else:
            njobs = 1

        data = np.random.normal(size=(3, 20))
        n = 1

        model = pm.Model()
        draws = 5
        with model:
            x = pm.Normal('x', 0, 1., shape=n)

            start = {'x': 0.}
            step = pm.Metropolis()
            cls.db = 'test.db'

            try:
                cls.ntrace = pm.sample(draws, step=step,
                                       njobs=njobs, random_seed=9)
                cls.strace = pm.sample(draws, step=step,
                                       njobs=njobs, random_seed=9,
                                       trace=pm.backends.SQLite(cls.db))
                ## Extend each trace.
                cls.ntrace = pm.sample(draws, step=step,
                                       njobs=njobs, random_seed=4,
                                       trace=cls.ntrace)
                cls.strace = pm.sample(draws, step=step,
                                       njobs=njobs, random_seed=4,
                                       trace=cls.strace)
                cls.draws = draws * 2  # Account for extension.
            except:
                remove_file_or_directory(cls.db)
                raise

    @classmethod
    def tearDownClass(cls):
        remove_file_or_directory(cls.db)

    def test_chain_length(self):
        assert self.ntrace.nchains == self.strace.nchains
        assert len(self.ntrace) == len(self.strace)

    def test_number_of_draws(self):
        nvalues = self.ntrace.get_values('x', squeeze=False)
        svalues = self.strace.get_values('x', squeeze=False)
        assert nvalues[0].shape[0] == self.draws
        assert svalues[0].shape[0] == self.draws

    def test_get_item(self):
        npt.assert_equal(self.ntrace['x'], self.strace['x'])

    def test_get_values(self):
        for cf in [False, True]:
            npt.assert_equal(self.ntrace.get_values('x', combine=cf),
                             self.strace.get_values('x', combine=cf))

    def test_get_values_no_squeeze(self):
        npt.assert_equal(self.ntrace.get_values('x', combine=False,
                                                squeeze=False),
                         self.strace.get_values('x', combine=False,
                                                squeeze=False))

    def test_get_values_combine_and_no_squeeze(self):
        npt.assert_equal(self.ntrace.get_values('x', combine=True,
                                                squeeze=False),
                         self.strace.get_values('x', combine=True,
                                                squeeze=False))

    def test_get_values_with_burn(self):
        for cf in [False, True]:
            npt.assert_equal(self.ntrace.get_values('x', combine=cf, burn=3),
                             self.strace.get_values('x', combine=cf, burn=3))

            ## Burn to one value.
            npt.assert_equal(self.ntrace.get_values('x', combine=cf,
                                                    burn=self.draws - 1),
                             self.strace.get_values('x', combine=cf,
                                                    burn=self.draws - 1))

    def test_get_values_with_thin(self):
        for cf in [False, True]:
            npt.assert_equal(self.ntrace.get_values('x', combine=cf, thin=2),
                             self.strace.get_values('x', combine=cf, thin=2))

    def test_get_values_with_burn_and_thin(self):
        for cf in [False, True]:
            npt.assert_equal(self.ntrace.get_values('x', combine=cf,
                                                    burn=2, thin=2),
                             self.strace.get_values('x', combine=cf,
                                                    burn=2, thin=2))

    def test_get_values_with_chains_arg(self):
        for cf in [False, True]:
            npt.assert_equal(self.ntrace.get_values('x', chains=[0]),
                             self.strace.get_values('x', chains=[0]))

    def test_point(self):
        npoint, spoint = self.ntrace[4], self.strace[4]
        npt.assert_equal(npoint['x'], spoint['x'])

    def test_point_with_chain_arg(self):
        npoint = self.ntrace.point(4, chain=0)
        spoint = self.strace.point(4, chain=0)
        npt.assert_equal(npoint['x'], spoint['x'])
