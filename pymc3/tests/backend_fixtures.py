import unittest
import numpy as np
import numpy.testing as npt
import os
import shutil

from pymc3.tests import models
from pymc3.backends import base


class ModelBackendSetupTestCase(unittest.TestCase):
    """Set up a backend trace.

    Provides the attributes
    - test_point
    - model
    - trace
    - draws

    Children must define
    - backend
    - name
    - shape
    """
    def setUp(self):
        self.test_point, self.model, _ = models.beta_bernoulli(self.shape)
        with self.model:
            self.trace = self.backend(self.name)
        self.draws, self.chain = 3, 0
        self.trace.setup(self.draws, self.chain)

    def tearDown(self):
        if self.name is not None:
            remove_file_or_directory(self.name)


class ModelBackendSampledTestCase(unittest.TestCase):
    """Setup and sample a backend trace.

    Provides the attributes
    - test_point
    - model
    - mtrace (MultiTrace object)
    - draws
    - expected
        Expected values mapped to chain number and variable name.

    Children must define
    - backend
    - name
    - shape
    """
    @classmethod
    def setUpClass(cls):
        cls.test_point, cls.model, _ = models.beta_bernoulli(cls.shape)
        with cls.model:
            trace0 = cls.backend(cls.name)
            trace1 = cls.backend(cls.name)

        cls.draws = 5
        trace0.setup(cls.draws, chain=0)
        trace1.setup(cls.draws, chain=1)

        varnames = list(cls.test_point.keys())
        shapes = {varname: value.shape
                  for varname, value in cls.test_point.items()}
        dtypes = {varname: value.dtype
                  for varname, value in cls.test_point.items()}

        cls.expected = {0: {}, 1: {}}
        for varname in varnames:
            mcmc_shape = (cls.draws,) + shapes[varname]
            values = np.arange(cls.draws * np.prod(shapes[varname]),
                               dtype=dtypes[varname])
            cls.expected[0][varname] = values.reshape(mcmc_shape)
            cls.expected[1][varname] = values.reshape(mcmc_shape) * 100

        for idx in range(cls.draws):
            point0 = {varname: cls.expected[0][varname][idx, ...]
                      for varname in varnames}
            point1 = {varname: cls.expected[1][varname][idx, ...]
                      for varname in varnames}
            trace0.record(point=point0)
            trace1.record(point=point1)
        trace0.close()
        trace1.close()
        cls.mtrace = base.MultiTrace([trace0, trace1])

    @classmethod
    def tearDownClass(cls):
        if cls.name is not None:
            remove_file_or_directory(cls.name)

    def test_varnames_nonempty(self):
        # Make sure the test_point has variables names because many
        # tests rely on looping through these and would pass silently
        # if the loop is never entered.
        assert list(self.test_point.keys())


class SamplingTestCase(ModelBackendSetupTestCase):
    """Test backend sampling.

    Children must define
    - backend
    - name
    - shape
    """

    def test_standard_close(self):
        for idx in range(self.draws):
            point = {varname: np.tile(idx, value.shape)
                     for varname, value in self.test_point.items()}
            self.trace.record(point=point)
        self.trace.close()

        for varname in self.test_point.keys():
            npt.assert_equal(self.trace[varname][0, ...],
                             np.zeros(self.trace.var_shapes[varname]))
            last_idx = self.draws - 1
            npt.assert_equal(self.trace[varname][last_idx, ...],
                             np.tile(last_idx, self.trace.var_shapes[varname]))

    def test_clean_interrupt(self):
        self.trace.record(point=self.test_point)
        self.trace.close()
        for varname in self.test_point.keys():
            self.assertEqual(self.trace[varname].shape[0], 1)


class SelectionTestCase(ModelBackendSampledTestCase):
    """Test backend selection.

    Children must define
    - backend
    - name
    - shape
    """
    def test_get_values_default(self):
        for varname in self.test_point.keys():
            expected = [self.expected[0][varname], self.expected[1][varname]]
            result = self.mtrace.get_values(varname)
            npt.assert_equal(result, expected)

    def test_get_values_burn_keyword(self):
        burn = 2
        for varname in self.test_point.keys():
            expected = [self.expected[0][varname][burn:],
                        self.expected[1][varname][burn:]]
            result = self.mtrace.get_values(varname, burn=burn)
            npt.assert_equal(result, expected)

    def test_len(self):
        self.assertEqual(len(self.mtrace), self.draws)

    def test_dtypes(self):
        for varname in self.test_point.keys():
            self.assertEqual(self.expected[0][varname].dtype,
                             self.mtrace.get_values(varname, chains=0).dtype)

    def test_get_values_thin_keyword(self):
        thin = 2
        for varname in self.test_point.keys():
            expected = [self.expected[0][varname][::thin],
                        self.expected[1][varname][::thin]]
            result = self.mtrace.get_values(varname, thin=thin)
            npt.assert_equal(result, expected)

    def test_get_point(self):
        idx = 2
        result = self.mtrace.point(idx)
        for varname in self.test_point.keys():
            expected = self.expected[1][varname][idx]
            npt.assert_equal(result[varname], expected)

    def test_get_slice(self):
        expected = []
        for chain in [0, 1]:
            expected.append({varname: self.expected[chain][varname][:2]
                             for varname in self.mtrace.varnames})
        result = self.mtrace[:2]
        for chain in [0, 1]:
            for varname in self.test_point.keys():
                npt.assert_equal(result.get_values(varname, chains=[chain]),
                                 expected[chain][varname])

    def test_get_values_one_chain(self):
        for varname in self.test_point.keys():
            expected = self.expected[0][varname]
            result = self.mtrace.get_values(varname, chains=[0])
            npt.assert_equal(result, expected)

    def test_get_values_chains_reversed(self):
        for varname in self.test_point.keys():
            expected = [self.expected[1][varname], self.expected[0][varname]]
            result = self.mtrace.get_values(varname, chains=[1, 0])
            npt.assert_equal(result, expected)

    def test_nchains(self):
        self.mtrace.nchains == 2

    def test_get_values_one_chain_int_arg(self):
        for varname in self.test_point.keys():
            npt.assert_equal(self.mtrace.get_values(varname, chains=[0]),
                             self.mtrace.get_values(varname, chains=0))

    def test_get_values_combine(self):
        for varname in self.test_point.keys():
            expected = np.concatenate([self.expected[chain][varname]
                                       for chain in [0, 1]])
            result = self.mtrace.get_values(varname, combine=True)
            npt.assert_equal(result, expected)

    def test_get_values_combine_burn_arg(self):
        burn = 2
        for varname in self.test_point.keys():
            expected = np.concatenate([self.expected[chain][varname][burn:]
                                       for chain in [0, 1]])
            result = self.mtrace.get_values(varname, combine=True, burn=burn)
            npt.assert_equal(result, expected)

    def test_get_values_combine_thin_arg(self):
        thin = 2
        for varname in self.test_point.keys():
            expected = np.concatenate([self.expected[chain][varname][::thin]
                                       for chain in [0, 1]])
            result = self.mtrace.get_values(varname, combine=True, thin=thin)
            npt.assert_equal(result, expected)


class SelectionNoSliceTestCase(SelectionTestCase):
    def test_get_slice(self):
        pass


class DumpLoadTestCase(ModelBackendSampledTestCase):
    """Test equality of a dumped and loaded trace with original.

    Children must define
    - backend
    - load_func
        Function to load dumped backend
    - name
    - shape
    """
    @classmethod
    def setUpClass(cls):
        super(DumpLoadTestCase, cls).setUpClass()
        try:
            with cls.model:
                cls.dumped = cls.load_func(cls.name)
        except:
            remove_file_or_directory(cls.name)
            raise

    @classmethod
    def tearDownClass(cls):
        remove_file_or_directory(cls.name)

    def test_nchains(self):
        self.assertEqual(self.mtrace.nchains, self.dumped.nchains)

    def test_varnames(self):
        trace_names = list(sorted(self.mtrace.varnames))
        dumped_names = list(sorted(self.dumped.varnames))
        self.assertEqual(trace_names, dumped_names)

    def test_values(self):
        trace = self.mtrace
        dumped = self.dumped
        for chain in trace.chains:
            for varname in self.test_point.keys():
                data = trace.get_values(varname, chains=[chain])
                dumped_data = dumped.get_values(varname, chains=[chain])
                npt.assert_equal(data, dumped_data)


class BackendEqualityTestCase(ModelBackendSampledTestCase):
    """Test equality of attirbutes from two backends.

    Children must define
    - backend0
    - backend1
    - name0
    - name1
    - shape
    """
    @classmethod
    def setUpClass(cls):
        cls.backend = cls.backend0
        cls.name = cls.name0
        super(BackendEqualityTestCase, cls).setUpClass()
        cls.mtrace0 = cls.mtrace

        cls.backend = cls.backend1
        cls.name = cls.name1
        super(BackendEqualityTestCase, cls).setUpClass()
        cls.mtrace1 = cls.mtrace

    @classmethod
    def tearDownClass(cls):
        for name in [cls.name0, cls.name1]:
            if name is not None:
                remove_file_or_directory(name)

    def test_chain_length(self):
        assert self.mtrace0.nchains == self.mtrace1.nchains
        assert len(self.mtrace0) == len(self.mtrace1)

    def test_dtype(self):
        for varname in self.test_point.keys():
            self.assertEqual(self.mtrace0.get_values(varname, chains=0).dtype,
                             self.mtrace1.get_values(varname, chains=0).dtype)

    def test_number_of_draws(self):
        for varname in self.test_point.keys():
            values0 = self.mtrace0.get_values(varname, squeeze=False)
            values1 = self.mtrace1.get_values(varname, squeeze=False)
            assert values0[0].shape[0] == self.draws
            assert values1[0].shape[0] == self.draws

    def test_get_item(self):
        for varname in self.test_point.keys():
            npt.assert_equal(self.mtrace0[varname], self.mtrace1[varname])

    def test_get_values(self):
        for varname in self.test_point.keys():
            for cf in [False, True]:
                npt.assert_equal(self.mtrace0.get_values(varname, combine=cf),
                                 self.mtrace1.get_values(varname, combine=cf))

    def test_get_values_no_squeeze(self):
        for varname in self.test_point.keys():
            npt.assert_equal(self.mtrace0.get_values(varname, combine=False,
                                                     squeeze=False),
                             self.mtrace1.get_values(varname, combine=False,
                                                     squeeze=False))

    def test_get_values_combine_and_no_squeeze(self):
        for varname in self.test_point.keys():
            npt.assert_equal(self.mtrace0.get_values(varname, combine=True,
                                                     squeeze=False),
                             self.mtrace1.get_values(varname, combine=True,
                                                     squeeze=False))

    def test_get_values_with_burn(self):
        for varname in self.test_point.keys():
            for cf in [False, True]:
                npt.assert_equal(self.mtrace0.get_values(varname, combine=cf,
                                                         burn=3),
                                 self.mtrace1.get_values(varname, combine=cf,
                                                         burn=3))
                ## Burn to one value.
                npt.assert_equal(self.mtrace0.get_values(varname, combine=cf,
                                                         burn=self.draws - 1),
                                 self.mtrace1.get_values(varname, combine=cf,
                                                         burn=self.draws - 1))

    def test_get_values_with_thin(self):
        for varname in self.test_point.keys():
            for cf in [False, True]:
                npt.assert_equal(self.mtrace0.get_values(varname, combine=cf,
                                                         thin=2),
                             self.mtrace1.get_values(varname, combine=cf,
                                                     thin=2))

    def test_get_values_with_burn_and_thin(self):
        for varname in self.test_point.keys():
            for cf in [False, True]:
                npt.assert_equal(self.mtrace0.get_values(varname, combine=cf,
                                                         burn=2, thin=2),
                                 self.mtrace1.get_values(varname, combine=cf,
                                                         burn=2, thin=2))

    def test_get_values_with_chains_arg(self):
        for varname in self.test_point.keys():
            for cf in [False, True]:
                npt.assert_equal(self.mtrace0.get_values(varname, chains=[0]),
                                 self.mtrace1.get_values(varname, chains=[0]))

    def test_get_point(self):
        npoint, spoint = self.mtrace0[4], self.mtrace1[4]
        for varname in self.test_point.keys():
            npt.assert_equal(npoint[varname], spoint[varname])

    def test_point_with_chain_arg(self):
        npoint = self.mtrace0.point(4, chain=0)
        spoint = self.mtrace1.point(4, chain=0)
        for varname in self.test_point.keys():
            npt.assert_equal(npoint[varname], spoint[varname])


def remove_file_or_directory(name):
    try:
        os.remove(name)
    except OSError:
        shutil.rmtree(name, ignore_errors=True)
