import numpy as np
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, hdf5
import os
import tempfile

STATS1 = [{
    'a': np.float64,
    'b': np.bool
}]

STATS2 = [{
    'a': np.float64
}, {
    'a': np.float64,
    'b': np.int64,
}]

DBNAME = os.path.join(tempfile.gettempdir(), 'test.h5')

class TestHDF50dSampling(bf.SamplingTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = ()


class TestHDF50dSamplingStats1(bf.SamplingTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    sampler_vars = STATS1
    shape = ()


class TestHDF50dSamplingStats2(bf.SamplingTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    sampler_vars = STATS2
    shape = ()


class TestHDF51dSampling(bf.SamplingTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = 2


class TestHDF52dSampling(bf.SamplingTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = (2, 3)


class TestHDF5Stats(bf.StatsTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = (2, 3)


class TestHDF50dSelection(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = ()


class TestHDF50dSelectionStats1(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = ()
    sampler_vars = STATS1


class TestHDF50dSelectionStats2(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = ()
    sampler_vars = STATS2


class TestHDF51dSelection(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = 2


class TestHDF52dSelection(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = (2, 3)


class TestHDF5DumpLoad(bf.DumpLoadTestCase):
    backend = hdf5.HDF5
    load_func = staticmethod(hdf5.load)
    name = DBNAME
    shape = (2, 3)


class TestNDArrayHDF5Equality(bf.BackendEqualityTestCase):
    backend0 = ndarray.NDArray
    name0 = None
    backend1 = hdf5.HDF5
    name1 = DBNAME
    shape = (2, 3)