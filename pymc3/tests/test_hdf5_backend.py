import os
from pymc3.tests import backend_fixtures as bf
from pymc3.backends import ndarray, hdf5
import tempfile

DBNAME = os.path.join(tempfile.gettempdir(), 'test.h5')


class TestHDF50dSampling(bf.SamplingTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = ()


class TestHDF51dSampling(bf.SamplingTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = 2


class TestHDF52dSampling(bf.SamplingTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = (2, 3)


class TestHDF50dSelection(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = ()


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
