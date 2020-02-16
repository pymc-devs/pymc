#   Copyright 2020 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
    skip_test_get_slice_neg_step = True


class TestHDF50dSelectionStats1(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = ()
    sampler_vars = STATS1
    skip_test_get_slice_neg_step = True


class TestHDF50dSelectionStats2(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = ()
    sampler_vars = STATS2
    skip_test_get_slice_neg_step = True


class TestHDF51dSelection(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = 2
    skip_test_get_slice_neg_step = True


class TestHDF52dSelection(bf.SelectionTestCase):
    backend = hdf5.HDF5
    name = DBNAME
    shape = (2, 3)
    skip_test_get_slice_neg_step = True


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
