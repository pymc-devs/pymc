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

import numpy.testing as npt

from pymc3.backends import ndarray
from pymc3.backends import tracetab as ttab
from pymc3.tests import backend_fixtures as bf


class TestTraceToDf(bf.ModelBackendSampledTestCase):
    backend = ndarray.NDArray
    name = "text-db"
    shape = (2, 3)

    def test_trace_to_dataframe(self):
        mtrace = self.mtrace
        df = ttab.trace_to_dataframe(mtrace)
        assert len(mtrace) * mtrace.nchains == df.shape[0]

        checked = False
        for varname in self.test_point.keys():
            vararr = mtrace.get_values(varname)
            # With `shape` above, only one variable has to have that
            # `shape`.
            if vararr.shape[1:] != self.shape:
                continue
            npt.assert_equal(vararr[:, 0, 0], df[varname + "__0_0"].values)
            npt.assert_equal(vararr[:, 1, 0], df[varname + "__1_0"].values)
            npt.assert_equal(vararr[:, 1, 2], df[varname + "__1_2"].values)
            checked = True
        assert checked

    def test_trace_to_dataframe_chain_arg(self):
        mtrace = self.mtrace
        df = ttab.trace_to_dataframe(mtrace, chains=0)
        assert len(mtrace) == df.shape[0]

        checked = False
        for varname in self.test_point.keys():
            vararr = mtrace.get_values(varname, chains=0)
            # With `shape` above, only one variable has to have that
            # `shape`.
            if vararr.shape[1:] != self.shape:
                continue
            npt.assert_equal(vararr[:, 0, 0], df[varname + "__0_0"].values)
            npt.assert_equal(vararr[:, 1, 0], df[varname + "__1_0"].values)
            npt.assert_equal(vararr[:, 1, 2], df[varname + "__1_2"].values)
            checked = True
        assert checked


def test_create_flat_names_0d():
    shape = ()
    result = ttab.create_flat_names("x", shape)
    expected = ["x"]
    assert result == expected
    assert ttab._create_shape(result) == shape


def test_create_flat_names_1d():
    shape = (2,)
    result = ttab.create_flat_names("x", shape)
    expected = ["x__0", "x__1"]
    assert result == expected
    assert ttab._create_shape(result) == shape


def test_create_flat_names_2d():
    shape = 2, 3
    result = ttab.create_flat_names("x", shape)
    expected = ["x__0_0", "x__0_1", "x__0_2", "x__1_0", "x__1_1", "x__1_2"]
    assert result == expected
    assert ttab._create_shape(result) == shape


def test_create_flat_names_3d():
    shape = 2, 3, 4
    assert ttab._create_shape(ttab.create_flat_names("x", shape)) == shape
