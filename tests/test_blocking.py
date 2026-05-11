#   Copyright 2024 - present The PyMC Developers
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

from pymc.blocking import DictToArrayBijection


class TestDictToArrayBijection:
    def test_map_basic(self):
        point = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}
        result = DictToArrayBijection.map(point)
        np.testing.assert_array_equal(result.data, [1.0, 2.0, 3.0])

    def test_map_empty(self):
        result = DictToArrayBijection.map({})
        assert result.data.shape == (0,)

    def test_map_preserves_shape_info(self):
        point = {"x": np.array([[1.0, 2.0], [3.0, 4.0]])}
        result = DictToArrayBijection.map(point)
        name, shape, size, dtype = result.point_map_info[0]
        assert name == "x"
        assert shape == (2, 2)
        assert size == 4

    def test_rmap_basic(self):
        point = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}
        raveled = DictToArrayBijection.map(point)
        result = DictToArrayBijection.rmap(raveled)
        np.testing.assert_array_equal(result["a"], point["a"])
        np.testing.assert_array_equal(result["b"], point["b"])

    def test_map_rmap_roundtrip(self):
        point = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([[4.0, 5.0]])}
        result = DictToArrayBijection.rmap(DictToArrayBijection.map(point))
        for k in point:
            np.testing.assert_array_equal(result[k], point[k])

    def test_rmap_with_start_point(self):
        point = {"a": np.array([1.0])}
        raveled = DictToArrayBijection.map(point)
        start = {"b": np.array([99.0])}
        result = DictToArrayBijection.rmap(raveled, start_point=start)
        assert "b" in result
        np.testing.assert_array_equal(result["a"], point["a"])

    def test_mapf(self):
        point = {"a": np.array([1.0, 2.0])}
        raveled = DictToArrayBijection.map(point)

        def f(d):
            return d["a"].sum()

        composed = DictToArrayBijection.mapf(f)
        assert composed(raveled) == 3.0
