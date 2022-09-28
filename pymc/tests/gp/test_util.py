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

import aesara.tensor as at
import numpy as np
import numpy.testing as npt
import pytest

import pymc as pm


class TestPlotGP:
    def test_plot_gp_dist(self):
        """Test that the plotting helper works with the stated input shapes."""
        import matplotlib.pyplot as plt

        X = 100
        S = 500
        fig, ax = plt.subplots()
        pm.gp.util.plot_gp_dist(
            ax, x=np.linspace(0, 50, X), samples=np.random.normal(np.arange(X), size=(S, X))
        )
        plt.close()
        pass

    def test_plot_gp_dist_warn_nan(self):
        """Test that the plotting helper works with the stated input shapes."""
        import matplotlib.pyplot as plt

        X = 100
        S = 500
        samples = np.random.normal(np.arange(X), size=(S, X))
        samples[15, 3] = np.nan
        fig, ax = plt.subplots()
        with pytest.warns(UserWarning):
            pm.gp.util.plot_gp_dist(ax, x=np.linspace(0, 50, X), samples=samples)
        plt.close()
        pass


class TestKmeansInducing:
    def setup_method(self):
        self.centers = (-5, 5)
        self.x = np.concatenate(
            (self.centers[0] + np.random.randn(500), self.centers[1] + np.random.randn(500))
        )

    def test_kmeans(self):
        X = self.x[:, None]
        Xu = pm.gp.util.kmeans_inducing_points(2, X).flatten()
        npt.assert_allclose(np.asarray(self.centers), np.sort(Xu), rtol=0.05)

        X = at.as_tensor_variable(self.x[:, None])
        Xu = pm.gp.util.kmeans_inducing_points(2, X).flatten()
        npt.assert_allclose(np.asarray(self.centers), np.sort(Xu), rtol=0.05)

    def test_kmeans_raises(self):
        with pytest.raises(TypeError):
            Xu = pm.gp.util.kmeans_inducing_points(2, "str is the wrong type").flatten()


class TestReplaceWithValues:
    def test_basic_replace(self):
        with pm.Model() as model:
            a = pm.Normal("a")
            b = pm.Normal("b", mu=a)
            c = a * b

        (c_val,) = pm.gp.util.replace_with_values(
            [c], replacements={"a": 2, "b": 3, "x": 100}, model=model
        )
        assert c_val == np.array(6.0)

    def test_replace_no_inputs_needed(self):
        with pm.Model() as model:
            a = at.as_tensor_variable(2.0)
            b = 1.0 + a
            c = a * b
            (c_val,) = pm.gp.util.replace_with_values([c], replacements={"x": 100})
        assert c_val == np.array(6.0)

    def test_missing_input(self):
        with pm.Model() as model:
            a = pm.Normal("a")
            b = pm.Normal("b", mu=a)
            c = a * b

        with pytest.raises(ValueError):
            (c_val,) = pm.gp.util.replace_with_values(
                [c], replacements={"a": 2, "x": 100}, model=model
            )
