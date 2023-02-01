#   Copyright 2023 The PyMC Developers
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
import numpy.testing as npt

import pymc as pm


class TestZeroMean:
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            zero_mean = pm.gp.mean.Zero()
        M = zero_mean(X).eval()
        assert np.all(M == 0)
        assert M.shape == (10,)


class TestConstantMean:
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            const_mean = pm.gp.mean.Constant(6)
        M = const_mean(X).eval()
        assert np.all(M == 6)
        assert M.shape == (10,)


class TestLinearMean:
    def test_value(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            linear_mean = pm.gp.mean.Linear(2, 0.5)
        M = linear_mean(X).eval()
        npt.assert_allclose(M[1], 0.7222, atol=1e-3)
        assert M.shape == (10,)


class TestAddProdMean:
    def test_add(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            mean1 = pm.gp.mean.Linear(coeffs=2, intercept=0.5)
            mean2 = pm.gp.mean.Constant(2)
            mean = mean1 + mean2 + mean2
        M = mean(X).eval()
        npt.assert_allclose(M[1], 0.7222 + 2 + 2, atol=1e-3)

    def test_prod(self):
        X = np.linspace(0, 1, 10)[:, None]
        with pm.Model() as model:
            mean1 = pm.gp.mean.Linear(coeffs=2, intercept=0.5)
            mean2 = pm.gp.mean.Constant(2)
            mean = mean1 * mean2 * mean2
        M = mean(X).eval()
        npt.assert_allclose(M[1], 0.7222 * 2 * 2, atol=1e-3)

    def test_add_multid(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        A = np.array([1, 2, 3])
        b = 10
        with pm.Model() as model:
            mean1 = pm.gp.mean.Linear(coeffs=A, intercept=b)
            mean2 = pm.gp.mean.Constant(2)
            mean = mean1 + mean2 + mean2
        M = mean(X).eval()
        npt.assert_allclose(M[1], 10.8965 + 2 + 2, atol=1e-3)

    def test_prod_multid(self):
        X = np.linspace(0, 1, 30).reshape(10, 3)
        A = np.array([1, 2, 3])
        b = 10
        with pm.Model() as model:
            mean1 = pm.gp.mean.Linear(coeffs=A, intercept=b)
            mean2 = pm.gp.mean.Constant(2)
            mean = mean1 * mean2 * mean2
        M = mean(X).eval()
        npt.assert_allclose(M[1], 10.8965 * 2 * 2, atol=1e-3)
