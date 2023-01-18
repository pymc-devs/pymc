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

from functools import reduce
from operator import add

import numpy as np
import numpy.testing as npt
import pytest

import pymc as pm

from pymc.math import cartesian


class TestSigmaParams:
    def setup_method(self):
        """Common setup."""
        self.x = np.linspace(-5, 5, 30)[:, None]
        self.xu = np.linspace(-5, 5, 10)[:, None]
        self.y = np.random.normal(0.25 * self.x, 0.1)

        with pm.Model() as self.model:
            cov_func = pm.gp.cov.Linear(1, c=0.0)
            c = pm.Normal("c", mu=20.0, sigma=100.0)
            mean_func = pm.gp.mean.Constant(c)
            self.gp = self.gp_implementation(mean_func=mean_func, cov_func=cov_func)
            self.sigma = pm.HalfNormal("sigma", sigma=100)


class TestMarginalSigmaParams(TestSigmaParams):
    R"""Tests for the deprecation warnings and raising ValueError."""

    gp_implementation = pm.gp.Marginal

    def test_catch_warnings(self):
        """Warning from using the old noise parameter."""
        with self.model:
            with pytest.warns(FutureWarning):
                self.gp.marginal_likelihood("lik_noise", X=self.x, y=self.y, noise=self.sigma)

            with pytest.warns(FutureWarning):
                self.gp.conditional(
                    "cond_noise",
                    Xnew=self.x,
                    given={
                        "noise": self.sigma,
                    },
                )

    def test_raise_value_error(self):
        """Either both or neither parameter is specified."""
        with self.model:
            with pytest.raises(ValueError):
                self.gp.marginal_likelihood(
                    "like_both", X=self.x, y=self.y, noise=self.sigma, sigma=self.sigma
                )

            with pytest.raises(ValueError):
                self.gp.marginal_likelihood("like_neither", X=self.x, y=self.y)


class TestMarginalApproxSigmaParams(TestSigmaParams):
    R"""Tests for the deprecation warnings and raising ValueError"""

    gp_implementation = pm.gp.MarginalApprox

    @pytest.mark.xfail(reason="Possible shape problem, see #6366")
    def test_catch_warnings(self):
        """Warning from using the old noise parameter."""
        with self.model:
            with pytest.warns(FutureWarning):
                self.gp.marginal_likelihood(
                    "lik_noise", X=self.x, Xu=self.xu, y=self.y, noise=self.sigma
                )

    def test_raise_value_error(self):
        """Either both or neither parameter is specified."""
        with self.model:
            with pytest.raises(ValueError):
                self.gp.marginal_likelihood(
                    "like_both", X=self.x, Xu=self.xu, y=self.y, noise=self.sigma, sigma=self.sigma
                )

            with pytest.raises(ValueError):
                self.gp.marginal_likelihood("like_neither", X=self.x, Xu=self.xu, y=self.y)


class TestMarginalVsMarginalApprox:
    R"""
    Compare test fits of models Marginal and MarginalApprox.
    """

    def setup_method(self):
        self.sigma = 0.1
        self.x = np.linspace(-5, 5, 30)
        self.y = np.random.normal(0.25 * self.x, self.sigma)
        with pm.Model() as model:
            cov_func = pm.gp.cov.Linear(1, c=0.0)
            c = pm.Normal("c", mu=20.0, sigma=100.0)  # far from true value
            mean_func = pm.gp.mean.Constant(c)
            self.gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
            sigma = pm.HalfNormal("sigma", sigma=100)
            self.gp.marginal_likelihood("lik", self.x[:, None], self.y, sigma)
            self.map_full = pm.find_MAP(method="bfgs")  # bfgs seems to work much better than lbfgsb

        self.x_new = np.linspace(-6, 6, 20)

        # Include additive Gaussian noise, return diagonal of predicted covariance matrix
        with model:
            self.pred_mu, self.pred_var = self.gp.predict(
                self.x_new[:, None], point=self.map_full, pred_noise=True, diag=True
            )

        # Dont include additive Gaussian noise, return full predicted covariance matrix
        with model:
            self.pred_mu, self.pred_covar = self.gp.predict(
                self.x_new[:, None], point=self.map_full, pred_noise=False, diag=False
            )

    @pytest.mark.parametrize("approx", ["FITC", "VFE", "DTC"])
    def test_fits_and_preds(self, approx):
        """Get MAP estimate for GP approximation, compare results and predictions to what's returned
        by an unapproximated GP.  The tolerances are fairly wide, but narrow relative to initial
        values of the unknown parameters.
        """

        with pm.Model() as model:
            cov_func = pm.gp.cov.Linear(1, c=0.0)
            c = pm.Normal("c", mu=20.0, sigma=100.0, initval=-500.0)
            mean_func = pm.gp.mean.Constant(c)
            gp = pm.gp.MarginalApprox(mean_func=mean_func, cov_func=cov_func, approx=approx)
            sigma = pm.HalfNormal("sigma", sigma=100, initval=50.0)
            gp.marginal_likelihood("lik", self.x[:, None], self.x[:, None], self.y, sigma)
            map_approx = pm.find_MAP(method="bfgs")

        # Check MAP gets approximately correct result
        npt.assert_allclose(self.map_full["c"], map_approx["c"], atol=0.01, rtol=0.1)
        npt.assert_allclose(self.map_full["sigma"], map_approx["sigma"], atol=0.01, rtol=0.1)

        # Check that predict (and conditional) work, include noise, with diagonal non-full pred var.
        with model:
            pred_mu_approx, pred_var_approx = gp.predict(
                self.x_new[:, None], point=map_approx, pred_noise=True, diag=True
            )
        npt.assert_allclose(self.pred_mu, pred_mu_approx, atol=0.0, rtol=0.1)
        npt.assert_allclose(self.pred_var, pred_var_approx, atol=0.0, rtol=0.1)

        # Check that predict (and conditional) work, no noise, full pred covariance.
        with model:
            pred_mu_approx, pred_var_approx = gp.predict(
                self.x_new[:, None], point=map_approx, pred_noise=True, diag=True
            )
        npt.assert_allclose(self.pred_mu, pred_mu_approx, atol=0.0, rtol=0.1)
        npt.assert_allclose(self.pred_var, pred_var_approx, atol=0.0, rtol=0.1)


class TestGPAdditive:
    def setup_method(self):
        self.X = np.random.randn(50, 3)
        self.y = np.random.randn(50)
        self.Xnew = np.random.randn(60, 3)
        self.noise = pm.gp.cov.WhiteNoise(0.1)
        self.covs = (
            pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]),
            pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]),
            pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]),
        )
        self.means = (pm.gp.mean.Constant(0.5), pm.gp.mean.Constant(0.5), pm.gp.mean.Constant(0.5))

    def testAdditiveMarginal(self):
        with pm.Model() as model1:
            gp1 = pm.gp.Marginal(mean_func=self.means[0], cov_func=self.covs[0])
            gp2 = pm.gp.Marginal(mean_func=self.means[1], cov_func=self.covs[1])
            gp3 = pm.gp.Marginal(mean_func=self.means[2], cov_func=self.covs[2])

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.marginal_likelihood("f", self.X, self.y, sigma=self.noise)
            model1_logp = model1.compile_logp()({})

        with pm.Model() as model2:
            gptot = pm.gp.Marginal(
                mean_func=reduce(add, self.means), cov_func=reduce(add, self.covs)
            )
            fsum = gptot.marginal_likelihood("f", self.X, self.y, sigma=self.noise)
            model2_logp = model2.compile_logp()({})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional(
                "fp1", self.Xnew, given={"X": self.X, "y": self.y, "sigma": self.noise, "gp": gpsum}
            )
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])
        logp1 = model1.compile_logp()({"fp1": fp})
        logp2 = model2.compile_logp()({"fp2": fp})
        npt.assert_allclose(logp1, logp2, atol=0, rtol=1e-2)

    @pytest.mark.parametrize("approx", ["FITC", "VFE", "DTC"])
    def testAdditiveMarginalApprox(self, approx):
        Xu = np.random.randn(10, 3)
        sigma = 0.1
        with pm.Model() as model1:
            gp1 = pm.gp.MarginalApprox(
                mean_func=self.means[0], cov_func=self.covs[0], approx=approx
            )
            gp2 = pm.gp.MarginalApprox(
                mean_func=self.means[1], cov_func=self.covs[1], approx=approx
            )
            gp3 = pm.gp.MarginalApprox(
                mean_func=self.means[2], cov_func=self.covs[2], approx=approx
            )

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.marginal_likelihood("f", self.X, Xu, self.y, sigma=sigma)
            model1_logp = model1.compile_logp()({})

        with pm.Model() as model2:
            gptot = pm.gp.MarginalApprox(
                mean_func=reduce(add, self.means), cov_func=reduce(add, self.covs), approx=approx
            )
            fsum = gptot.marginal_likelihood("f", self.X, Xu, self.y, sigma=sigma)
            model2_logp = model2.compile_logp()({})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional(
                "fp1",
                self.Xnew,
                given={"X": self.X, "Xu": Xu, "y": self.y, "sigma": sigma, "gp": gpsum},
            )
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])

        model1_logp = model1.compile_logp()({"fp1": fp})
        model2_logp = model2.compile_logp()({"fp2": fp})

        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

    def testAdditiveLatent(self):
        with pm.Model() as model1:
            gp1 = pm.gp.Latent(mean_func=self.means[0], cov_func=self.covs[0])
            gp2 = pm.gp.Latent(mean_func=self.means[1], cov_func=self.covs[1])
            gp3 = pm.gp.Latent(mean_func=self.means[2], cov_func=self.covs[2])

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.prior("fsum", self.X, reparameterize=False)
            model1_logp = model1.compile_logp()({"fsum": self.y})

        with pm.Model() as model2:
            gptot = pm.gp.Latent(mean_func=reduce(add, self.means), cov_func=reduce(add, self.covs))
            fsum = gptot.prior("fsum", self.X, reparameterize=False)
            model2_logp = model2.compile_logp()({"fsum": self.y})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional("fp1", self.Xnew, given={"X": self.X, "f": self.y, "gp": gpsum})
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])
        logp1 = model1.compile_logp()({"fsum": self.y, "fp1": fp})
        logp2 = model2.compile_logp()({"fsum": self.y, "fp2": fp})
        npt.assert_allclose(logp1, logp2, atol=0, rtol=1e-2)

    def testAdditiveSparseRaises(self):
        # cant add different approximations
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.MarginalApprox(cov_func=cov_func, approx="DTC")
            gp2 = pm.gp.MarginalApprox(cov_func=cov_func, approx="FITC")
            with pytest.raises(Exception) as e_info:
                gp1 + gp2

    def testAdditiveTypeRaises1(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.MarginalApprox(cov_func=cov_func, approx="DTC")
            gp2 = pm.gp.Marginal(cov_func=cov_func)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2

    def testAdditiveTypeRaises2(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.Latent(cov_func=cov_func)
            gp2 = pm.gp.Marginal(cov_func=cov_func)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2


class TestMarginalVsLatent:
    R"""
    Compare the logp of models Marginal, sigma=0 and Latent.
    """

    def setup_method(self):
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        Xnew = np.random.randn(30, 3)
        pnew = np.random.randn(30)
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
            f = gp.marginal_likelihood("f", X, y, sigma=0.0)
            p = gp.conditional("p", Xnew)
        self.logp = model.compile_logp()({"p": pnew})
        self.X = X
        self.Xnew = Xnew
        self.y = y
        self.pnew = pnew

    def testLatent1(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
            f = gp.prior("f", self.X, reparameterize=False)
            p = gp.conditional("p", self.Xnew)
        assert tuple(f.shape.eval()) == (self.X.shape[0],)
        assert tuple(p.shape.eval()) == (self.Xnew.shape[0],)
        latent_logp = model.compile_logp()({"f": self.y, "p": self.pnew})
        npt.assert_allclose(latent_logp, self.logp, atol=0, rtol=1e-2)

    def testLatent2(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov_func)
            f = gp.prior("f", self.X, reparameterize=True)
            p = gp.conditional("p", self.Xnew)
        assert tuple(f.shape.eval()) == (self.X.shape[0],)
        assert tuple(p.shape.eval()) == (self.Xnew.shape[0],)
        chol = np.linalg.cholesky(cov_func(self.X).eval())
        y_rotated = np.linalg.solve(chol, self.y - 0.5)
        latent_logp = model.compile_logp()({"f_rotated_": y_rotated, "p": self.pnew})
        npt.assert_allclose(latent_logp, self.logp, atol=5)


class TestTP:
    R"""
    Compare TP with high degrees of freedom to GP
    """

    def setup_method(self):
        rng = np.random.default_rng(20221125)
        X = rng.standard_normal(size=(20, 3))
        y = rng.standard_normal(size=(20,))
        Xnew = rng.standard_normal(size=(30, 3))
        pnew = rng.standard_normal(size=(30,))

        with pm.Model() as model1:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp = pm.gp.Latent(cov_func=cov_func)
            f = gp.prior("f", X, reparameterize=False)
            p = gp.conditional("p", Xnew)
        self.gp_latent_logp = model1.compile_logp()({"f": y, "p": pnew})
        self.X = X
        self.y = y
        self.Xnew = Xnew
        self.pnew = pnew
        self.nu = 10000

    def testTPvsLatent(self):
        with pm.Model() as model:
            scale_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            tp = pm.gp.TP(scale_func=scale_func, nu=self.nu)
            f = tp.prior("f", self.X, reparameterize=False)
            p = tp.conditional("p", self.Xnew)
        assert tuple(f.shape.eval()) == (self.X.shape[0],)
        assert tuple(p.shape.eval()) == (self.Xnew.shape[0],)
        tp_logp = model.compile_logp()({"f": self.y, "p": self.pnew})
        npt.assert_allclose(self.gp_latent_logp, tp_logp, atol=0, rtol=1e-2)

    def testTPvsLatentReparameterized(self):
        with pm.Model() as model:
            scale_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            tp = pm.gp.TP(scale_func=scale_func, nu=self.nu)
            f = tp.prior("f", self.X, reparameterize=True)
            p = tp.conditional("p", self.Xnew)
        assert tuple(f.shape.eval()) == (self.X.shape[0],)
        assert tuple(p.shape.eval()) == (self.Xnew.shape[0],)
        chol = np.linalg.cholesky(scale_func(self.X).eval())
        f_rotated = np.linalg.solve(chol, self.y)
        tp_logp = model.compile_logp()({"f_rotated_": f_rotated, "p": self.pnew})
        npt.assert_allclose(self.gp_latent_logp, tp_logp, atol=0, rtol=1e-2)

    def testAdditiveTPRaises(self):
        with pm.Model() as model:
            scale_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.TP(scale_func=scale_func, nu=10)
            gp2 = pm.gp.TP(scale_func=scale_func, nu=10)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2


class TestLatentKron:
    """
    Compare gp.LatentKron to gp.Latent, both with Gaussian noise.
    """

    def setup_method(self):
        rng = np.random.default_rng(20221125)
        self.Xs = [
            np.linspace(0, 1, 7)[:, None],
            np.linspace(0, 1, 5)[:, None],
            np.linspace(0, 1, 6)[:, None],
        ]
        self.X = cartesian(*self.Xs)
        self.N = np.prod([len(X) for X in self.Xs])
        self.y = np.random.randn(self.N) * 0.1
        self.Xnews = (
            rng.standard_normal(size=(5, 1)),
            rng.standard_normal(size=(5, 1)),
            rng.standard_normal(size=(5, 1)),
        )
        self.Xnew = np.concatenate(self.Xnews, axis=1)
        self.pnew = rng.standard_normal(size=(len(self.Xnew),))
        ls = 0.2
        with pm.Model() as latent_model:
            self.cov_funcs = (
                pm.gp.cov.ExpQuad(1, ls),
                pm.gp.cov.ExpQuad(1, ls),
                pm.gp.cov.ExpQuad(1, ls),
            )
            cov_func = pm.gp.cov.Kron(self.cov_funcs)
            self.mean = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Latent(mean_func=self.mean, cov_func=cov_func)
            f = gp.prior("f", self.X)
            p = gp.conditional("p", self.Xnew)
        chol = np.linalg.cholesky(cov_func(self.X).eval())
        self.y_rotated = np.linalg.solve(chol, self.y - 0.5)
        self.logp = latent_model.compile_logp()({"f_rotated_": self.y_rotated, "p": self.pnew})

    def testLatentKronvsLatent(self):
        with pm.Model() as kron_model:
            kron_gp = pm.gp.LatentKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
            f = kron_gp.prior("f", self.Xs)
            p = kron_gp.conditional("p", self.Xnew)
        assert tuple(f.shape.eval()) == (self.X.shape[0],)
        assert tuple(p.shape.eval()) == (self.Xnew.shape[0],)
        kronlatent_logp = kron_model.compile_logp()({"f_rotated_": self.y_rotated, "p": self.pnew})
        npt.assert_allclose(kronlatent_logp, self.logp, atol=0, rtol=1e-3)

    def testLatentKronRaisesAdditive(self):
        with pm.Model() as kron_model:
            gp1 = pm.gp.LatentKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
            gp2 = pm.gp.LatentKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
        with pytest.raises(TypeError):
            gp1 + gp2

    def testLatentKronRaisesSizes(self):
        with pm.Model() as kron_model:
            gp = pm.gp.LatentKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
        with pytest.raises(ValueError):
            gp.prior("f", Xs=[np.linspace(0, 1, 7)[:, None], np.linspace(0, 1, 5)[:, None]])


class TestMarginalKron:
    """
    Compare gp.MarginalKron to gp.Marginal.
    """

    def setup_method(self):
        self.Xs = [
            np.linspace(0, 1, 7)[:, None],
            np.linspace(0, 1, 5)[:, None],
            np.linspace(0, 1, 6)[:, None],
        ]
        self.X = cartesian(*self.Xs)
        self.N = np.prod([len(X) for X in self.Xs])
        self.y = np.random.randn(self.N) * 0.1
        self.Xnews = (np.random.randn(5, 1), np.random.randn(5, 1), np.random.randn(5, 1))
        self.Xnew = np.concatenate(self.Xnews, axis=1)
        self.sigma = 0.2
        self.pnew = np.random.randn(len(self.Xnew))

        ls = 0.2
        with pm.Model() as model:
            self.cov_funcs = [
                pm.gp.cov.ExpQuad(1, ls),
                pm.gp.cov.ExpQuad(1, ls),
                pm.gp.cov.ExpQuad(1, ls),
            ]
            cov_func = pm.gp.cov.Kron(self.cov_funcs)
            self.mean = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Marginal(mean_func=self.mean, cov_func=cov_func)
            f = gp.marginal_likelihood("f", self.X, self.y, sigma=self.sigma)
            p = gp.conditional("p", self.Xnew)
            self.mu, self.cov = gp.predict(self.Xnew)
        self.logp = model.compile_logp()({"p": self.pnew})

    def testMarginalKronvsMarginalpredict(self):
        with pm.Model() as kron_model:
            kron_gp = pm.gp.MarginalKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
            f = kron_gp.marginal_likelihood("f", self.Xs, self.y, sigma=self.sigma)
            p = kron_gp.conditional("p", self.Xnew)
            mu, cov = kron_gp.predict(self.Xnew)
        assert tuple(f.shape.eval()) == (self.X.shape[0],)
        assert tuple(p.shape.eval()) == (self.Xnew.shape[0],)
        npt.assert_allclose(mu, self.mu, atol=1e-5, rtol=1e-2)
        npt.assert_allclose(cov, self.cov, atol=1e-5, rtol=1e-2)
        with kron_model:
            _, var = kron_gp.predict(self.Xnew, diag=True)
        npt.assert_allclose(np.diag(cov), var, atol=1e-5, rtol=1e-2)

    def testMarginalKronvsMarginal(self):
        with pm.Model() as kron_model:
            kron_gp = pm.gp.MarginalKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
            f = kron_gp.marginal_likelihood("f", self.Xs, self.y, sigma=self.sigma)
            p = kron_gp.conditional("p", self.Xnew)
        kron_logp = kron_model.compile_logp()({"p": self.pnew})
        npt.assert_allclose(kron_logp, self.logp, atol=0, rtol=1e-2)

    def testMarginalKronRaises(self):
        with pm.Model() as kron_model:
            gp1 = pm.gp.MarginalKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
            gp2 = pm.gp.MarginalKron(mean_func=self.mean, cov_funcs=self.cov_funcs)
        with pytest.raises(TypeError):
            gp1 + gp2


class TestHSGP:
    @pytest.fixture
    def rng(self):
        return np.random.RandomState(10)

    @pytest.fixture
    def data(self, rng):
        X = rng.randn(100, 1)
        X1 = rng.randn(52, 1)
        return X, X1

    @pytest.fixture
    def X(self, data):
        return data[0]

    @pytest.fixture
    def X1(self, data):
        return data[1]

    @pytest.fixture
    def model(self):
        return pm.Model()

    @pytest.fixture
    def cov_func(self):
        return pm.gp.cov.ExpQuad(1, ls=0.1)

    @pytest.fixture
    def gp(self, cov_func):
        gp = pm.gp.HSGP(m=[500], c=4.0, cov_func=cov_func)
        return gp

    def test_shapes(self, model, gp, X):
        with model:
            ka = gp.approx_K(X)
            kf = gp.cov_func(X)
        # NOTE: relative difference is still high
        np.testing.assert_allclose(ka.eval(), kf.eval(), atol=1e-10)

    def test_prior(self, model, gp, X):
        # TODO: improve mathematical side of tests
        # So far I just check interfaces are the same for latent and HSGP
        with model:
            f1 = gp.prior("f1", X)
            assert pm.draw(f1).shape == (X.shape[0],)
            assert ~np.isnan(pm.draw(f1)).any()

    def test_conditional(self, model, gp, X, X1):
        # TODO: improve mathematical side of tests
        # So far I just check interfaces are the same for latent and HSGP
        with model:
            with pytest.raises(ValueError, match="Prior is not set"):
                gp.conditional("f1", X1)
            gp.prior("f1", X)
            gp.conditional("f2", X1)

    def test_parametrization_m(self):
        cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
        with pytest.raises(
            ValueError, match="must be sequences, with one element per active dimension"
        ):
            pm.gp.HSGP(m=500, c=4.0, cov_func=cov_func)

        with pytest.raises(
            ValueError, match="must be sequences, with one element per active dimension"
        ):
            pm.gp.HSGP(m=[500, 12], c=4.0, cov_func=cov_func)

    def test_parametrization_L(self):
        cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
        with pytest.raises(
            ValueError, match="must be sequences, with one element per active dimension"
        ):
            pm.gp.HSGP(m=[500], L=[12, 12], cov_func=cov_func)

    @pytest.mark.parametrize("drop_first", [True, False])
    def test_parametrization_drop_first(self, model, cov_func, X, drop_first):
        with model:
            gp = pm.gp.HSGP(m=[500], c=4.0, cov_func=cov_func, drop_first=drop_first)
            gp.prior("f1", X)
            assert model.f1_coeffs_.type.shape == (500 - drop_first,)
