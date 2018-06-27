#  pylint:disable=unused-variable
from functools import reduce
from ..math import cartesian
from operator import add
import pymc3 as pm
import numpy as np
import numpy.testing as npt
import pytest

np.random.seed(101)


class TestMarginalVsLatent(object):
    R"""
    Compare the logp of models Marginal, noise=0 and Latent.
    """
    def setup_method(self):
        X = np.random.randn(50,3)
        y = np.random.randn(50)*0.01
        Xnew = np.random.randn(60, 3)
        pnew = np.random.randn(60)*0.01
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Marginal(mean_func, cov_func)
            f = gp.marginal_likelihood("f", X, y, noise=0.0, is_observed=False, observed=y)
            p = gp.conditional("p", Xnew)
        self.logp = model.logp({"p": pnew})
        self.X = X
        self.Xnew = Xnew
        self.y = y
        self.pnew = pnew

    def testLatent1(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Latent(mean_func, cov_func)
            f = gp.prior("f", self.X, reparameterize=False)
            p = gp.conditional("p", self.Xnew)
        latent_logp = model.logp({"f": self.y, "p": self.pnew})
        npt.assert_allclose(latent_logp, self.logp, atol=0, rtol=1e-2)

    def testLatent2(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Latent(mean_func, cov_func)
            f = gp.prior("f", self.X, reparameterize=True)
            p = gp.conditional("p", self.Xnew)
        chol = np.linalg.cholesky(cov_func(self.X).eval())
        y_rotated = np.linalg.solve(chol, self.y - 0.5)
        latent_logp = model.logp({"f_rotated_": y_rotated, "p": self.pnew})
        npt.assert_allclose(latent_logp, self.logp, atol=5)


class TestMarginalVsMarginalSparse(object):
    R"""
    Compare logp of models Marginal and MarginalSparse.
    Should be nearly equal when inducing points are same as inputs.
    """
    def setup_method(self):
        X = np.random.randn(50,3)
        y = np.random.randn(50)*0.01
        Xnew = np.random.randn(60, 3)
        pnew = np.random.randn(60)*0.01
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Marginal(mean_func, cov_func)
            sigma = 0.1
            f = gp.marginal_likelihood("f", X, y, noise=sigma)
            p = gp.conditional("p", Xnew)
        self.logp = model.logp({"p": pnew})
        self.X = X
        self.Xnew = Xnew
        self.y = y
        self.sigma = sigma
        self.pnew = pnew
        self.gp = gp

    @pytest.mark.parametrize('approx', ['FITC', 'VFE', 'DTC'])
    def testApproximations(self, approx):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.MarginalSparse(mean_func, cov_func, approx=approx)
            f = gp.marginal_likelihood("f", self.X, self.X, self.y, self.sigma)
            p = gp.conditional("p", self.Xnew)
        approx_logp = model.logp({"f": self.y, "p": self.pnew})
        npt.assert_allclose(approx_logp, self.logp, atol=0, rtol=1e-2)

    @pytest.mark.parametrize('approx', ['FITC', 'VFE', 'DTC'])
    def testPredictVar(self, approx):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.MarginalSparse(mean_func, cov_func, approx=approx)
            f = gp.marginal_likelihood("f", self.X, self.X, self.y, self.sigma)
        mu1, var1 = self.gp.predict(self.Xnew, diag=True)
        mu2, var2 = gp.predict(self.Xnew, diag=True)
        npt.assert_allclose(mu1, mu2, atol=0, rtol=1e-3)
        npt.assert_allclose(var1, var2, atol=0, rtol=1e-3)

    def testPredictCov(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            mean_func = pm.gp.mean.Constant(0.5)
            gp = pm.gp.MarginalSparse(mean_func, cov_func, approx="DTC")
            f = gp.marginal_likelihood("f", self.X, self.X, self.y, self.sigma, is_observed=False)
        mu1, cov1 = self.gp.predict(self.Xnew, pred_noise=True)
        mu2, cov2 = gp.predict(self.Xnew, pred_noise=True)
        npt.assert_allclose(mu1, mu2, atol=0, rtol=1e-3)
        npt.assert_allclose(cov1, cov2, atol=0, rtol=1e-3)


class TestGPAdditive(object):
    def setup_method(self):
        self.X = np.random.randn(50,3)
        self.y = np.random.randn(50)*0.01
        self.Xnew = np.random.randn(60, 3)
        self.noise = pm.gp.cov.WhiteNoise(0.1)
        self.covs = (pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]),
                     pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]),
                     pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3]))
        self.means = (pm.gp.mean.Constant(0.5),
                      pm.gp.mean.Constant(0.5),
                      pm.gp.mean.Constant(0.5))

    def testAdditiveMarginal(self):
        with pm.Model() as model1:
            gp1 = pm.gp.Marginal(self.means[0], self.covs[0])
            gp2 = pm.gp.Marginal(self.means[1], self.covs[1])
            gp3 = pm.gp.Marginal(self.means[2], self.covs[2])

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.marginal_likelihood("f", self.X, self.y, noise=self.noise)
            model1_logp = model1.logp({"fsum": self.y})

        with pm.Model() as model2:
            gptot = pm.gp.Marginal(reduce(add, self.means), reduce(add, self.covs))
            fsum = gptot.marginal_likelihood("f", self.X, self.y, noise=self.noise)
            model2_logp = model2.logp({"fsum": self.y})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional("fp1", self.Xnew, given={"X": self.X, "y": self.y,
                                                            "noise": self.noise, "gp": gpsum})
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])
        npt.assert_allclose(fp1.logp({"fp1": fp}), fp2.logp({"fp2": fp}), atol=0, rtol=1e-2)

    @pytest.mark.parametrize('approx', ['FITC', 'VFE', 'DTC'])
    def testAdditiveMarginalSparse(self, approx):
        Xu = np.random.randn(10, 3)
        sigma = 0.1
        with pm.Model() as model1:
            gp1 = pm.gp.MarginalSparse(self.means[0], self.covs[0], approx=approx)
            gp2 = pm.gp.MarginalSparse(self.means[1], self.covs[1], approx=approx)
            gp3 = pm.gp.MarginalSparse(self.means[2], self.covs[2], approx=approx)

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.marginal_likelihood("f", self.X, Xu, self.y, noise=sigma)
            model1_logp = model1.logp({"fsum": self.y})

        with pm.Model() as model2:
            gptot = pm.gp.MarginalSparse(reduce(add, self.means), reduce(add, self.covs), approx=approx)
            fsum = gptot.marginal_likelihood("f", self.X, Xu, self.y, noise=sigma)
            model2_logp = model2.logp({"fsum": self.y})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional("fp1", self.Xnew, given={"X": self.X, "Xu": Xu, "y": self.y,
                                                            "sigma": sigma, "gp": gpsum})
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])
        npt.assert_allclose(fp1.logp({"fp1": fp}), fp2.logp({"fp2": fp}), atol=0, rtol=1e-2)

    def testAdditiveLatent(self):
        with pm.Model() as model1:
            gp1 = pm.gp.Latent(self.means[0], self.covs[0])
            gp2 = pm.gp.Latent(self.means[1], self.covs[1])
            gp3 = pm.gp.Latent(self.means[2], self.covs[2])

            gpsum = gp1 + gp2 + gp3
            fsum = gpsum.prior("fsum", self.X, reparameterize=False)
            model1_logp = model1.logp({"fsum": self.y})

        with pm.Model() as model2:
            gptot = pm.gp.Latent(reduce(add, self.means), reduce(add, self.covs))
            fsum = gptot.prior("fsum", self.X, reparameterize=False)
            model2_logp = model2.logp({"fsum": self.y})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gpsum.conditional("fp1", self.Xnew, given={"X": self.X, "f": self.y, "gp": gpsum})
        with model2:
            fp2 = gptot.conditional("fp2", self.Xnew)

        fp = np.random.randn(self.Xnew.shape[0])
        npt.assert_allclose(fp1.logp({"fsum": self.y, "fp1": fp}),
                            fp2.logp({"fsum": self.y, "fp2": fp}), atol=0, rtol=1e-2)

    def testAdditiveSparseRaises(self):
        # cant add different approximations
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.MarginalSparse(cov_func=cov_func, approx="DTC")
            gp2 = pm.gp.MarginalSparse(cov_func=cov_func, approx="FITC")
            with pytest.raises(Exception) as e_info:
                gp1 + gp2

    def testAdditiveTypeRaises1(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.MarginalSparse(cov_func=cov_func, approx="DTC")
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


class TestTP(object):
    R"""
    Compare TP with high degress of freedom to GP
    """
    def setup_method(self):
        X = np.random.randn(20,3)
        y = np.random.randn(20)*0.01
        Xnew = np.random.randn(50, 3)
        pnew = np.random.randn(50)*0.01
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp = pm.gp.Latent(cov_func=cov_func)
            f = gp.prior("f", X, reparameterize=False)
            p = gp.conditional("p", Xnew)
        self.X = X
        self.y = y
        self.Xnew = Xnew
        self.pnew = pnew
        self.latent_logp = model.logp({"f": y, "p": pnew})
        self.plogp = p.logp({"f": y, "p": pnew})

    def testTPvsLatent(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            tp = pm.gp.TP(cov_func=cov_func, nu=10000)
            f = tp.prior("f", self.X, reparameterize=False)
            p = tp.conditional("p", self.Xnew)
        tp_logp = model.logp({"f": self.y, "p": self.pnew})
        npt.assert_allclose(self.latent_logp, tp_logp, atol=0, rtol=1e-2)

    def testTPvsLatentReparameterized(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            tp = pm.gp.TP(cov_func=cov_func, nu=10000)
            f = tp.prior("f", self.X, reparameterize=True)
            p = tp.conditional("p", self.Xnew)
        chol = np.linalg.cholesky(cov_func(self.X).eval())
        y_rotated = np.linalg.solve(chol, self.y)
        # testing full model logp unreliable due to introduction of chi2__log__
        plogp = p.logp({"f_rotated_": y_rotated, "p": self.pnew, "chi2__log__": np.log(1e20)})
        npt.assert_allclose(self.plogp, plogp, atol=0, rtol=1e-2)

    def testAdditiveTPRaises(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.TP(cov_func=cov_func, nu=10)
            gp2 = pm.gp.TP(cov_func=cov_func, nu=10)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2


class TestMarginalKron(object):
    def setup_method(self):
        self.Xs = [np.linspace(0, 1, 7)[:, None],
                   np.linspace(0, 1, 5)[:, None],
                   np.linspace(0, 1, 6)[:, None]]
        self.X = cartesian(*self.Xs)
        self.N = np.prod([len(X) for X in self.Xs])
        self.y = np.random.randn(self.N) * 0.1
        self.Xnews = [np.random.randn(5, 1),
                      np.random.randn(5, 1),
                      np.random.randn(5, 1)]
        self.Xnew = np.concatenate(tuple(self.Xnews), axis=1)
        self.sigma = 0.2
        self.pnew = np.random.randn(len(self.Xnew))*0.01
        ls = 0.2
        with pm.Model() as model:
            self.cov_funcs = [pm.gp.cov.ExpQuad(1, ls),
                              pm.gp.cov.ExpQuad(1, ls),
                              pm.gp.cov.ExpQuad(1, ls)]
            cov_func = pm.gp.cov.Kron(self.cov_funcs)
            self.mean = pm.gp.mean.Constant(0.5)
            gp = pm.gp.Marginal(mean_func=self.mean, cov_func=cov_func)
            f = gp.marginal_likelihood("f", self.X, self.y, noise=self.sigma)
            p = gp.conditional("p", self.Xnew)
            self.mu, self.cov = gp.predict(self.Xnew)
        self.logp = model.logp({"p": self.pnew})

    def testMarginalKronvsMarginalpredict(self):
        with pm.Model() as kron_model:
            kron_gp = pm.gp.MarginalKron(mean_func=self.mean,
                                         cov_funcs=self.cov_funcs)
            f = kron_gp.marginal_likelihood('f', self.Xs, self.y,
                                            sigma=self.sigma, shape=self.N)
            p = kron_gp.conditional('p', self.Xnew)
            mu, cov = kron_gp.predict(self.Xnew)
        npt.assert_allclose(mu, self.mu, atol=0, rtol=1e-2)
        npt.assert_allclose(cov, self.cov, atol=0, rtol=1e-2)

    def testMarginalKronvsMarginal(self):
        with pm.Model() as kron_model:
            kron_gp = pm.gp.MarginalKron(mean_func=self.mean,
                                         cov_funcs=self.cov_funcs)
            f = kron_gp.marginal_likelihood('f', self.Xs, self.y,
                                            sigma=self.sigma, shape=self.N)
            p = kron_gp.conditional('p', self.Xnew)
        kron_logp = kron_model.logp({'p': self.pnew})
        npt.assert_allclose(kron_logp, self.logp, atol=0, rtol=1e-2)

    def testMarginalKronRaises(self):
        with pm.Model() as kron_model:
            gp1 = pm.gp.MarginalKron(mean_func=self.mean,
                                     cov_funcs=self.cov_funcs)
            gp2 = pm.gp.MarginalKron(mean_func=self.mean,
                                     cov_funcs=self.cov_funcs)
        with pytest.raises(TypeError):
            gp1 + gp2
