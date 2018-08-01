#  pylint:disable=unused-variable
from functools import reduce, partial
from ..math import cartesian, kronecker
from ..distributions import draw_values
from .helpers import SeededTest
from operator import add
import pymc3 as pm
import theano
import theano.tensor as tt
import numpy as np
import scipy as sp
import scipy.linalg
import scipy.stats
import numpy.testing as npt
import pytest


def assert_allclose_corr(x1, x2, atol):
    corr = np.corrcoef(x1, x2)
    npt.assert_allclose(corr, np.ones((2,2)), atol=atol)
        
        
class TestMarginal(object):
    def setup_method(self):
        np.random.seed(100)
        n = 150
        idx = slice(10, 140, 2)
        self.Xs = np.linspace(-10, 10, n)[:,None]
        self.X = self.Xs[idx, :]
        self.true_ls, self.true_sigma, self.true_c = (3.0, 0.1, 1.0)
        self.cov = pm.gp.cov.ExpQuad(1, ls=self.true_ls)
        self.mean = pm.gp.mean.Constant(self.true_c)
        self.true_f = np.random.multivariate_normal(self.mean(self.Xs).eval(),
                                                    self.cov(self.Xs).eval(), 1).flatten()
        true_y = self.true_f + self.true_sigma * np.random.randn(n)
        self.obs = true_y[idx]

        with pm.Model() as model:
            ls = pm.Gamma("ls", alpha=1500, beta=500)
            sigma = pm.Gamma("sigma", alpha=50, beta=500)
            c = pm.Normal("c", mu=1.0, sd=0.05)
            gp = pm.gp.Marginal(mean_func=pm.gp.mean.Constant(c),
                                cov_func=pm.gp.cov.ExpQuad(1, ls=ls))
            y_ = gp.marginal_likelihood("y", self.X, self.obs, noise=sigma)
            self.map_point = pm.find_MAP(method="BFGS")
            
        self.model = model
        self.tr = tr
        self.gp = gp
        
        with self.model:
            fs = self.gp.conditional("fs", Xnew=self.Xs)
            self.ppc = pm.sample_ppc([self.map_point], 100, vars=[fs])
        
        self.f_mu = np.mean(self.ppc["fs"], 0)
        self.f_sd = np.std(self.ppc["fs"], 0)

    def test_priors(self):
        # posteriors must be close to true values (which had strong priors)
        npt.assert_allclose(self.map_point["ls"], self.true_ls, atol=0.1)
        npt.assert_allclose(self.map_point["sigma"], self.true_sigma, atol=0.01)
        npt.assert_allclose(self.map_point["c"], self.true_c, atol=0.1)

    def test_conditionals(self):
        test_idx = slice(10, 140)
        assert_allclose_norm(self.f_mu[test_idx], self.true_f[test_idx], maxnorm=2.0)
        npt.assert_array_less(self.true_f, self.f_mu + 3*self.f_sd)
        npt.assert_array_less(self.f_mu - 3*self.f_sd, self.true_f)

    #@pytest.mark.parametrize("diag", [True, False])
    #@pytest.mark.parametrize("pred_noise", [True, False])
    #def test_predictions(self, diag, pred_noise):
    #    if diag:
    #        mu, var = self.gp.predict(self.Xs, point=self.map_point, diag=diag, pred_noise=pred_noise)
    #        npt.assert_allclose(np.sqrt(var), self.f_sd, atol=0.1)
    #    else:
    #        mu, cov = self.gp.predict(self.Xs, point=self.map_point, diag=diag, pred_noise=pred_noise)
    #        npt.assert_allclose(np.sqrt(np.diag(cov)), self.f_sd, atol=0.1)
    #    npt.assert_allclose(mu, self.f_mu, atol=0.1)
    
class TestAdditive(SeededTest):
    def test_marginal(self):
        n = 150
        idx = slice(10, 140, 2)
        Xs = np.linspace(-10, 10, n)[:,None]
        X = Xs[idx, :]
        true_sigma, true_ls = (0.25, 1.0)
        cov1 = pm.gp.cov.ExpQuad(1, ls=true_ls)
        true_f1 = np.random.multivariate_normal(np.zeros(n),
                                                    cov1(Xs).eval(), 1).flatten()
        true_f2 = (0.3 * Xs).flatten()
        true_f = true_f1 + true_f2
        true_y = true_f + true_sigma * np.random.randn(n)
        obs = true_y[idx]

        with pm.Model() as model:
            ls = pm.Gamma("ls", alpha=100, beta=100)
            sigma = pm.Gamma("sigma", alpha=100, beta=400)
            c = pm.Normal("c", mu=0.0, sd=2.0)
            cov1 = pm.gp.cov.ExpQuad(1, ls=ls)
            cov2 = pm.gp.cov.Linear(1, c=c)
            gp1 = pm.gp.Marginal(cov_func=cov1)
            gp2 = pm.gp.Marginal(cov_func=cov2)
            gp = gp1 + gp2
            y_ = gp.marginal_likelihood("y", X=X, noise=sigma, y=obs)

        with model:
            map_point = pm.find_MAP(method="BFGS")

        with model:
            fs1 = gp1.conditional("fs1", Xnew=Xs, given={"X": X, "y": obs, "noise": sigma, "gp": gp})
            fs2 = gp2.conditional("fs2", Xnew=Xs, given={"X": X, "y": obs, "noise": sigma, "gp": gp})
            fst = gp.conditional("fst", Xnew=Xs)
            ppc = pm.sample_ppc([map_point], samples=200, vars=[fs1, fs2, fst])

        test_idx = slice(10, 140)
        assert_allclose_corr(true_f[test_idx],  np.mean(ppc["fst"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f1[test_idx], np.mean(ppc["fs1"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f2[test_idx], np.mean(ppc["fs2"], 0)[test_idx], atol=0.1)

    def test_latent(self):
        n = 150
        idx = slice(10, 140, 2)
        Xs = np.linspace(-10, 10, n)[:,None]
        X = Xs[idx, :]
        true_sigma, true_ls = (0.25, 1.0)
        cov1 = pm.gp.cov.ExpQuad(1, ls=true_ls)
        true_f1 = np.random.multivariate_normal(np.zeros(n),
                                                    cov1(Xs).eval(), 1).flatten()
        true_f2 = (0.3 * Xs).flatten()
        true_f = true_f1 + true_f2
        true_y = true_f + true_sigma * np.random.randn(n)
        obs = true_y[idx]

        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(1, ls=true_ls)
            cov2 = pm.gp.cov.Linear(1, c=0.0)
            gp1 = pm.gp.Latent(cov_func=cov1)
            gp2 = pm.gp.Latent(cov_func=cov2)
            gp = gp1 + gp2
            f = gp.prior("f", X)
            y_ = pm.Normal("y", mu=f, sd=true_sigma, observed=obs)
            tr = pm.sample(100, chains=1)

        with model:
            fs1 = gp1.conditional("fs1", Xnew=Xs, given={"f": f, "X": X, "gp": gp})
            fs2 = gp2.conditional("fs2", Xnew=Xs, given={"f": f, "X": X, "gp": gp})
            fst = gp.conditional("fst", Xnew=Xs)
            ppc = pm.sample_ppc(tr, 200, vars=[fs1, fs2, fst])

        test_idx = slice(10, 140)
        assert_allclose_corr(true_f[test_idx],  np.mean(ppc["fst"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f1[test_idx], np.mean(ppc["fs1"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f2[test_idx], np.mean(ppc["fs2"], 0)[test_idx], atol=0.1)
    
    @pytest.mark.parametrize("approx", ["DTC", "VFE", "FITC"])
    def test_marginalsparse(self, approx):
        n = 150
        idx = slice(10, 140, 2)
        Xs = np.linspace(-10, 10, n)[:,None]
        X = Xs[idx, :]
        true_sigma, true_ls = (0.25, 1.0)
        cov1 = pm.gp.cov.ExpQuad(1, ls=true_ls)
        true_f1 = np.random.multivariate_normal(np.zeros(n),
                                                    cov1(Xs).eval(), 1).flatten()
        true_f2 = (0.3 * Xs).flatten()
        true_f = true_f1 + true_f2
        true_y = true_f + true_sigma * np.random.randn(n)
        obs = true_y[idx]

        with pm.Model() as model:
            ls = pm.Gamma("ls", alpha=100, beta=100)
            sigma = pm.Gamma("sigma", alpha=100, beta=400)
            c = pm.Normal("c", mu=0.0, sd=2.0)
            cov1 = pm.gp.cov.ExpQuad(1, ls=ls)
            cov2 = pm.gp.cov.Linear(1, c=c)
            gp1 = pm.gp.MarginalSparse(cov_func=cov1, approx=approx)
            gp2 = pm.gp.MarginalSparse(cov_func=cov2, approx=approx)
            gp = gp1 + gp2
            Xu = 20 * (np.random.randn(50, 1) - 0.5)
            #Xu = pm.Normal("xu", mu=xu_mu, sd=2.0, shape=(50, 1))
            y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, noise=sigma, y=obs)

        with model:
            map_point = pm.find_MAP(method="L-BFGS-B")

        with model:
            fs1 = gp1.conditional("fs1", Xnew=Xs, given={"X": X, "Xu": Xu, "y": obs, "noise": sigma, "gp": gp})
            fs2 = gp2.conditional("fs2", Xnew=Xs, given={"X": X, "Xu": Xu, "y": obs, "noise": sigma, "gp": gp})
            fst = gp.conditional("fst", Xnew=Xs)
            ppc = pm.sample_ppc([map_point], samples=200, vars=[fs1, fs2, fst])

        test_idx = slice(10, 140)
        assert_allclose_corr(true_f[test_idx],  np.mean(ppc["fst"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f1[test_idx], np.mean(ppc["fs1"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f2[test_idx], np.mean(ppc["fs2"], 0)[test_idx], atol=0.1)
        

class TestLatent(SeededTest):
    def test_conditionals(self):
        n = 150
        idx = slice(10, 140, 2)
        Xs = np.linspace(-10, 10, n)[:,None]
        X = Xs[idx, :]
        true_ls, true_sigma, true_c = (3.0, 0.1, 1.0)
        cov = pm.gp.cov.ExpQuad(1, ls=true_ls)
        mean = pm.gp.mean.Constant(true_c)
        true_f = np.random.multivariate_normal(mean(Xs).eval(),
                                                    cov(Xs).eval(), 1).flatten()
        true_y = true_f + true_sigma * np.random.randn(n)
        obs = true_y[idx]

        with pm.Model() as model:
            gp = pm.gp.Latent(mean_func=mean,
                                cov_func=cov)
            f = gp.prior("f", X)
            y_ = pm.Normal("y", mu=f, sd=true_sigma, observed=obs)
            tr = pm.sample(100, chains=1)

        model = model
        tr = tr
        gp = gp

        with model:
            fs = gp.conditional("fs", Xnew=Xs)
            ppc = pm.sample_ppc(tr, 500, vars=[fs])
        f_mu = np.mean(ppc["fs"], 0)
        f_sd = np.std(ppc["fs"], 0)
        
        test_idx = slice(10, 140)
        assert_allclose_norm(f_mu[test_idx], true_f[test_idx], maxnorm=2.0)
        npt.assert_array_less(true_f, f_mu + 3*f_sd)
        npt.assert_array_less(f_mu - 3*f_sd, true_f)

    def test_additive(self):
        n = 150
        idx = slice(10, 140, 2)
        Xs = np.linspace(-10, 10, n)[:,None]
        X = Xs[idx, :]
        true_sigma = 0.2
        cov1 = pm.gp.cov.ExpQuad(1, ls=3)
        cov2 = pm.gp.cov.Linear(1, c=0)
        true_f1 = np.random.multivariate_normal(np.zeros(n),
                                                    cov1(Xs).eval(), 1).flatten()
        true_f2 = np.random.multivariate_normal(np.zeros(n),
                                                    cov2(Xs).eval(), 1).flatten()
        true_y = true_f1 + true_f2 + true_sigma * np.random.randn(n)
        obs = true_y[idx]

        with pm.Model() as model:
            gp1 = pm.gp.Latent(cov_func=cov1)
            gp2 = pm.gp.Latent(cov_func=cov2)
            gp = gp1 + gp2
            f = gp.prior("f", X)
            y_ = pm.Normal("y", mu=f, sd=true_sigma, observed=obs)
            tr = pm.sample(100, chains=1)

        model = model
        tr = tr
        gp = gp

        with model:
            fs1 = gp1.conditional("fs1", Xnew=Xs, given={"f": f, "X": X, "gp": gp})
            fs2 = gp2.conditional("fs2", Xnew=Xs, given={"f": f, "X": X, "gp": gp})
            fst = gp.conditional("fst", Xnew=Xs)
            ppc = pm.sample_ppc(tr, 200, vars=[fs1, fs2, fst])

        f_mu = np.mean(ppc["fst"], 0)
        f_sd = np.std(ppc["fst"], 0)

        test_idx = slice(10, 140)
        assert_allclose_norm(true_f1[test_idx], np.mean(ppc["fs1"], 0)[test_idx], maxnorm=2.0)
        assert_allclose_norm(true_f2[test_idx], np.mean(ppc["fs2"], 0)[test_idx], maxnorm=2.0)

        
        
class TestGPAdditiveRaises(object):
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
            gp1 = pm.gp.MarginalSparse(cov_func=cov_func, approx="VFE")
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

    def testAdditiveTPRaises(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.TP(cov_func=cov_func, nu=10)
            gp2 = pm.gp.TP(cov_func=cov_func, nu=10)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2










