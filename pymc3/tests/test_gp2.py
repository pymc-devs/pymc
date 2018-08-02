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
    # test two vectors similarity by their correlation
    corr = np.corrcoef(x1, x2)
    npt.assert_allclose(corr, np.ones((2,2)), atol=atol)

def kron_idx(dim1_idx, dim2_idx, n1, n2):
    # translate indices in two dimensions to a flattened index
    idx_grid = np.arange(n1*n2).reshape(n1, n2)
    dim_idx = cartesian(dim1_idx, dim2_idx)
    return idx_grid[dim_idx[:, 0], dim_idx[:, 1]]

class TestMarginalKron(object):
    def setup_method(self):
        n1, n2 = (35, 35)
        x1 = np.linspace(0, 10, n1)
        x2 = np.linspace(0, 10, n2)
        dim_idx1_tr, dim_idx2_tr = (np.arange(0, 30, 2), np.arange(0, 30, 2))
        dim_idx1_te, dim_idx2_te = (np.arange(20, 35, 2), np.arange(20, 35, 2))
        self.idx_tr = kron_idx(dim_idx1_tr, dim_idx2_tr, n1, n2)
        self.idx_te = kron_idx(dim_idx1_te, dim_idx2_te, n1, n2)
        full_X = cartesian(x1[:, None], x2[:, None])
        self.X  = [x1[dim_idx1_tr, None], x2[dim_idx2_tr, None]]
        self.Xs = cartesian(x1[dim_idx1_te, None], x2[dim_idx2_te, None])
        self.true_ls1, self.true_ls2, self.true_sigma = (2.0, 3.0, 0.1)
        cov = pm.gp.cov.Matern52(2, ls=self.true_ls1, active_dims=[0]) *\
              pm.gp.cov.Cosine(2, ls=self.true_ls2, active_dims=[1])
        self.true_f = pm.MvNormal.dist(mu=np.zeros(n1 * n2), cov=cov(full_X)).random(size=1)
        true_y = self.true_f + self.true_sigma * pm.Normal.dist(mu=0.0, sd=1.0).random(size=n1*n2)
        self.obs = true_y[self.idx_tr]

        with pm.Model() as model:
            cov_x1 = pm.gp.cov.Matern52(1, ls=self.true_ls1)
            cov_x2 = pm.gp.cov.Cosine(1, ls=self.true_ls2)
            gp = pm.gp.MarginalKron(cov_funcs=[cov_x1, cov_x2])
            sigma = pm.Gamma("sigma", alpha=10, beta=100)
            y_ = gp.marginal_likelihood("y", Xs=self.X, y=self.obs, sigma=sigma)

        with model:
            self.map_point = pm.find_MAP(method="BFGS")

        self.model = model
        self.gp = gp

        with self.model:
            fs = self.gp.conditional("fs", Xnew=self.Xs)
            self.ppc = pm.sample_ppc([self.map_point], 50, vars=[fs])

        self.f_mu = np.mean(self.ppc["fs"], 0)
        self.f_sd = np.std(self.ppc["fs"], 0)

    def test_priors(self):
        # posteriors must be close to true values (which had strong priors)
        npt.assert_allclose(self.map_point["sigma"], self.true_sigma, atol=0.01)

    def test_conditionals(self):
        true_f = self.true_f[self.idx_te]
        npt.assert_array_less(true_f, self.f_mu + 3*self.f_sd)
        npt.assert_array_less(self.f_mu - 3*self.f_sd, true_f)
        assert_allclose_corr(self.f_mu, true_f, atol=0.1)

    def testPredictions(self):
        mu, var = self.gp.predict(self.Xs, point=self.map_point, diag=True, pred_noise=False)
        assert_allclose_corr(np.sqrt(var), self.f_sd, atol=0.1)
        assert_allclose_corr(mu, self.f_mu, atol=0.1)



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
        self.gp = gp

        with self.model:
            fs = self.gp.conditional("fs", Xnew=self.Xs)
            self.ppc = pm.sample_ppc([self.map_point], 100, vars=[fs])

        self.f_mu = np.mean(self.ppc["fs"], 0)
        self.f_sd = np.std(self.ppc["fs"], 0)

    def testPriors(self):
        # posteriors must be close to true values (which had strong priors)
        npt.assert_allclose(self.map_point["ls"], self.true_ls, atol=0.1)
        npt.assert_allclose(self.map_point["sigma"], self.true_sigma, atol=0.01)
        npt.assert_allclose(self.map_point["c"], self.true_c, atol=0.1)

    def testConditionals(self):
        test_idx = slice(10, 140)
        assert_allclose_corr(self.f_mu[test_idx], self.true_f[test_idx], atol=0.1)
        npt.assert_array_less(self.true_f, self.f_mu + 3*self.f_sd)
        npt.assert_array_less(self.f_mu - 3*self.f_sd, self.true_f)

    def testPredictions(self):
        mu, var = self.gp.predict(self.Xs, point=self.map_point, diag=True, pred_noise=False)
        npt.assert_allclose(np.sqrt(var), self.f_sd, atol=0.1)
        npt.assert_allclose(mu, self.f_mu, atol=0.1)


class TestLatent(SeededTest):
    def testConditionals(self):
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
            ppc = pm.sample_ppc(tr, 100, vars=[fs])
        f_mu = np.mean(ppc["fs"], 0)
        f_sd = np.std(ppc["fs"], 0)

        test_idx = slice(10, 140)
        assert_allclose_corr(f_mu[test_idx], true_f[test_idx], atol=0.1)
        npt.assert_array_less(true_f, f_mu + 3*f_sd)
        npt.assert_array_less(f_mu - 3*f_sd, true_f)


class TestTP(SeededTest):
    def testConditionals(self):
        n = 150
        idx = slice(10, 140, 2)
        Xs = np.linspace(-10, 10, n)[:,None]
        X = Xs[idx, :]
        true_ls, true_sigma, true_c, true_nu = (3.0, 0.1, 1.0, 5)
        cov = pm.gp.cov.ExpQuad(1, ls=true_ls)
        mean = pm.gp.mean.Constant(true_c)
        true_f = pm.MvStudentT.dist(mu=mean(Xs).eval(),
                                    cov=cov(Xs).eval(),
                                    nu=true_nu).random(size=1)
        true_y = true_f + true_sigma * np.random.randn(n)
        obs = true_y[idx]

        with pm.Model() as model:
            gp = pm.gp.TP(mean_func=mean,
                          cov_func=cov, nu=true_nu)
            f = gp.prior("f", X)
            y_ = pm.Normal("y", mu=f, sd=true_sigma, observed=obs)
            tr = pm.sample(100, chains=1)

        model = model
        tr = tr
        gp = gp

        with model:
            fs = gp.conditional("fs", Xnew=Xs)
            ppc = pm.sample_ppc(tr, 100, vars=[fs])
        f_mu = np.mean(ppc["fs"], 0)
        f_sd = np.std(ppc["fs"], 0)

        test_idx = slice(10, 140)
        assert_allclose_corr(f_mu[test_idx], true_f[test_idx], atol=0.1)
        npt.assert_array_less(true_f, f_mu + 3*f_sd)
        npt.assert_array_less(f_mu - 3*f_sd, true_f)


class TestMarginalSparse(object):
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
            gp = pm.gp.MarginalSparse(mean_func=pm.gp.mean.Constant(c),
                                      cov_func=pm.gp.cov.ExpQuad(1, ls=ls),
                                      approx="VFE")
            Xu = 20 * (np.random.randn(50, 1) - 0.5)
            y_ = gp.marginal_likelihood("y", X=self.X, Xu=Xu, noise=sigma, y=self.obs)
            self.map_point = pm.find_MAP(method="BFGS")

        self.model = model
        self.gp = gp

        with self.model:
            fs = self.gp.conditional("fs", Xnew=self.Xs)
            self.ppc = pm.sample_ppc([self.map_point], 100, vars=[fs])

        self.f_mu = np.mean(self.ppc["fs"], 0)
        self.f_sd = np.std(self.ppc["fs"], 0)

    def testPriors(self):
        # posteriors must be close to true values (which had strong priors)
        npt.assert_allclose(self.map_point["ls"], self.true_ls, atol=0.1)
        npt.assert_allclose(self.map_point["sigma"], self.true_sigma, atol=0.01)
        npt.assert_allclose(self.map_point["c"], self.true_c, atol=0.1)

    def testConditionals(self):
        test_idx = slice(10, 140)
        assert_allclose_corr(self.f_mu[test_idx], self.true_f[test_idx], atol=0.1)
        npt.assert_array_less(self.true_f, self.f_mu + 3*self.f_sd)
        npt.assert_array_less(self.f_mu - 3*self.f_sd, self.true_f)

    @pytest.mark.parametrize("diag", [True, False])
    @pytest.mark.parametrize("pred_noise", [True, False])
    def testPredictions(self, diag, pred_noise):
        if diag:
            mu, var = self.gp.predict(self.Xs, point=self.map_point, diag=diag, pred_noise=pred_noise)
            npt.assert_allclose(np.sqrt(var), self.f_sd, atol=0.1)
        else:
            mu, cov = self.gp.predict(self.Xs, point=self.map_point, diag=diag, pred_noise=pred_noise)
            npt.assert_allclose(np.sqrt(np.diag(cov)), self.f_sd, atol=0.1)
        npt.assert_allclose(mu, self.f_mu, atol=0.1)





class TestAdditive(SeededTest):
    def testMarginal(self):
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
            ppc = pm.sample_ppc([map_point], samples=100, vars=[fs1, fs2, fst])

        test_idx = slice(10, 140)
        assert_allclose_corr(true_f[test_idx],  np.mean(ppc["fst"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f1[test_idx], np.mean(ppc["fs1"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f2[test_idx], np.mean(ppc["fs2"], 0)[test_idx], atol=0.1)

    def testLatent(self):
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
            ppc = pm.sample_ppc(tr, 100, vars=[fs1, fs2, fst])

        test_idx = slice(10, 140)
        assert_allclose_corr(true_f[test_idx],  np.mean(ppc["fst"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f1[test_idx], np.mean(ppc["fs1"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f2[test_idx], np.mean(ppc["fs2"], 0)[test_idx], atol=0.1)

    @pytest.mark.parametrize("approx", ["DTC", "VFE", "FITC"])
    def testMarginalSparse(self, approx):
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
            y_ = gp.marginal_likelihood("y", X=X, Xu=Xu, noise=sigma, y=obs)

        with model:
            map_point = pm.find_MAP(method="L-BFGS-B")

        with model:
            fs1 = gp1.conditional("fs1", Xnew=Xs, given={"X": X, "Xu": Xu, "y": obs, "noise": sigma, "gp": gp})
            fs2 = gp2.conditional("fs2", Xnew=Xs, given={"X": X, "Xu": Xu, "y": obs, "noise": sigma, "gp": gp})
            fst = gp.conditional("fst", Xnew=Xs)
            ppc = pm.sample_ppc([map_point], samples=100, vars=[fs1, fs2, fst])

        test_idx = slice(10, 140)
        assert_allclose_corr(true_f[test_idx],  np.mean(ppc["fst"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f1[test_idx], np.mean(ppc["fs1"], 0)[test_idx], atol=0.1)
        assert_allclose_corr(true_f2[test_idx], np.mean(ppc["fs2"], 0)[test_idx], atol=0.1)

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







