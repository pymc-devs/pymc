#  pylint:disable=unused-variable
from functools import reduce, partial
from ..math import cartesian, kronecker
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

np.random.seed(101)


class GPReference(object):
    R"""
    Reference implementation of GP to compare pymc3 implementations to.
    """
    def __init__(self, mean_func, cov_func):
        self.mean_func = mean_func
        self.cov_func = cov_func
        self.solve = partial(sp.linalg.solve, assume_a="pos")

    def eval_mu_covs(self, X, Xs=None):
        Kxx = self.cov_func(X)
        mu_xx = self.mean_func(X)
        if Xs is not None:
            mu_ss = self.mean_func(Xs)
            Kxs = self.cov_func(X, Xs)
            Ksx = Kxs.T
            Kss = self.cov_func(Xs)
            return Kxx, Kxs, Ksx, Kss, mu_xx, mu_ss
        else:
            return Kxx, mu_xx

    def marginal(self, X, sigma, y):
        R"""
        Marginal distribution
        """
        Kxx, mu = [x.eval() for x in self.eval_mu_covs(X)]
        cov = Kxx + np.square(sigma) * np.eye(Kxx.shape[0])
        return mu, cov + 1e-6*np.eye(cov.shape[0])

    def conditional(self, X, Xs, sigma=0.0, y=None, f=None):
        R"""
        Conditional distribution. Comparable to .conditional for
        gp.Latent and .conditional for gp.Marginal implementations.
        """
        tt_vals = self.eval_mu_covs(X, Xs)
        Kxx, Kxs, Ksx, Kss, mu_xx, mu_ss = [x.eval() for x in tt_vals]
        if (f is not None) and (sigma != 0.0):
            raise ValueError("when using f, sigma must be zero")
        if (f is None) and (y is None):
            raise ValueError("one of f or y must be provided")
        Sigma = Kxx + np.square(sigma)*np.eye(Kxx.shape[0])
        z = f if y is None else y
        mu = np.dot(Ksx, self.solve(Sigma, z))
        cov = Kss - np.dot(Ksx, self.solve(Sigma, Kxs))
        return mu, cov + 1e-6*np.eye(cov.shape[0])

    def logp_func(self, mu, cov):
        return partial(sp.stats.multivariate_normal.logpdf, mean=mu, cov=cov)



class TestMarginal(object):
    def setup_method(self):
        n, ns, d, m = (20, 30, 3, 4)
        self.X, self.Xs = (np.random.randn(n, d), np.random.randn(ns, d))
        self.ls, self.sigma = (np.random.rand(d) + 0.1, 0.1)
        self.y_vec = np.random.randn(n)
        self.y_mat = np.random.randn(n, m)
        self.n, self.ns = (n, ns)
        self.d, self.m = (d, m)

    @pytest.mark.parametrize('multiple_obs', [False, pytest.param(True, marks=pytest.mark.xfail)])
    def testMarginalLogp(self, multiple_obs):
        y = self.y_vec if not multiple_obs else self.y_mat
        # pymc3 model
        ls = np.random.rand(self.d)
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(self.d, ls)
            mean_func = pm.gp.mean.Constant(0.0)
            gp = pm.gp.Marginal(mean_func, cov_func)
            y_ = gp.marginal_likelihood("y_var",
                                        self.X, y, self.sigma)
            f = gp.conditional("f", self.Xs)
        # reference GP
        ref_gp = GPReference(mean_func, cov_func)
        # test marginal_likelihood logp
        x = np.random.randn(self.n)
        mu, cov = ref_gp.marginal(self.X, self.sigma, y)
        npt.assert_allclose(y_.distribution.logp(tt.as_tensor_variable(x)).eval(),
                            ref_gp.logp_func(mu, cov)(x),
                            rtol=1e-3)
        # test conditional logp
        x = np.random.randn(self.ns)
        mu, cov = ref_gp.conditional(self.X, self.Xs, self.sigma, y=y)
        npt.assert_allclose(f.distribution.logp(tt.as_tensor_variable(x)).eval(),
                            ref_gp.logp_func(mu, cov)(x),
                            rtol=1e-3)

    @pytest.mark.parametrize('multiple_obs', [False, pytest.param(True, marks=pytest.mark.xfail)])
    def testMarginalPredict(self, multiple_obs):
        y = self.y_vec if not multiple_obs else self.y_mat
        # pymc3 model
        ls = np.random.rand(self.d)
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(self.d, ls)
            mean_func = pm.gp.mean.Constant(0.0)
            gp = pm.gp.Marginal(mean_func, cov_func)
            y_ = gp.marginal_likelihood("y_var",
                                        self.X, y, self.sigma)
            f = gp.conditional("f", self.Xs)
        # reference GP
        ref_gp = GPReference(mean_func, cov_func)
        # test marginal_likelihood logp
        x = np.random.randn(self.ns)

        # test predict
        mu_ref, cov_ref = ref_gp.conditional(self.X, self.Xs, self.sigma, y=y)
        mu_pymc3, cov_pymc3 = gp.predict(self.Xs)
        npt.assert_allclose(mu_ref, mu_pymc3, rtol=1e-3)
        npt.assert_allclose(cov_ref, cov_pymc3, rtol=1e-3)

        # test predict, diag=True
        mu_pymc3, var_pymc3 = gp.predict(self.Xs, diag=True)
        npt.assert_allclose(mu_ref, mu_pymc3, rtol=1e-3)
        npt.assert_allclose(np.diag(cov_ref), var_pymc3, rtol=1e-3)



class TestAdditive(object):
    def setup_method(self):
        n, ns, d, m = (20, 30, 2, 3)
        self.X, self.Xs = (np.random.randn(n, d), np.random.randn(ns, d))
        self.ls, self.sigma = (np.random.rand(d) + 0.1, 0.1)
        self.y_vec = np.random.randn(n)
        self.y_mat = np.random.randn(n, m)
        self.n, self.ns = (n, ns)
        self.d, self.m = (d, m)
        self.ls1, self.ls2 = (np.random.rand(self.d) + 0.1, np.random.rand(self.d) + 0.1)


    @pytest.mark.parametrize('multiple_obs', [False, pytest.param(True, marks=pytest.mark.xfail)])
    def testMarginal(self, multiple_obs):
        y = self.y_vec if not multiple_obs else self.y_mat
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(self.d, self.ls1)
            cov2 = pm.gp.cov.ExpQuad(self.d, self.ls2)
            cov_func = cov1 + cov2
            mean_func = pm.gp.mean.Constant(0.0)
            gp1 = pm.gp.Marginal(mean_func, cov1)
            gp2 = pm.gp.Marginal(mean_func, cov2)
            gp = gp1 + gp2
            y_ = gp.marginal_likelihood("y_var",
                                        self.X, y, self.sigma)
            f = gp.conditional("f", self.Xs)

        ref_gp = GPReference(mean_func, cov_func)

        # test marginal likelihood
        x = np.random.randn(self.n)
        mu, cov = ref_gp.marginal(self.X, self.sigma, y)
        npt.assert_allclose(y_.distribution.logp(tt.as_tensor_variable(x)).eval(),
                            ref_gp.logp_func(mu, cov)(x),
                            rtol=1e-3)

        # test conditional logp
        x = np.random.randn(self.ns)
        mu, cov = ref_gp.conditional(self.X, self.Xs, self.sigma, y=y)
        npt.assert_allclose(f.distribution.logp(tt.as_tensor_variable(x)).eval(),
                            ref_gp.logp_func(mu, cov)(x),
                            rtol=1e-3)

        # test conditional alternative syntax
        with model:
            f2 = gp.conditional("fp1", self.Xs, given={"X": self.X, "y": y,
                                                       "sigma": self.sigma, "gp": gp})
        x = np.random.randn(self.ns)
        mu, cov = ref_gp.conditional(self.X, self.Xs, self.sigma, y=y)
        npt.assert_allclose(f2.distribution.logp(tt.as_tensor_variable(x)).eval(),
                            ref_gp.logp_func(mu, cov)(x),
                            rtol=1e-3)


    @pytest.mark.parametrize('multiple_obs', [False, pytest.param(True, marks=pytest.mark.xfail)])
    def testLatent(self, multiple_obs):
        f = self.y_vec if not multiple_obs else self.y_mat
        # additive version
        with pm.Model() as model:
            cov1 = pm.gp.cov.ExpQuad(self.d, self.ls1)
            cov2 = pm.gp.cov.ExpQuad(self.d, self.ls2)
            cov_func = cov1 + cov2
            mean_func = pm.gp.mean.Constant(0.0)
            gp1 = pm.gp.Latent(mean_func, cov1)
            gp2 = pm.gp.Latent(mean_func, cov2)
            gp = gp1 + gp2
            fv = gp.prior("fv", self.X)
            fc = gp.conditional("fc", self.Xs)

        ref_gp = GPReference(mean_func, cov_func)

        x = np.random.randn(self.n)
        xs = np.random.randn(self.ns)
        # evaluate reference logp
        mu, cov = ref_gp.marginal(self.X, sigma=0.0, y=f)
        ref_prior_logp = ref_gp.logp_func(mu, cov)(x)
        mu, cov = ref_gp.conditional(self.X, self.Xs, sigma=0.0, f=f)
        ref_cond_logp = ref_gp.logp_func(mu, cov)(xs)

        # evaluate pymc3 model logp
        chol = sp.linalg.cholesky(cov_func(self.X).eval() + 1e-6*np.eye(self.n), lower=True)
        f_rotated = sp.linalg.solve_triangular(chol, f, lower=True)
        pymc3_logp = model.logp({"fv_rotated_": f_rotated, "fc": xs})
        npt.assert_allclose(pymc3_logp, ref_prior_logp + ref_cond_logp,
                            rtol=1e-3)














