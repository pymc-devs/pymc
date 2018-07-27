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


def rotate_vector(K, f, mu=0.0):
    chol = sp.linalg.cholesky(K, lower=True)
    return sp.linalg.solve_triangular(chol, f - mu, lower=True)

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
        if Xs is None:
            return Kxx, mu_xx
        else:
            mu_ss = self.mean_func(Xs)
            Kxs = self.cov_func(X, Xs)
            Ksx = Kxs.T
            Kss = self.cov_func(Xs)
            return Kxx, Kxs, Ksx, Kss, mu_xx, mu_ss

    def marginal(self, X, sigma):
        R"""
        Marginal distribution
        """
        Kxx, mu = [x.eval() for x in self.eval_mu_covs(X)]
        cov = Kxx + np.square(sigma) * np.eye(Kxx.shape[0])
        return mu, cov

    def conditional(self, X, Xs, sigma=0.0, y=None, f=None):
        R"""
        Conditional distribution. Comparable to .conditional for
        gp.Latent and .conditional for gp.Marginal implementations.
        """
        tt_vals = self.eval_mu_covs(X, Xs)
        Kxx, Kxs, Ksx, Kss, mu_xx, mu_ss = [x.eval() for x in tt_vals]
        if (f is not None) and (sigma > 1e-6):
            raise ValueError("when using f, sigma must <= 1e-6")
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
        mu, cov = ref_gp.marginal(self.X, self.sigma)
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

    @pytest.mark.parametrize('multiple_obs', [False, pytest.param(True, marks=pytest.mark.xfail)])
    def testMarginalAdditive(self, multiple_obs):
        y = self.y_vec if not multiple_obs else self.y_mat
        means = (pm.gp.mean.Constant(0.0),
                 pm.gp.mean.Constant(0.5),
                 pm.gp.mean.Constant(1.0))
        covs = (pm.gp.cov.ExpQuad(self.d, self.ls),
                pm.gp.cov.ExpQuad(self.d, self.ls),
                pm.gp.cov.ExpQuad(self.d, self.ls))

        with pm.Model() as model1:
            gp1 = pm.gp.Marginal(means[0], covs[0])
            gp2 = pm.gp.Marginal(means[1], covs[1])
            gp3 = pm.gp.Marginal(means[2], covs[2])

            gp_sum = gp1 + gp2 + gp3
            f_sum = gp_sum.marginal_likelihood("f_sum", self.X, y, noise=self.sigma)
        model1_logp = model1.logp({"f_sum": y})

        with pm.Model() as model2:
            gp_total = pm.gp.Marginal(reduce(add, means), reduce(add, covs))
            f_total = gp_total.marginal_likelihood("f_total", self.X, y, noise=self.sigma)
        model2_logp = model2.logp({"f_total": y})

        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gp_sum.conditional("fp1", self.Xs, given={"X": self.X, "y": y,
                                                           "noise": self.sigma, "gp": gp_sum})
        with model2:
            fp2 = gp_total.conditional("fp2", self.Xs)

        fp = np.random.randn(self.ns)
        npt.assert_allclose(fp1.logp({"fp1": fp}), fp2.logp({"fp2": fp}), atol=0, rtol=1e-2)


class TestLatent(object):
    def setup_method(self):
        n, ns, d, m = (20, 30, 3, 4)
        self.X, self.Xs = (np.random.randn(n, d), np.random.randn(ns, d))
        self.ls = np.random.rand(d) + 0.1
        self.f_vec = np.random.randn(n)
        self.f_mat = np.random.randn(n, m)
        self.n, self.ns = (n, ns)
        self.d, self.m = (d, m)
        # problem with parameterizing multiple_obs, will need shapes, and 'x' and 'xs' will need
        # to be different sizes.

    @pytest.mark.parametrize('multiple_obs', [False, pytest.param(True, marks=pytest.mark.xfail)])
    def testLatentLogp(self, multiple_obs):
        f = self.f_vec if not multiple_obs else self.f_mat
        # pymc3 model
        ls = np.random.rand(self.d)
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(self.d, ls)
            mean_func = pm.gp.mean.Constant(0.0)
            gp = pm.gp.Latent(mean_func, cov_func)
            fv = gp.prior("fv", self.X)
            fp = gp.conditional("fp", self.Xs)

        f_rotated = rotate_vector(cov_func(self.X).eval(), f)
        fs = np.random.randn(self.ns)
        latent_logp = model.logp({"fv_rotated_": f_rotated, "fp": fs})

        # reference GP
        ref_gp = GPReference(mean_func, cov_func)
        ref_priorlogp = np.sum(sp.stats.norm.logpdf(f_rotated, loc=np.zeros(self.n), scale=np.ones(self.n)))
        mu, cov = ref_gp.conditional(self.X, self.Xs, 1e-8, f=f)
        ref_condlogp = ref_gp.logp_func(mu, cov)(fs)
        npt.assert_allclose(latent_logp, ref_priorlogp + ref_condlogp, rtol=1e-3)

    @pytest.mark.parametrize('multiple_obs', [False, pytest.param(True, marks=pytest.mark.xfail)])
    def testLatentAdditive(self, multiple_obs):
        f = self.f_vec if not multiple_obs else self.f_mat
        means = (pm.gp.mean.Constant(0.0),
                 pm.gp.mean.Constant(0.5),
                 pm.gp.mean.Constant(1.0))
        covs = (pm.gp.cov.ExpQuad(self.d, self.ls),
                pm.gp.cov.ExpQuad(self.d, self.ls),
                pm.gp.cov.ExpQuad(self.d, self.ls))
        mean_total = reduce(add, means)
        cov_total = reduce(add, covs)
        with pm.Model() as model1:
            gp1 = pm.gp.Latent(means[0], covs[0])
            gp2 = pm.gp.Latent(means[1], covs[1])
            gp3 = pm.gp.Latent(means[2], covs[2])
            gp_sum = gp1 + gp2 + gp3
            f_sum = gp_sum.prior("f_sum", self.X)

        f_rotated = rotate_vector(cov_total(self.X).eval(), f, mean_total(self.X).eval())
        model1_logp = model1.logp({"f_sum_rotated_": f_rotated})

        with pm.Model() as model2:
            gp_total = pm.gp.Latent(mean_total, cov_total)
            f_total = gp_total.prior("f_total", self.X)
        model2_logp = model2.logp({"f_total_rotated_": f_rotated})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gp_sum.conditional("fp1", self.Xs, given={"X": self.X, "f": f, "gp": gp_sum})
        with model2:
            fp2 = gp_total.conditional("fp2", self.Xs)

        fp = np.random.randn(self.ns)
        npt.assert_allclose(fp1.logp({"f_sum_rotated_": f_rotated, "fp1": fp}),
                            fp2.logp({"f_total_rotated_": f_rotated, "fp2": fp}), atol=0, rtol=1e-2)



class TestGPAdditiveRaises(object):
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














