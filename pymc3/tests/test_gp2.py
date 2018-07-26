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
        if Xs is None:
            return Kxx, mu_xx
        else:
            mu_ss = self.mean_func(Xs)
            Kxs = self.cov_func(X, Xs)
            Ksx = Kxs.T
            Kss = self.cov_func(Xs)
            return Kxx, Kxs, Ksx, Kss, mu_xx, mu_ss

    def marginal(self, X, sigma, y):
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

        chol = sp.linalg.cholesky(cov_func(self.X).eval(), lower=True)
        x = sp.linalg.solve_triangular(chol, f, lower=True)
        xs = np.random.randn(self.ns)
        latent_logp = model.logp({"fv_rotated_": x, "fp": xs})

        # reference GP
        ref_gp = GPReference(mean_func, cov_func)
        # test marginal_likelihood logp
        mu, cov = ref_gp.marginal(self.X, 1e-6, f)
        ref_priorlogp = ref_gp.logp_func(np.zeros(self.n), cov_func(self.X).eval())(f)

        mu, cov = ref_gp.conditional(self.X, self.Xs, 1e-6, f=f)
        ref_condlogp = ref_gp.logp_func(mu, cov)(xs)

        npt.assert_allclose(np.random.randn(3), [latent_logp, ref_condlogp, ref_priorlogp])



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














