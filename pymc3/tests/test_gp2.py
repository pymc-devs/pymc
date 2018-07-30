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


class TestMarginal2(object):
    def setup_method(self):
        n = 50
        self.Xs = np.linspace(0, 10, n)[:,None]
        self.X = self.Xs[::2, :]
        self.true_ls, self.true_sigma, self.true_c = (3.0, 0.05, 1.0)
        self.cov = pm.gp.cov.ExpQuad(1, ls=self.true_ls)
        self.mean = pm.gp.mean.Constant(self.true_c)
        self.true_f = np.random.multivariate_normal(self.mean(self.Xs).eval(),
                                                    self.cov(self.Xs).eval(), 1).flatten()
        true_y = self.true_f + self.true_sigma * np.random.randn(n)
        self.obs = true_y[::2]

        with pm.Model() as model:
            ls = pm.Gamma("ls", alpha=300, beta=100)
            sigma = pm.Gamma("sigma", alpha=20, beta=400)
            c = pm.Normal("c", mu=1.0, sd=0.1)
            gp = pm.gp.Marginal(mean_func=pm.gp.mean.Constant(c),
                                cov_func=pm.gp.cov.ExpQuad(1, ls=ls))
            y_ = gp.marginal_likelihood("y", self.X, self.obs, noise=sigma)
            tr = pm.sample(200, tune=200, chains=1)

        self.map_point = {varname: np.mean(tr[varname]) for varname in tr.varnames}
        self.model = model
        self.tr = tr
        self.gp = gp

        with self.model:
            f = self.gp.conditional("f", Xnew=self.Xs)
            self.ppc = pm.sample_ppc(self.tr, 200, vars=[f])

        self.f_mu = np.mean(self.ppc["f"], 0)
        self.f_sd = np.std(self.ppc["f"], 0)

    def test_priors(self):
        # posteriors must be close to true values (which had strong priors)
        npt.assert_allclose(np.mean(self.tr["ls"]), self.true_ls, atol=0.1)
        npt.assert_allclose(np.mean(self.tr["sigma"]), self.true_sigma, atol=0.01)
        npt.assert_allclose(np.mean(self.tr["c"]), self.true_c, atol=0.1)

    def test_conditionals(self):
        npt.assert_allclose(self.f_mu, self.true_f, atol=0.1)
        npt.assert_array_less(self.true_f, self.f_mu + 3*self.f_sd)
        npt.assert_array_less(self.f_mu - 3*self.f_sd, self.true_f)

    @pytest.mark.parametrize("diag", [True, False])
    @pytest.mark.parametrize("pred_noise", [True, False])
    def test_predictions(self, diag, pred_noise):
        if diag:
            mu, var = self.gp.predict(self.Xs, point=self.map_point, diag=diag, pred_noise=pred_noise)
            npt.assert_allclose(np.sqrt(var), self.f_sd, atol=0.1)
        else:
            mu, cov = self.gp.predict(self.Xs, point=self.map_point, diag=diag, pred_noise=pred_noise)
            npt.assert_allclose(np.sqrt(np.diag(cov)), self.f_sd, atol=0.1)
        npt.assert_allclose(mu, self.f_mu, atol=0.1)


class TestMarginal(object):
    def setup_method(self):
        n, ns, d = (40, 50, 3)
        self.X, self.Xs = (np.random.randn(n, d), np.random.randn(ns, d))
        self.ls, self.sigma = (np.random.rand(d) + 0.1, 0.1)
        self.y = np.random.randn(n)
        self.n, self.ns, self.d = (n, ns, d)

    def testMarginalLogp(self):
        # pymc3 model
        ls = np.random.rand(self.d)
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(self.d, ls)
            mean_func = pm.gp.mean.Constant(0.0)
            gp = pm.gp.Marginal(mean_func, cov_func)
            y_ = gp.marginal_likelihood("y_var",
                                        self.X, self.y, self.sigma)
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
        mu, cov = ref_gp.conditional(self.X, self.Xs, self.sigma, y=self.y)
        npt.assert_allclose(f.distribution.logp(tt.as_tensor_variable(x)).eval(),
                            ref_gp.logp_func(mu, cov)(x),
                            rtol=1e-3)

    def testMarginalPredict(self):
        # pymc3 model
        ls = np.random.rand(self.d)
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(self.d, ls)
            mean_func = pm.gp.mean.Constant(0.0)
            gp = pm.gp.Marginal(mean_func, cov_func)
            y_ = gp.marginal_likelihood("y_var",
                                        self.X, self.y, self.sigma)
            f = gp.conditional("f", self.Xs)

        ref_gp = GPReference(mean_func, cov_func)
        mu_ref, cov_ref = ref_gp.conditional(self.X, self.Xs, self.sigma, y=self.y)
        mu_pymc3, cov_pymc3 = gp.predict(self.Xs)
        npt.assert_allclose(mu_ref, mu_pymc3, rtol=1e-2)
        npt.assert_allclose(cov_ref, cov_pymc3, rtol=1e-2)

        # test predict, diag=True
        mu_pymc3, var_pymc3 = gp.predict(self.Xs, diag=True)
        npt.assert_allclose(mu_ref, mu_pymc3, rtol=1e-2)
        npt.assert_allclose(np.diag(cov_ref), var_pymc3, rtol=1e-2)

    def testMarginalAdditive(self):
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
            f_sum = gp_sum.marginal_likelihood("f_sum", self.X,
                                               self.y, noise=self.sigma)
        model1_logp = model1.logp({"f_sum": self.y})

        with pm.Model() as model2:
            gp_total = pm.gp.Marginal(reduce(add, means), reduce(add, covs))
            f_total = gp_total.marginal_likelihood("f_total", self.X,
                                                   self.y, noise=self.sigma)
        model2_logp = model2.logp({"f_total": self.y})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gp_sum.conditional("fp1", self.Xs,
                                     given={"X": self.X, "y": self.y,
                                            "noise": self.sigma, "gp": gp_sum})
        with model2:
            fp2 = gp_total.conditional("fp2", self.Xs)

        fp = np.random.randn(self.ns)
        npt.assert_allclose(fp1.logp({"fp1": fp}), fp2.logp({"fp2": fp}), rtol=1e-2)


class TestLatent(object):
    # to test multiobs, do two/three column, compare
    # one column results to single column result
    def setup_method(self):
        n, ns, d = (40, 50, 3)
        self.X, self.Xs = (np.random.randn(n, d), np.random.randn(ns, d))
        self.ls = np.random.rand(d) + 0.1
        self.f = np.random.randn(n)
        self.n, self.ns, self.d = (n, ns, d)

    def testLatentLogp(self):
        # pymc3 model
        ls = np.random.rand(self.d)
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(self.d, ls)
            mean_func = pm.gp.mean.Constant(0.0)
            gp = pm.gp.Latent(mean_func, cov_func)
            fv = gp.prior("fv", self.X)
            fp = gp.conditional("fp", self.Xs)

        f_rotated = rotate_vector(cov_func(self.X).eval(), self.f)
        fs = np.random.randn(self.ns)
        latent_logp = model.logp({"fv_rotated_": f_rotated, "fp": fs})

        # reference GP
        ref_gp = GPReference(mean_func, cov_func)
        ref_priorlogp = np.sum(sp.stats.norm.logpdf(f_rotated, loc=np.zeros(self.n), scale=np.ones(self.n)))
        mu, cov = ref_gp.conditional(self.X, self.Xs, 1e-8, f=self.f)
        ref_condlogp = ref_gp.logp_func(mu, cov)(fs)
        npt.assert_allclose(latent_logp, ref_priorlogp + ref_condlogp,
                            rtol=1e-3)

    def testLatentAdditive(self):
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

        f_rotated = rotate_vector(cov_total(self.X).eval(),
                                            self.f, mean_total(self.X).eval())
        model1_logp = model1.logp({"f_sum_rotated_": f_rotated})

        with pm.Model() as model2:
            gp_total = pm.gp.Latent(mean_total, cov_total)
            f_total = gp_total.prior("f_total", self.X)
        model2_logp = model2.logp({"f_total_rotated_": f_rotated})
        npt.assert_allclose(model1_logp, model2_logp, atol=0, rtol=1e-2)

        with model1:
            fp1 = gp_sum.conditional("fp1", self.Xs,
                                     given={"X": self.X, "f": self.f,
                                            "gp": gp_sum})
        with model2:
            fp2 = gp_total.conditional("fp2", self.Xs)

        fp = np.random.randn(self.ns)
        npt.assert_allclose(fp1.logp({"f_sum_rotated_": f_rotated,
                                      "fp1": fp}),
                            fp2.logp({"f_total_rotated_": f_rotated,
                                      "fp2": fp}), atol=0, rtol=1e-2)



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

    def testAdditiveTPRaises(self):
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, [0.1, 0.2, 0.3])
            gp1 = pm.gp.TP(cov_func=cov_func, nu=10)
            gp2 = pm.gp.TP(cov_func=cov_func, nu=10)
            with pytest.raises(Exception) as e_info:
                gp1 + gp2


class TestTP(object):
    # to test multiobs, do two/three column, compare
    # one column results to single column result
    def setup_method(self):
        n, ns, d = (10, 15, 2)
        self.X, self.Xs = (np.random.randn(n, d), np.random.randn(ns, d))
        self.ls = np.random.rand(d) + 0.1
        self.f = np.random.randn(n)
        self.n, self.ns, self.d = (n, ns, d)

    def testTPPredict(self):
        ls = np.random.rand(self.d)
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(self.d, ls)
            mean_func = pm.gp.mean.Constant(1.0)
            tp = pm.gp.TP(mean_func=mean_func, cov_func=cov_func, nu=10000)
            fv = tp.prior("fv", self.X)
            fp = tp.conditional("fp", self.Xs)

        f_rotated = rotate_vector(cov_func(self.X).eval(),
                                  self.f - mean_func(self.X).eval())
        mu, cov = draw_values([fp.distribution.mu, fp.distribution.cov],
                               point={"fv_rotated_": f_rotated,
                                      "chi2__log__": np.log(10000)})

        # reference GP
        ref_gp = GPReference(mean_func, cov_func)
        mu_ref, cov_ref = ref_gp.conditional(self.X, self.Xs, 0.0, f=self.f)
        npt.assert_allclose(mu_ref, mu, rtol=1e-2)
        npt.assert_allclose(cov_ref, cov, rtol=1e-2)


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













