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
        return mu, cov

    def conditional(self, X, Xs, sigma=0.0, y=None, f=None):
        R"""
        Conditional distribution. Comparable to .conditional for
        gp.Latent and .conditional for gp.Marginal implementations.
        """
        tt_vals = self.eval_mu_covs(self, X, Xs)
        Kxx, Kxs, Ksx, Kss, mu_xx, mu_ss = [x.eval() for x in tt_vals]
        if (f is not None) and (sigma != 0.0):
            raise ValueError("when using f, sigma must be zero")
        if (f is None) and (y is None):
            raise ValueError("one of f or y must be provided")
        Sigma = Kxx + np.square(sigma)*np.eye(Kxx.shape[0])
        z = f or y
        mu = np.dot(Ksx, self.solve(Sigma, z))
        cov = Kss - np.dot(Ksx, self.solve(Sigma, Kxs))
        return mu, cov

    def logp_func(self, mu, cov):
        return partial(sp.stats.multivariate_normal.logpdf, mean=mu, cov=cov)



class TestMarginal(object):
    def setup_method(self):
        n = 20
        ns = 30
        d = 3
        X = np.random.randn(n, d)
        Xs = np.random.randn(ns, d)
        y = np.random.randn(n)
        ls = np.random.randn(d)
        sigma = 0.1

        # pymc3 GP
        with pm.Model() as model:
            cov_func = pm.gp.cov.ExpQuad(3, ls)
            mean_func = pm.gp.mean.Constant(0.0)
            gp = pm.gp.Marginal(mean_func, cov_func)
            y_ = gp.marginal_likelihood("y_var", X, y, sigma)
            f = gp.conditional("f", Xs)

        # reference GP
        self.ref_gp = GPReference(mean_func, cov_func)
        self.n, self.ns = (n, ns)
        self.X, self.Xs = (X, Xs)
        self.y = y
        self.sigma = sigma
        self.f, self.y_ = (f, y_)

    def testMarginalMarginalLikelihoodLogp(self):
        x = np.random.randn(self.n)
        mu, cov = self.ref_gp.marginal(self.X, self.sigma, self.y)
        logp_ref = self.ref_gp.logp_func(mu, cov)(x)
        logp_pymc3 = self.y_.distribution.logp(tt.as_tensor_variable(x)).eval()
        npt.assert_allclose(logp_pymc3, logp_ref, atol=0.1)








