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

import warnings

import aesara
import aesara.tensor as at
import numpy as np
import numpy.random as npr
import pytest
import scipy.special as sp
import scipy.stats as st

from aeppl.logprob import ParameterValueError
from aesara.tensor.random.utils import broadcast_params

import pymc as pm

from pymc.aesaraf import compile_pymc, floatX, intX
from pymc.distributions import logp
from pymc.distributions.shape_utils import to_tuple
from pymc.math import kronecker
from pymc.tests.distributions.util import (
    Domain,
    Nat,
    R,
    RandomPdMatrix,
    RealMatrix,
    Rplus,
    Simplex,
    Vector,
    assert_moment_is_expected,
    check_logp,
)
from pymc.tests.helpers import select_by_precision


def betafn(a):
    return floatX(sp.gammaln(a).sum(-1) - sp.gammaln(a.sum(-1)))


def logpow(v, p):
    return np.choose(v == 0, [p * np.log(v), 0])


def _dirichlet_logpdf(value, a):
    # scipy.stats.dirichlet.logpdf suffers from numerical precision issues
    return -betafn(a) + logpow(value, a - 1).sum()


dirichlet_logpdf = np.vectorize(_dirichlet_logpdf, signature="(n),(n)->()")


def _dirichlet_multinomial_logpmf(value, n, a):
    if value.sum() == n and (0 <= value).all() and (value <= n).all():
        sum_a = a.sum()
        const = sp.gammaln(n + 1) + sp.gammaln(sum_a) - sp.gammaln(n + sum_a)
        series = sp.gammaln(value + a) - sp.gammaln(value + 1) - sp.gammaln(a)
        return const + series.sum()
    else:
        return -np.inf


dirichlet_multinomial_logpmf = np.vectorize(
    _dirichlet_multinomial_logpmf, signature="(n),(),(n)->()"
)


def normal_logpdf_tau(value, mu, tau):
    return normal_logpdf_cov(value, mu, np.linalg.inv(tau)).sum()


def normal_logpdf_cov(value, mu, cov):
    return st.multivariate_normal.logpdf(value, mu, cov).sum()


def normal_logpdf_chol(value, mu, chol):
    return normal_logpdf_cov(value, mu, np.dot(chol, chol.T)).sum()


def normal_logpdf_chol_upper(value, mu, chol):
    return normal_logpdf_cov(value, mu, np.dot(chol.T, chol)).sum()


def matrix_normal_logpdf_cov(value, mu, rowcov, colcov):
    return st.matrix_normal.logpdf(value, mu, rowcov, colcov)


def matrix_normal_logpdf_chol(value, mu, rowchol, colchol):
    return matrix_normal_logpdf_cov(
        value, mu, np.dot(rowchol, rowchol.T), np.dot(colchol, colchol.T)
    )


def kron_normal_logpdf_cov(value, mu, covs, sigma, size=None):
    cov = kronecker(*covs).eval()
    if sigma is not None:
        cov += sigma**2 * np.eye(*cov.shape)
    return st.multivariate_normal.logpdf(value, mu, cov).sum()


def kron_normal_logpdf_chol(value, mu, chols, sigma, size=None):
    covs = [np.dot(chol, chol.T) for chol in chols]
    return kron_normal_logpdf_cov(value, mu, covs, sigma=sigma)


def kron_normal_logpdf_evd(value, mu, evds, sigma, size=None):
    covs = []
    for eigs, Q in evds:
        try:
            eigs = eigs.eval()
        except AttributeError:
            pass
        try:
            Q = Q.eval()
        except AttributeError:
            pass
        covs.append(np.dot(Q, np.dot(np.diag(eigs), Q.T)))
    return kron_normal_logpdf_cov(value, mu, covs, sigma)


def mvt_logpdf(value, nu, Sigma, mu=0):
    d = len(Sigma)
    dist = np.atleast_2d(value) - mu
    chol = np.linalg.cholesky(Sigma)
    trafo = np.linalg.solve(chol, dist.T).T
    logdet = np.log(np.diag(chol)).sum()

    lgamma = sp.gammaln
    norm = lgamma((nu + d) / 2.0) - 0.5 * d * np.log(nu * np.pi) - lgamma(nu / 2.0)
    logp_mvt = norm - logdet - (nu + d) / 2.0 * np.log1p((trafo * trafo).sum(-1) / nu)
    return logp_mvt.sum()


@pytest.fixture(scope="module")
def stickbreakingweights_logpdf():
    _value = at.vector()
    _alpha = at.scalar()
    _k = at.iscalar()
    _logp = logp(pm.StickBreakingWeights.dist(_alpha, _k), _value)
    core_fn = compile_pymc([_value, _alpha, _k], _logp)

    return np.vectorize(core_fn, signature="(n),(),()->()")


def PdMatrix(n):
    if n == 1:
        return PdMatrix1
    elif n == 2:
        return PdMatrix2
    elif n == 3:
        return PdMatrix3
    else:
        raise ValueError("n out of bounds")


PdMatrix1 = Domain([np.eye(1), [[0.5]]], edges=(None, None))

PdMatrix2 = Domain([np.eye(2), [[0.5, 0.05], [0.05, 4.5]]], edges=(None, None))

PdMatrix3 = Domain([np.eye(3), [[0.5, 0.1, 0], [0.1, 1, 0], [0, 0, 2.5]]], edges=(None, None))


PdMatrixChol1 = Domain([np.eye(1), [[0.001]]], edges=(None, None))
PdMatrixChol2 = Domain([np.eye(2), [[0.1, 0], [10, 1]]], edges=(None, None))
PdMatrixChol3 = Domain([np.eye(3), [[0.1, 0, 0], [10, 100, 0], [0, 1, 10]]], edges=(None, None))


def PdMatrixChol(n):
    if n == 1:
        return PdMatrixChol1
    elif n == 2:
        return PdMatrixChol2
    elif n == 3:
        return PdMatrixChol3
    else:
        raise ValueError("n out of bounds")


PdMatrixCholUpper1 = Domain([np.eye(1), [[0.001]]], edges=(None, None))
PdMatrixCholUpper2 = Domain([np.eye(2), [[0.1, 10], [0, 1]]], edges=(None, None))
PdMatrixCholUpper3 = Domain(
    [np.eye(3), [[0.1, 10, 0], [0, 100, 1], [0, 0, 10]]], edges=(None, None)
)


def PdMatrixCholUpper(n):
    if n == 1:
        return PdMatrixCholUpper1
    elif n == 2:
        return PdMatrixCholUpper2
    elif n == 3:
        return PdMatrixCholUpper3
    else:
        raise ValueError("n out of bounds")


def get_lkj_cases():
    """
    Log probabilities calculated using the formulas in:
    http://www.sciencedirect.com/science/article/pii/S0047259X09000876
    """
    tri = np.array([0.7, 0.0, -0.7])
    return [
        (tri, 1, 3, 1.5963125911388549),
        (tri, 3, 3, -7.7963493376312742),
        (tri, 0, 3, -np.inf),
        (np.array([1.1, 0.0, -0.7]), 1, 3, -np.inf),
        (np.array([0.7, 0.0, -1.1]), 1, 3, -np.inf),
    ]


LKJ_CASES = get_lkj_cases()


class TestMatchesScipy:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mvnormal(self, n):
        check_logp(
            pm.MvNormal,
            RealMatrix(5, n),
            {"mu": Vector(R, n), "tau": PdMatrix(n)},
            normal_logpdf_tau,
            extra_args={"size": 5},
        )
        check_logp(
            pm.MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "tau": PdMatrix(n)},
            normal_logpdf_tau,
        )
        check_logp(
            pm.MvNormal,
            RealMatrix(5, n),
            {"mu": Vector(R, n), "cov": PdMatrix(n)},
            normal_logpdf_cov,
            extra_args={"size": 5},
        )
        check_logp(
            pm.MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "cov": PdMatrix(n)},
            normal_logpdf_cov,
        )
        check_logp(
            pm.MvNormal,
            RealMatrix(5, n),
            {"mu": Vector(R, n), "chol": PdMatrixChol(n)},
            normal_logpdf_chol,
            decimal=select_by_precision(float64=6, float32=-1),
            extra_args={"size": 5},
        )
        check_logp(
            pm.MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "chol": PdMatrixChol(n)},
            normal_logpdf_chol,
            decimal=select_by_precision(float64=6, float32=0),
        )
        check_logp(
            pm.MvNormal,
            Vector(R, n),
            {"mu": Vector(R, n), "chol": PdMatrixCholUpper(n)},
            normal_logpdf_chol_upper,
            decimal=select_by_precision(float64=6, float32=0),
            extra_args={"lower": False},
        )

    @pytest.mark.skipif(
        condition=(aesara.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_mvnormal_indef(self):
        cov_val = np.array([[1, 0.5], [0.5, -2]])
        cov = at.matrix("cov")
        cov.tag.test_value = np.eye(2)
        mu = floatX(np.zeros(2))
        x = at.vector("x")
        x.tag.test_value = np.zeros(2)
        mvn_logp = logp(pm.MvNormal.dist(mu=mu, cov=cov), x)
        f_logp = aesara.function([cov, x], mvn_logp)
        with pytest.raises(ParameterValueError):
            f_logp(cov_val, np.ones(2))
        dlogp = at.grad(mvn_logp, cov)
        f_dlogp = aesara.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

        mvn_logp = logp(pm.MvNormal.dist(mu=mu, tau=cov), x)
        f_logp = aesara.function([cov, x], mvn_logp)
        with pytest.raises(ParameterValueError):
            f_logp(cov_val, np.ones(2))
        dlogp = at.grad(mvn_logp, cov)
        f_dlogp = aesara.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

    def test_mvnormal_init_fail(self):
        with pm.Model():
            with pytest.raises(ValueError):
                x = pm.MvNormal("x", mu=np.zeros(3), size=3)
            with pytest.raises(ValueError):
                x = pm.MvNormal("x", mu=np.zeros(3), cov=np.eye(3), tau=np.eye(3), size=3)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_matrixnormal(self, n):
        mat_scale = 1e3  # To reduce logp magnitude
        mean_scale = 0.1
        check_logp(
            pm.MatrixNormal,
            RealMatrix(n, n),
            {
                "mu": RealMatrix(n, n) * mean_scale,
                "rowcov": PdMatrix(n) * mat_scale,
                "colcov": PdMatrix(n) * mat_scale,
            },
            matrix_normal_logpdf_cov,
            decimal=select_by_precision(float64=5, float32=3),
        )
        check_logp(
            pm.MatrixNormal,
            RealMatrix(2, n),
            {
                "mu": RealMatrix(2, n) * mean_scale,
                "rowcov": PdMatrix(2) * mat_scale,
                "colcov": PdMatrix(n) * mat_scale,
            },
            matrix_normal_logpdf_cov,
            decimal=select_by_precision(float64=5, float32=3),
        )
        check_logp(
            pm.MatrixNormal,
            RealMatrix(3, n),
            {
                "mu": RealMatrix(3, n) * mean_scale,
                "rowchol": PdMatrixChol(3) * mat_scale,
                "colchol": PdMatrixChol(n) * mat_scale,
            },
            matrix_normal_logpdf_chol,
            decimal=select_by_precision(float64=5, float32=3),
        )
        check_logp(
            pm.MatrixNormal,
            RealMatrix(n, 3),
            {
                "mu": RealMatrix(n, 3) * mean_scale,
                "rowchol": PdMatrixChol(n) * mat_scale,
                "colchol": PdMatrixChol(3) * mat_scale,
            },
            matrix_normal_logpdf_chol,
            decimal=select_by_precision(float64=5, float32=3),
        )

    @pytest.mark.parametrize("n", [2, 3])
    @pytest.mark.parametrize("m", [3])
    @pytest.mark.parametrize("sigma", [None, 1])
    def test_kroneckernormal(self, n, m, sigma):
        np.random.seed(5)
        N = n * m
        covs = [RandomPdMatrix(n), RandomPdMatrix(m)]
        chols = list(map(np.linalg.cholesky, covs))
        evds = list(map(np.linalg.eigh, covs))
        dom = Domain([np.random.randn(N) * 0.1], edges=(None, None), shape=N)
        mu = Domain([np.random.randn(N) * 0.1], edges=(None, None), shape=N)

        std_args = {"mu": mu}
        cov_args = {"covs": covs}
        chol_args = {"chols": chols}
        evd_args = {"evds": evds}
        if sigma is not None and sigma != 0:
            std_args["sigma"] = Domain([sigma], edges=(None, None))
        else:
            for args in [cov_args, chol_args, evd_args]:
                args["sigma"] = sigma

        check_logp(
            pm.KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_cov,
            extra_args=cov_args,
            scipy_args=cov_args,
        )
        check_logp(
            pm.KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_chol,
            extra_args=chol_args,
            scipy_args=chol_args,
        )
        check_logp(
            pm.KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_evd,
            extra_args=evd_args,
            scipy_args=evd_args,
        )

        dom = Domain([np.random.randn(2, N) * 0.1], edges=(None, None), shape=(2, N))
        cov_args["size"] = 2
        chol_args["size"] = 2
        evd_args["size"] = 2

        check_logp(
            pm.KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_cov,
            extra_args=cov_args,
            scipy_args=cov_args,
        )
        check_logp(
            pm.KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_chol,
            extra_args=chol_args,
            scipy_args=chol_args,
        )
        check_logp(
            pm.KroneckerNormal,
            dom,
            std_args,
            kron_normal_logpdf_evd,
            extra_args=evd_args,
            scipy_args=evd_args,
        )

    @pytest.mark.parametrize("n", [1, 2])
    def test_mvt(self, n):
        check_logp(
            pm.MvStudentT,
            Vector(R, n),
            {"nu": Rplus, "Sigma": PdMatrix(n), "mu": Vector(R, n)},
            mvt_logpdf,
        )
        check_logp(
            pm.MvStudentT,
            RealMatrix(2, n),
            {"nu": Rplus, "Sigma": PdMatrix(n), "mu": Vector(R, n)},
            mvt_logpdf,
            extra_args={"size": 2},
        )

    @pytest.mark.parametrize("n", [2, 3])
    def test_wishart(self, n):
        with pytest.warns(UserWarning, match="Wishart distribution can currently not be used"):
            check_logp(
                pm.Wishart,
                PdMatrix(n),
                {"nu": Domain([0, 3, 4, np.inf], "int64"), "V": PdMatrix(n)},
                lambda value, nu, V: st.wishart.logpdf(value, int(nu), V),
            )

    @pytest.mark.parametrize("x,eta,n,lp", LKJ_CASES)
    def test_lkjcorr(self, x, eta, n, lp):
        with pm.Model() as model:
            pm.LKJCorr("lkj", eta=eta, n=n, transform=None)

        pt = {"lkj": x}
        decimals = select_by_precision(float64=6, float32=4)
        np.testing.assert_almost_equal(
            model.compile_logp()(pt), lp, decimal=decimals, err_msg=str(pt)
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_dirichlet(self, n):
        check_logp(
            pm.Dirichlet,
            Simplex(n),
            {"a": Vector(Rplus, n)},
            dirichlet_logpdf,
        )

    def test_dirichlet_invalid(self):
        # Test non-scalar invalid parameters/values
        value = np.array([[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]])

        invalid_dist = pm.Dirichlet.dist(a=[-1, 1, 2], size=2)
        with pytest.raises(ParameterValueError):
            pm.logp(invalid_dist, value).eval()

        value[1] -= 1
        valid_dist = pm.Dirichlet.dist(a=[1, 1, 1])
        assert np.all(np.isfinite(pm.logp(valid_dist, value).eval()) == np.array([True, False]))

    @pytest.mark.parametrize(
        "a",
        [
            ([2, 3, 5]),
            ([[2, 3, 5], [9, 19, 3]]),
            (np.abs(np.random.randn(2, 2, 4)) + 1),
        ],
    )
    @pytest.mark.parametrize("extra_size", [(2,), (1, 2), (2, 4, 3)])
    def test_dirichlet_vectorized(self, a, extra_size):
        a = floatX(np.array(a))
        size = extra_size + a.shape[:-1]

        dir = pm.Dirichlet.dist(a=a, size=size)
        vals = dir.eval()

        np.testing.assert_almost_equal(
            dirichlet_logpdf(vals, a),
            pm.logp(dir, vals).eval(),
            decimal=4,
            err_msg=f"vals={vals}",
        )

    @pytest.mark.parametrize("n", [2, 3])
    def test_multinomial(self, n):
        check_logp(
            pm.Multinomial,
            Vector(Nat, n),
            {"p": Simplex(n), "n": Nat},
            lambda value, n, p: st.multinomial.logpmf(value, n, p),
        )

    def test_multinomial_invalid_value(self):
        # Test passing non-scalar invalid parameters/values to an otherwise valid Multinomial,
        # evaluates to -inf
        value = np.array([[1, 2, 2], [3, -1, 0]])
        valid_dist = pm.Multinomial.dist(n=5, p=np.ones(3) / 3)
        assert np.all(np.isfinite(pm.logp(valid_dist, value).eval()) == np.array([True, False]))

    def test_multinomial_negative_p(self):
        # test passing a list/numpy with negative p raises an immediate error
        with pytest.raises(ValueError, match="[-1, 1, 1]"):
            with pm.Model() as model:
                x = pm.Multinomial("x", n=5, p=[-1, 1, 1])

    def test_multinomial_p_not_normalized(self):
        # test UserWarning is raised for p vals that sum to more than 1
        # and normaliation is triggered
        with pytest.warns(UserWarning, match="[5]"):
            with pm.Model() as m:
                x = pm.Multinomial("x", n=5, p=[1, 1, 1, 1, 1])
        # test stored p-vals have been normalised
        assert np.isclose(m.x.owner.inputs[4].sum().eval(), 1.0)

    def test_multinomial_negative_p_symbolic(self):
        # Passing symbolic negative p does not raise an immediate error, but evaluating
        # logp raises a ParameterValueError
        with pytest.raises(ParameterValueError):
            value = np.array([[1, 1, 1]])
            invalid_dist = pm.Multinomial.dist(n=1, p=at.as_tensor_variable([-1, 0.5, 0.5]))
            pm.logp(invalid_dist, value).eval()

    def test_multinomial_p_not_normalized_symbolic(self):
        # Passing symbolic p that do not add up to on does not raise any warning, but evaluating
        # logp raises a ParameterValueError
        with pytest.raises(ParameterValueError):
            value = np.array([[1, 1, 1]])
            invalid_dist = pm.Multinomial.dist(n=1, p=at.as_tensor_variable([1, 0.5, 0.5]))
            pm.logp(invalid_dist, value).eval()

    @pytest.mark.parametrize("n", [(10), ([10, 11]), ([[5, 6], [10, 11]])])
    @pytest.mark.parametrize(
        "p",
        [
            ([0.2, 0.3, 0.5]),
            ([[0.2, 0.3, 0.5], [0.9, 0.09, 0.01]]),
            (np.abs(np.random.randn(2, 2, 4))),
        ],
    )
    @pytest.mark.parametrize("extra_size", [(1,), (2,), (2, 3)])
    def test_multinomial_vectorized(self, n, p, extra_size):
        n = intX(np.array(n))
        p = floatX(np.array(p))
        p /= p.sum(axis=-1, keepdims=True)

        _, bcast_p = broadcast_params([n, p], ndims_params=[0, 1])
        size = extra_size + bcast_p.shape[:-1]

        mn = pm.Multinomial.dist(n=n, p=p, size=size)
        vals = mn.eval()

        np.testing.assert_almost_equal(
            st.multinomial.logpmf(vals, n, p),
            pm.logp(mn, vals).eval(),
            decimal=4,
            err_msg=f"vals={vals}",
        )

    def test_multinomial_zero_probs(self):
        # test multinomial accepts 0 probabilities / observations:
        mn = pm.Multinomial.dist(n=100, p=[0.0, 0.0, 1.0])
        assert pm.logp(mn, np.array([0, 0, 100])).eval() >= 0
        assert pm.logp(mn, np.array([50, 50, 0])).eval() == -np.inf

    @pytest.mark.parametrize("n", [2, 3])
    def test_dirichlet_multinomial(self, n):
        check_logp(
            pm.DirichletMultinomial,
            Vector(Nat, n),
            {"a": Vector(Rplus, n), "n": Nat},
            dirichlet_multinomial_logpmf,
        )

    def test_dirichlet_multinomial_invalid(self):
        # Test non-scalar invalid parameters/values
        value = np.array([[1, 2, 2], [4, 0, 1]])

        invalid_dist = pm.DirichletMultinomial.dist(n=5, a=[-1, 1, 1], size=2)
        with pytest.raises(ParameterValueError):
            pm.logp(invalid_dist, value).eval()

        value[1] -= 1
        valid_dist = pm.DirichletMultinomial.dist(n=5, a=[1, 1, 1])
        assert np.all(np.isfinite(pm.logp(valid_dist, value).eval()) == np.array([True, False]))

    def test_dirichlet_multinomial_matches_beta_binomial(self):
        a, b, n = 2, 1, 5
        ns = np.arange(n + 1)
        ns_dm = np.vstack((ns, n - ns)).T  # convert ns=1 to ns_dm=[1, 4], for all ns...

        bb = pm.BetaBinomial.dist(n=n, alpha=a, beta=b, size=2)
        bb_logp = logp(bb, ns).eval()

        dm = pm.DirichletMultinomial.dist(n=n, a=[a, b], size=2)
        dm_logp = logp(dm, ns_dm).eval().ravel()

        np.testing.assert_almost_equal(
            dm_logp,
            bb_logp,
            decimal=select_by_precision(float64=6, float32=3),
        )

    @pytest.mark.parametrize("n", [(10), ([10, 11]), ([[5, 6], [10, 11]])])
    @pytest.mark.parametrize(
        "a",
        [
            ([0.2, 0.3, 0.5]),
            ([[0.2, 0.3, 0.5], [0.9, 0.09, 0.01]]),
            (np.abs(np.random.randn(2, 2, 4))),
        ],
    )
    @pytest.mark.parametrize("extra_size", [(1,), (2,), (2, 3)])
    def test_dirichlet_multinomial_vectorized(self, n, a, extra_size):
        n = intX(np.array(n))
        a = floatX(np.array(a))

        _, bcast_a = broadcast_params([n, a], ndims_params=[0, 1])
        size = extra_size + bcast_a.shape[:-1]

        dm = pm.DirichletMultinomial.dist(n=n, a=a, size=size)
        vals = dm.eval()

        np.testing.assert_almost_equal(
            dirichlet_multinomial_logpmf(vals, n, a),
            pm.logp(dm, vals).eval(),
            decimal=4,
            err_msg=f"vals={vals}",
        )

    @pytest.mark.parametrize(
        "value,alpha,K,logp",
        [
            (np.array([5, 4, 3, 2, 1]) / 15, 0.5, 4, 1.5126301307277439),
            (np.tile(1, 13) / 13, 2, 12, 13.980045245672827),
            (np.array([0.001] * 10 + [0.99]), 0.1, 10, -22.971662448814723),
            (np.append(0.5 ** np.arange(1, 20), 0.5**20), 5, 19, 94.20462772778092),
            (
                (np.array([[7, 5, 3, 2], [19, 17, 13, 11]]) / np.array([[17], [60]])),
                2.5,
                3,
                np.array([1.29317672, 1.50126157]),
            ),
        ],
    )
    def test_stickbreakingweights_logp(self, value, alpha, K, logp):
        with pm.Model() as model:
            sbw = pm.StickBreakingWeights("sbw", alpha=alpha, K=K, transform=None)
        pt = {"sbw": value}
        np.testing.assert_almost_equal(
            pm.logp(sbw, value).eval(),
            logp,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(pt),
        )

    def test_stickbreakingweights_invalid(self):
        sbw = pm.StickBreakingWeights.dist(3.0, 3)
        sbw_wrong_K = pm.StickBreakingWeights.dist(3.0, 7)
        assert pm.logp(sbw, np.array([0.4, 0.3, 0.2, 0.15])).eval() == -np.inf
        assert pm.logp(sbw, np.array([1.1, 0.3, 0.2, 0.1])).eval() == -np.inf
        assert pm.logp(sbw, np.array([0.4, 0.3, 0.2, -0.1])).eval() == -np.inf
        assert pm.logp(sbw_wrong_K, np.array([0.4, 0.3, 0.2, 0.1])).eval() == -np.inf

    @pytest.mark.parametrize(
        "alpha,K",
        [
            (np.array([0.5, 1.0, 2.0]), 3),
            (np.arange(1, 7, dtype="float64").reshape(2, 3), 5),
        ],
    )
    def test_stickbreakingweights_vectorized(self, alpha, K, stickbreakingweights_logpdf):
        value = pm.StickBreakingWeights.dist(alpha, K).eval()
        with pm.Model():
            sbw = pm.StickBreakingWeights("sbw", alpha=alpha, K=K, transform=None)
        pt = {"sbw": value}
        np.testing.assert_almost_equal(
            pm.logp(sbw, value).eval(),
            stickbreakingweights_logpdf(value, alpha, K),
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(pt),
        )

    @pytest.mark.parametrize(
        "sparse, size",
        [(False, ()), (False, (1,)), (False, (4,)), (False, (4, 4, 4)), (True, ()), (True, (4,))],
        ids=str,
    )
    def test_car_logp(self, sparse, size):
        """
        Tests the log probability function for the CAR distribution by checking
        against Scipy's multivariate normal logpdf, up to an additive constant.
        The formula used by the CAR logp implementation omits several additive terms.
        """
        npr.seed(1)

        # d x d adjacency matrix for a square (d=4) of rook-adjacent sites
        W = np.array(
            [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
        )

        tau = 2
        alpha = 0.5
        mu = np.zeros(4)

        xs = npr.randn(*(size + mu.shape))

        # Compute CAR covariance matrix and resulting MVN logp
        D = W.sum(axis=0)
        prec = tau * (np.diag(D) - alpha * W)
        cov = np.linalg.inv(prec)
        scipy_logp = st.multivariate_normal.logpdf(xs, mu, cov)

        W = aesara.tensor.as_tensor_variable(W)
        if sparse:
            W = aesara.sparse.csr_from_dense(W)

        car_dist = pm.CAR.dist(mu, W, alpha, tau, size=size)
        car_logp = logp(car_dist, xs).eval()

        # Check to make sure that the CAR and MVN log PDFs are equivalent
        # up to an additive constant which is independent of the CAR parameters
        delta_logp = scipy_logp - car_logp

        # Check to make sure all the delta values are identical.
        tol = 1e-08
        if aesara.config.floatX == "float32":
            tol = 1e-5
        assert np.allclose(delta_logp - delta_logp[0], 0.0, atol=tol)


@pytest.mark.parametrize(
    "sparse",
    [False, True],
    ids=str,
)
def test_car_matrix_check(sparse):
    """
    Tests the check of W matrix symmetry in CARRV.make_node.
    """
    npr.seed(1)
    tau = 2
    alpha = 0.5
    mu = np.zeros(4)
    xs = npr.randn(*mu.shape)

    # non-symmetric matrix
    W = np.array(
        [[0.0, 1.0, 2.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
    )
    W = aesara.tensor.as_tensor_variable(W)
    if sparse:
        W = aesara.sparse.csr_from_dense(W)

    car_dist = pm.CAR.dist(mu, W, alpha, tau)
    with pytest.raises(AssertionError, match="W must be a symmetric adjacency matrix"):
        logp(car_dist, xs).eval()

    # W.ndim != 2
    if not sparse:
        W = np.array([0.0, 1.0, 2.0, 0.0])
        W = aesara.tensor.as_tensor_variable(W)
        with pytest.raises(ValueError, match="W must be a matrix"):
            car_dist = pm.CAR.dist(mu, W, alpha, tau)


class TestLKJCholeskCov:
    def test_dist(self):
        sd_dist = pm.Exponential.dist(1, size=(10, 3))
        x = pm.LKJCholeskyCov.dist(n=3, eta=1, sd_dist=sd_dist, size=10, compute_corr=False)
        assert x.eval().shape == (10, 6)

        sd_dist = pm.Exponential.dist(1, size=3)
        chol, corr, stds = pm.LKJCholeskyCov.dist(n=3, eta=1, sd_dist=sd_dist)
        assert chol.eval().shape == (3, 3)
        assert corr.eval().shape == (3, 3)
        assert stds.eval().shape == (3,)

    def test_sd_dist_distribution(self):
        with pm.Model() as m:
            sd_dist = at.constant([1, 2, 3])
            with pytest.raises(TypeError, match="^sd_dist must be a scalar or vector distribution"):
                x = pm.LKJCholeskyCov("x", n=3, eta=1, sd_dist=sd_dist)

    def test_sd_dist_registered(self):
        with pm.Model() as m:
            sd_dist = pm.Exponential("sd_dist", 1, size=3)
            with pytest.raises(
                ValueError, match="The dist sd_dist was already registered in the current model"
            ):
                x = pm.LKJCholeskyCov("x", n=3, eta=1, sd_dist=sd_dist)

    def test_no_warning_logp(self):
        # Check that calling logp of a model with LKJCholeskyCov does not issue any warnings
        # due to the RandomVariable in the graph
        with pm.Model() as m:
            sd_dist = pm.Exponential.dist(1, size=3)
            x = pm.LKJCholeskyCov("x", n=3, eta=1, sd_dist=sd_dist)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m.logp()

    @pytest.mark.parametrize(
        "sd_dist",
        [
            pm.Exponential.dist(1),
            pm.MvNormal.dist(np.ones(3), np.eye(3)),
        ],
    )
    @pytest.mark.parametrize(
        "size, shape",
        [
            ((10,), None),
            (None, (10, 6)),
            (None, (10, ...)),
        ],
    )
    def test_sd_dist_automatically_resized(self, sd_dist, size, shape):
        x = pm.LKJCholeskyCov.dist(
            n=3, eta=1, sd_dist=sd_dist, size=size, shape=shape, compute_corr=False
        )
        resized_sd_dist = x.owner.inputs[-1]
        assert resized_sd_dist.eval().shape == (10, 3)
        # LKJCov has support shape `(n * (n+1)) // 2`
        assert x.eval().shape == (10, 6)


# Used for MvStudentT moment test
rand1d = np.random.rand(2)
rand2d = np.random.rand(2, 3)


class TestMoments:
    @pytest.mark.parametrize(
        "p, n, size, expected",
        [
            (np.array([0.25, 0.25, 0.25, 0.25]), 1, None, np.array([1, 0, 0, 0])),
            (np.array([0.3, 0.6, 0.05, 0.05]), 2, None, np.array([1, 1, 0, 0])),
            (np.array([0.3, 0.6, 0.05, 0.05]), 10, None, np.array([4, 6, 0, 0])),
            (
                np.array([[0.3, 0.6, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]]),
                10,
                None,
                np.array([[4, 6, 0, 0], [4, 2, 2, 2]]),
            ),
            (
                np.array([0.3, 0.6, 0.05, 0.05]),
                np.array([2, 10]),
                (1, 2),
                np.array([[[1, 1, 0, 0], [4, 6, 0, 0]]]),
            ),
            (
                np.array([[0.25, 0.25, 0.25, 0.25], [0.26, 0.26, 0.26, 0.22]]),
                np.array([1, 10]),
                None,
                np.array([[1, 0, 0, 0], [2, 3, 3, 2]]),
            ),
            (
                np.array([[0.25, 0.25, 0.25, 0.25], [0.26, 0.26, 0.26, 0.22]]),
                np.array([1, 10]),
                (3, 2),
                np.full((3, 2, 4), [[1, 0, 0, 0], [2, 3, 3, 2]]),
            ),
        ],
    )
    def test_multinomial_moment(self, p, n, size, expected):
        with pm.Model() as model:
            pm.Multinomial("x", n=n, p=p, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "a, size, expected",
        [
            (
                np.array([2, 3, 5, 7, 11]),
                None,
                np.array([2, 3, 5, 7, 11]) / 28,
            ),
            (
                np.array([[1, 2, 3], [5, 6, 7]]),
                None,
                np.array([[1, 2, 3], [5, 6, 7]]) / np.array([6, 18])[..., np.newaxis],
            ),
            (
                np.array([[1, 2, 3], [5, 6, 7]]),
                (7, 2),
                np.apply_along_axis(
                    lambda x: np.divide(x, np.array([6, 18])),
                    1,
                    np.broadcast_to([[1, 2, 3], [5, 6, 7]], shape=[7, 2, 3]),
                ),
            ),
            (
                np.full(shape=np.array([7, 3]), fill_value=np.array([13, 17, 19])),
                (11, 5, 7),
                np.broadcast_to([13, 17, 19], shape=[11, 5, 7, 3]) / 49,
            ),
        ],
    )
    def test_dirichlet_moment(self, a, size, expected):
        with pm.Model() as model:
            pm.Dirichlet("x", a=a, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, cov, size, expected",
        [
            (np.ones(1), np.identity(1), None, np.ones(1)),
            (np.ones(3), np.identity(3), None, np.ones(3)),
            (np.ones((2, 2)), np.identity(2), None, np.ones((2, 2))),
            (np.array([1, 0, 3.0]), np.identity(3), None, np.array([1, 0, 3.0])),
            (np.array([1, 0, 3.0]), np.identity(3), (4, 2), np.full((4, 2, 3), [1, 0, 3.0])),
            (
                np.array([1, 3.0]),
                np.identity(2),
                5,
                np.full((5, 2), [1, 3.0]),
            ),
            (
                np.array([1, 3.0]),
                np.array([[1.0, 0.5], [0.5, 2]]),
                (4, 5),
                np.full((4, 5, 2), [1, 3.0]),
            ),
            (
                np.array([[3.0, 5], [1, 4]]),
                np.identity(2),
                (4, 5, 2),
                np.full((4, 5, 2, 2), [[3.0, 5], [1, 4]]),
            ),
        ],
    )
    def test_mv_normal_moment(self, mu, cov, size, expected):
        with pm.Model() as model:
            x = pm.MvNormal("x", mu=mu, cov=cov, size=size)

        # MvNormal logp is only implemented for up to 2D variables
        assert_moment_is_expected(model, expected, check_finite_logp=x.ndim < 3)

    @pytest.mark.parametrize(
        "mu, size, expected",
        [
            (
                np.array([1, 0, 3.0, 4]),
                None,
                np.array([1, 0, 3.0, 4]),
            ),
            (np.array([1, 0, 3.0, 4]), 6, np.full((6, 4), [1, 0, 3.0, 4])),
            (np.array([1, 0, 3.0, 4]), (5, 3), np.full((5, 3, 4), [1, 0, 3.0, 4])),
            (
                np.array([[3.0, 5, 2, 1], [1, 4, 0.5, 9]]),
                (4, 5, 2),
                np.full((4, 5, 2, 4), [[3.0, 5, 2, 1], [1, 4, 0.5, 9]]),
            ),
        ],
    )
    def test_car_moment(self, mu, size, expected):
        W = np.array(
            [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
        )
        tau = 2
        alpha = 0.5
        with pm.Model() as model:
            pm.CAR("x", mu=mu, W=W, alpha=alpha, tau=tau, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "nu, mu, cov, size, expected",
        [
            (2, np.ones(1), np.eye(1), None, np.ones(1)),
            (2, rand1d, np.eye(2), None, rand1d),
            (2, rand1d, np.eye(2), 2, np.full((2, 2), rand1d)),
            (2, rand1d, np.eye(2), (2, 5), np.full((2, 5, 2), rand1d)),
            (2, rand2d, np.eye(3), None, rand2d),
            (2, rand2d, np.eye(3), (2, 2), np.full((2, 2, 3), rand2d)),
            (2, rand2d, np.eye(3), (2, 5, 2), np.full((2, 5, 2, 3), rand2d)),
        ],
    )
    def test_mvstudentt_moment(self, nu, mu, cov, size, expected):
        with pm.Model() as model:
            x = pm.MvStudentT("x", nu=nu, mu=mu, cov=cov, size=size)

        # MvStudentT logp is only impemented for up to 2D variables
        assert_moment_is_expected(model, expected, check_finite_logp=x.ndim < 3)

    @pytest.mark.parametrize(
        "mu, rowchol, colchol, size, expected",
        [
            (np.ones((1, 1)), np.eye(1), np.eye(1), None, np.ones((1, 1))),
            (np.ones((1, 1)), np.eye(2), np.eye(3), None, np.ones((2, 3))),
            (rand2d, np.eye(2), np.eye(3), None, rand2d),
            (rand2d, np.eye(2), np.eye(3), 2, np.full((2, 2, 3), rand2d)),
            (rand2d, np.eye(2), np.eye(3), (2, 5), np.full((2, 5, 2, 3), rand2d)),
        ],
    )
    def test_matrixnormal_moment(self, mu, rowchol, colchol, size, expected):
        with pm.Model() as model:
            x = pm.MatrixNormal("x", mu=mu, rowchol=rowchol, colchol=colchol, size=size)

        # MatrixNormal logp is only implemented for 2d values
        check_logp = x.ndim == 2
        assert_moment_is_expected(model, expected, check_finite_logp=check_logp)

    @pytest.mark.parametrize(
        "alpha, K, size, expected",
        [
            (3, 11, None, np.append((3 / 4) ** np.arange(11) * 1 / 4, (3 / 4) ** 11)),
            (5, 19, None, np.append((5 / 6) ** np.arange(19) * 1 / 6, (5 / 6) ** 19)),
            (
                1,
                7,
                (13,),
                np.full(
                    shape=(13, 8),
                    fill_value=np.append((1 / 2) ** np.arange(7) * 1 / 2, (1 / 2) ** 7),
                ),
            ),
            (
                0.5,
                5,
                (3, 5, 7),
                np.full(
                    shape=(3, 5, 7, 6),
                    fill_value=np.append((1 / 3) ** np.arange(5) * 2 / 3, (1 / 3) ** 5),
                ),
            ),
            (
                np.array([1, 3]),
                11,
                None,
                np.array(
                    [
                        np.append((1 / 2) ** np.arange(11) * 1 / 2, (1 / 2) ** 11),
                        np.append((3 / 4) ** np.arange(11) * 1 / 4, (3 / 4) ** 11),
                    ]
                ),
            ),
            (
                np.array([1, 3, 5]),
                9,
                (5, 3),
                np.full(
                    shape=(5, 3, 10),
                    fill_value=np.array(
                        [
                            np.append((1 / 2) ** np.arange(9) * 1 / 2, (1 / 2) ** 9),
                            np.append((3 / 4) ** np.arange(9) * 1 / 4, (3 / 4) ** 9),
                            np.append((5 / 6) ** np.arange(9) * 1 / 6, (5 / 6) ** 9),
                        ]
                    ),
                ),
            ),
        ],
    )
    def test_stickbreakingweights_moment(self, alpha, K, size, expected):
        with pm.Model() as model:
            pm.StickBreakingWeights("x", alpha=alpha, K=K, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "mu, covs, size, expected",
        [
            (np.ones(1), [np.identity(1), np.identity(1)], None, np.ones(1)),
            (np.ones(6), [np.identity(2), np.identity(3)], 5, np.ones((5, 6))),
            (np.zeros(6), [np.identity(2), np.identity(3)], 6, np.zeros((6, 6))),
            (np.zeros(3), [np.identity(3), np.identity(1)], 6, np.zeros((6, 3))),
            (
                np.array([1, 2, 3, 4]),
                [
                    np.array([[1.0, 0.5], [0.5, 2]]),
                    np.array([[1.0, 0.4], [0.4, 2]]),
                ],
                2,
                np.array(
                    [
                        [1, 2, 3, 4],
                        [1, 2, 3, 4],
                    ]
                ),
            ),
        ],
    )
    def test_kronecker_normal_moment(self, mu, covs, size, expected):
        with pm.Model() as model:
            pm.KroneckerNormal("x", mu=mu, covs=covs, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "n, eta, size, expected",
        [
            (3, 1, None, np.zeros(3)),
            (5, 1, None, np.zeros(10)),
            (3, 1, 1, np.zeros((1, 3))),
            (5, 1, (2, 3), np.zeros((2, 3, 10))),
        ],
    )
    def test_lkjcorr_moment(self, n, eta, size, expected):
        with pm.Model() as model:
            pm.LKJCorr("x", n=n, eta=eta, size=size)
        assert_moment_is_expected(model, expected)

    @pytest.mark.parametrize(
        "n, eta, size, expected",
        [
            (3, 1, None, np.array([1, 0, 1, 0, 0, 1])),
            (4, 1, None, np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])),
            (3, 1, 1, np.array([[1, 0, 1, 0, 0, 1]])),
            (
                4,
                1,
                (2, 3),
                np.full((2, 3, 10), np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])),
            ),
        ],
    )
    def test_lkjcholeskycov_moment(self, n, eta, size, expected):
        with pm.Model() as model:
            sd_dist = pm.Exponential.dist(1, size=(*to_tuple(size), n))
            pm.LKJCholeskyCov("x", n=n, eta=eta, sd_dist=sd_dist, size=size, compute_corr=False)
        assert_moment_is_expected(model, expected, check_finite_logp=size is None)

    @pytest.mark.parametrize(
        "a, n, size, expected",
        [
            (np.array([2, 2, 2, 2]), 1, None, np.array([1, 0, 0, 0])),
            (np.array([3, 6, 0.5, 0.5]), 2, None, np.array([1, 1, 0, 0])),
            (np.array([30, 60, 5, 5]), 10, None, np.array([4, 6, 0, 0])),
            (
                np.array([[30, 60, 5, 5], [26, 26, 26, 22]]),
                10,
                (1, 2),
                np.array([[[4, 6, 0, 0], [2, 3, 3, 2]]]),
            ),
            (
                np.array([26, 26, 26, 22]),
                np.array([1, 10]),
                None,
                np.array([[1, 0, 0, 0], [2, 3, 3, 2]]),
            ),
            (
                np.array([[26, 26, 26, 22]]),  # Dim: 1 x 4
                np.array([[1], [10]]),  # Dim: 2 x 1
                (2, 1, 2, 1),
                np.full(
                    (2, 1, 2, 1, 4),
                    np.array([[[1, 0, 0, 0]], [[2, 3, 3, 2]]]),  # Dim: 2 x 1 x 4
                ),
            ),
        ],
    )
    def test_dirichlet_multinomial_moment(self, a, n, size, expected):
        with pm.Model() as model:
            pm.DirichletMultinomial("x", n=n, a=a, size=size)
        assert_moment_is_expected(model, expected)
