#   Copyright 2023 The PyMC Developers
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

import functools as ft
import re
import warnings

import numpy as np
import numpy.random as npr
import numpy.testing as npt
import pytensor
import pytensor.tensor as pt
import pytest
import scipy.special as sp
import scipy.stats as st

from pytensor.tensor import TensorVariable
from pytensor.tensor.random.utils import broadcast_params

import pymc as pm

from pymc.distributions.multivariate import (
    _LKJCholeskyCov,
    _OrderedMultinomial,
    posdef,
    quaddist_matrix,
)
from pymc.distributions.shape_utils import change_dist_size, to_tuple
from pymc.logprob.basic import logp
from pymc.logprob.utils import ParameterValueError
from pymc.math import kronecker
from pymc.pytensorf import compile_pymc, floatX, intX
from pymc.sampling.forward import draw
from pymc.testing import (
    BaseTestDistributionRandom,
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
    continuous_random_tester,
    seeded_numpy_distribution_builder,
    select_by_precision,
)


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
    _value = pt.vector()
    _alpha = pt.scalar()
    _k = pt.iscalar()
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
        condition=(pytensor.config.floatX == "float32"),
        reason="Fails on float32 due to inf issues",
    )
    def test_mvnormal_indef(self):
        cov_val = np.array([[1, 0.5], [0.5, -2]])
        cov = pt.matrix("cov")
        cov.tag.test_value = np.eye(2)
        mu = floatX(np.zeros(2))
        x = pt.vector("x")
        x.tag.test_value = np.zeros(2)
        mvn_logp = logp(pm.MvNormal.dist(mu=mu, cov=cov), x)
        f_logp = pytensor.function([cov, x], mvn_logp)
        with pytest.raises(ParameterValueError):
            f_logp(cov_val, np.ones(2))
        dlogp = pt.grad(mvn_logp, cov)
        f_dlogp = pytensor.function([cov, x], dlogp)
        assert not np.all(np.isfinite(f_dlogp(cov_val, np.ones(2))))

        mvn_logp = logp(pm.MvNormal.dist(mu=mu, tau=cov), x)
        f_logp = pytensor.function([cov, x], mvn_logp)
        with pytest.raises(ParameterValueError):
            f_logp(cov_val, np.ones(2))
        dlogp = pt.grad(mvn_logp, cov)
        f_dlogp = pytensor.function([cov, x], dlogp)
        try:
            res = f_dlogp(cov_val, np.ones(2))
        except ValueError:
            pass  # Op raises internally
        else:
            assert not np.all(np.isfinite(res))  # Otherwise, should return nan

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

        point = {"lkj": x}
        decimals = select_by_precision(float64=6, float32=4)
        npt.assert_almost_equal(
            model.compile_logp()(point), lp, decimal=decimals, err_msg=str(point)
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

        npt.assert_almost_equal(
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
        with pytest.raises(ValueError, match="Negative `p` parameters are not valid"):
            with pm.Model() as model:
                x = pm.Multinomial("x", n=5, p=[-1, 1, 1])

    def test_multinomial_p_not_normalized(self):
        # test UserWarning is raised for p vals that sum to more than 1
        # and normaliation is triggered
        with pytest.warns(UserWarning, match="They will be automatically rescaled"):
            with pm.Model() as m:
                x = pm.Multinomial("x", n=5, p=[1, 1, 1, 1, 1])
        # test stored p-vals have been normalised
        assert np.isclose(m.x.owner.inputs[4].sum().eval(), 1.0)

    def test_multinomial_negative_p_symbolic(self):
        # Passing symbolic negative p does not raise an immediate error, but evaluating
        # logp raises a ParameterValueError
        value = np.array([[1, 1, 1]])

        x = pt.scalar("x")
        invalid_dist = pm.Multinomial.dist(n=1, p=[x, x, x])

        with pytest.raises(ParameterValueError):
            pm.logp(invalid_dist, value).eval({x: -1 / 3})

    def test_multinomial_p_not_normalized_symbolic(self):
        # Passing symbolic p that do not add up to on does not raise any warning, but evaluating
        # logp raises a ParameterValueError
        value = np.array([[1, 1, 1]])

        x = pt.scalar("x")
        invalid_dist = pm.Multinomial.dist(n=1, p=(x, x, x))
        with pytest.raises(ParameterValueError):
            pm.logp(invalid_dist, value).eval({x: 0.5})

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

        npt.assert_almost_equal(
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

    def test_ordered_multinomial_probs(self):
        with pm.Model() as m:
            pm.OrderedMultinomial("om_p", n=1000, cutpoints=np.array([-2, 0, 2]), eta=0)
            pm.OrderedMultinomial(
                "om_no_p", n=1000, cutpoints=np.array([-2, 0, 2]), eta=0, compute_p=False
            )
        assert len(m.deterministics) == 1

        x = pm.OrderedMultinomial.dist(n=1000, cutpoints=np.array([-2, 0, 2]), eta=0)
        assert isinstance(x, TensorVariable)

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

        npt.assert_almost_equal(
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

        npt.assert_almost_equal(
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
        point = {"sbw": value}
        npt.assert_almost_equal(
            pm.logp(sbw, value).eval(),
            logp,
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(point),
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
        point = {"sbw": value}
        npt.assert_almost_equal(
            pm.logp(sbw, value).eval(),
            stickbreakingweights_logpdf(value, alpha, K),
            decimal=select_by_precision(float64=6, float32=2),
            err_msg=str(point),
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

        W = pytensor.tensor.as_tensor_variable(W)
        if sparse:
            W = pytensor.sparse.csr_from_dense(W)

        car_dist = pm.CAR.dist(mu, W, alpha, tau, size=size)
        car_logp = logp(car_dist, xs).eval()

        # Check to make sure that the CAR and MVN log PDFs are equivalent
        # up to an additive constant which is independent of the CAR parameters
        delta_logp = scipy_logp - car_logp

        # Check to make sure all the delta values are identical.
        tol = 1e-08
        if pytensor.config.floatX == "float32":
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
    W = pytensor.tensor.as_tensor_variable(W)
    if sparse:
        W = pytensor.sparse.csr_from_dense(W)

    car_dist = pm.CAR.dist(mu, W, alpha, tau)
    with pytest.raises(AssertionError, match="W must be a symmetric adjacency matrix"):
        logp(car_dist, xs).eval()

    # W.ndim != 2
    if not sparse:
        W = np.array([0.0, 1.0, 2.0, 0.0])
        W = pytensor.tensor.as_tensor_variable(W)
        with pytest.raises(ValueError, match="W must be a matrix"):
            car_dist = pm.CAR.dist(mu, W, alpha, tau)


@pytest.mark.parametrize("alpha", [1, -1])
def test_car_alpha_bounds(alpha):
    """
    Tests the check that -1 < alpha < 1
    """

    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    tau = 1
    mu = np.array([0, 0, 0])
    values = np.array([-0.5, 0, 0.5])

    car_dist = pm.CAR.dist(W=W, alpha=alpha, mu=mu, tau=tau)

    with pytest.raises(ValueError, match="the domain of alpha is: -1 < alpha < 1"):
        pm.draw(car_dist)

    with pytest.raises(ValueError, match="-1 < alpha < 1, tau > 0"):
        pm.logp(car_dist, values).eval()


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
            sd_dist = pt.constant([1, 2, 3])
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

    def test_change_dist_size(self):
        x1 = pm.LKJCholeskyCov.dist(
            n=3, eta=1, sd_dist=pm.Dirichlet.dist(np.ones(3)), size=(5,), compute_corr=False
        )
        x2 = change_dist_size(x1, new_size=(10, 3), expand=False)
        x3 = change_dist_size(x2, new_size=(3,), expand=True)

        draw_x1, draw_x2, draw_x3 = pm.draw([x1, x2, x3])
        assert draw_x1.shape == (5, 6)
        assert draw_x2.shape == (10, 3, 6)
        assert draw_x3.shape == (3, 10, 3, 6)


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
        "shape, n_zerosum_axes, expected",
        [
            ((2, 5), None, np.zeros((2, 5))),
            ((2, 5, 6), 2, np.zeros((2, 5, 6))),
            ((2, 5, 6), 3, np.zeros((2, 5, 6))),
        ],
    )
    def test_zerosum_normal_moment(self, shape, n_zerosum_axes, expected):
        with pm.Model() as model:
            pm.ZeroSumNormal("x", shape=shape, n_zerosum_axes=n_zerosum_axes)
        assert_moment_is_expected(model, expected)

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
        "W, expected",
        [
            (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), np.array([0, 0, 0])),
            (np.array([[0, 1], [1, 0]]), np.array([0, 0])),
        ],
    )
    def test_icar_moment(self, W, expected):
        with pm.Model() as model:
            RV = pm.ICAR("x", W=W)
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
            x = pm.MvStudentT("x", nu=nu, mu=mu, scale=cov, size=size)

        # MvStudentT logp is only implemented for up to 2D variables
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


class TestMvNormalCov(BaseTestDistributionRandom):
    pymc_dist = pm.MvNormal
    pymc_dist_params = {
        "mu": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "mu": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    sizes_to_check = [None, (1), (2, 3)]
    sizes_expected = [(2,), (1, 2), (2, 3, 2)]
    reference_dist_params = {
        "mean": np.array([1.0, 2.0]),
        "cov": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    reference_dist = seeded_numpy_distribution_builder("multivariate_normal")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
        "check_mu_broadcast_helper",
    ]

    def check_mu_broadcast_helper(self):
        """Test that mu is broadcasted to the shape of cov"""
        x = pm.MvNormal.dist(mu=1, cov=np.eye(3))
        mu = x.owner.inputs[3]
        assert mu.eval().shape == (3,)

        x = pm.MvNormal.dist(mu=np.ones(1), cov=np.eye(3))
        mu = x.owner.inputs[3]
        assert mu.eval().shape == (3,)

        x = pm.MvNormal.dist(mu=np.ones((1, 1)), cov=np.eye(3))
        mu = x.owner.inputs[3]
        assert mu.eval().shape == (1, 3)

        x = pm.MvNormal.dist(mu=np.ones((10, 1)), cov=np.eye(3))
        mu = x.owner.inputs[3]
        assert mu.eval().shape == (10, 3)

        # Cov is artificially limited to being 2D
        # x = pm.MvNormal.dist(mu=np.ones((10, 1)), cov=np.full((2, 3, 3), np.eye(3)))
        # mu = x.owner.inputs[3]
        # assert mu.eval().shape == (10, 2, 3)


class TestMvNormalChol(BaseTestDistributionRandom):
    pymc_dist = pm.MvNormal
    pymc_dist_params = {
        "mu": np.array([1.0, 2.0]),
        "chol": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "mu": np.array([1.0, 2.0]),
        "cov": quaddist_matrix(chol=pymc_dist_params["chol"]).eval(),
    }
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestMvNormalTau(BaseTestDistributionRandom):
    pymc_dist = pm.MvNormal
    pymc_dist_params = {
        "mu": np.array([1.0, 2.0]),
        "tau": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "mu": np.array([1.0, 2.0]),
        "cov": quaddist_matrix(tau=pymc_dist_params["tau"]).eval(),
    }
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestMvNormalMisc:
    def test_with_chol_rv(self):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, size=3)
            sd_dist = pm.Exponential.dist(1.0, size=3)
            # pylint: disable=unpacking-non-sequence
            chol, _, _ = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            # pylint: enable=unpacking-non-sequence
            mv = pm.MvNormal("mv", mu, chol=chol, size=4)
            prior = pm.sample_prior_predictive(samples=10, return_inferencedata=False)

        assert prior["mv"].shape == (10, 4, 3)

    def test_with_cov_rv(
        self,
    ):
        with pm.Model() as model:
            mu = pm.Normal("mu", 0.0, 1.0, shape=3)
            sd_dist = pm.Exponential.dist(1.0, shape=3)
            # pylint: disable=unpacking-non-sequence
            chol, corr, stds = pm.LKJCholeskyCov(
                "chol_cov", n=3, eta=2, sd_dist=sd_dist, compute_corr=True
            )
            # pylint: enable=unpacking-non-sequence
            mv = pm.MvNormal("mv", mu, cov=pm.math.dot(chol, chol.T), size=4)
            prior = pm.sample_prior_predictive(samples=10, return_inferencedata=False)

        assert prior["mv"].shape == (10, 4, 3)

    def test_issue_3758(self):
        np.random.seed(42)
        ndim = 50
        with pm.Model() as model:
            a = pm.Normal("a", sigma=100, shape=ndim)
            b = pm.Normal("b", mu=a, sigma=1, shape=ndim)
            c = pm.MvNormal("c", mu=a, chol=np.linalg.cholesky(np.eye(ndim)), shape=ndim)
            d = pm.MvNormal("d", mu=a, cov=np.eye(ndim), shape=ndim)
            samples = pm.sample_prior_predictive(1000, return_inferencedata=False)

        for var in "abcd":
            assert not np.isnan(np.std(samples[var]))

        for var in "bcd":
            std = np.std(samples[var] - samples["a"])
            npt.assert_allclose(std, 1, rtol=2e-2)

    def test_issue_3829(self):
        with pm.Model() as model:
            x = pm.MvNormal("x", mu=np.zeros(5), cov=np.eye(5), shape=(2, 5))
            trace_pp = pm.sample_prior_predictive(50, return_inferencedata=False)

        assert np.shape(trace_pp["x"][0]) == (2, 5)

    def test_issue_3706(self):
        N = 10
        Sigma = np.eye(2)

        with pm.Model() as model:
            X = pm.MvNormal("X", mu=np.zeros(2), cov=Sigma, shape=(N, 2))
            betas = pm.Normal("betas", 0, 1, shape=2)
            y = pm.Deterministic("y", pm.math.dot(X, betas))

            prior_pred = pm.sample_prior_predictive(1, return_inferencedata=False)

        assert prior_pred["X"].shape == (1, N, 2)


class TestZeroSumNormal:
    coords = {
        "regions": ["a", "b", "c"],
        "answers": ["yes", "no", "whatever", "don't understand question"],
    }

    def assert_zerosum_axes(self, random_samples, axes_to_check, check_zerosum_axes=True):
        if check_zerosum_axes:
            for ax in axes_to_check:
                assert np.isclose(
                    random_samples.mean(axis=ax), 0
                ).all(), f"{ax} is a zerosum_axis but is not summing to 0 across all samples."
        else:
            for ax in axes_to_check:
                assert not np.isclose(
                    random_samples.mean(axis=ax), 0
                ).all(), f"{ax} is not a zerosum_axis, but is nonetheless summing to 0 across all samples."

    @pytest.mark.parametrize(
        "dims, n_zerosum_axes",
        [
            (("regions", "answers"), None),
            (("regions", "answers"), 1),
            (("regions", "answers"), 2),
        ],
    )
    def test_zsn_dims(self, dims, n_zerosum_axes):
        with pm.Model(coords=self.coords) as m:
            v = pm.ZeroSumNormal("v", dims=dims, n_zerosum_axes=n_zerosum_axes)
            s = pm.sample(10, chains=1, tune=100)

        # to test forward graph
        random_samples = pm.draw(v, draws=10)

        assert s.posterior.v.shape == (
            1,
            10,
            len(self.coords["regions"]),
            len(self.coords["answers"]),
        )

        ndim_supp = v.owner.op.ndim_supp
        n_zerosum_axes = np.arange(-ndim_supp, 0)
        nonzero_axes = np.arange(v.ndim - ndim_supp)
        for samples in [
            s.posterior.v,
            random_samples,
        ]:
            self.assert_zerosum_axes(samples, n_zerosum_axes)
            self.assert_zerosum_axes(samples, nonzero_axes, check_zerosum_axes=False)

    @pytest.mark.parametrize(
        "n_zerosum_axes",
        (None, 1, 2),
    )
    def test_zsn_shape(self, n_zerosum_axes):
        shape = (len(self.coords["regions"]), len(self.coords["answers"]))

        with pm.Model(coords=self.coords) as m:
            v = pm.ZeroSumNormal("v", shape=shape, n_zerosum_axes=n_zerosum_axes)
            s = pm.sample(10, chains=1, tune=100)

        # to test forward graph
        random_samples = pm.draw(v, draws=10)

        assert s.posterior.v.shape == (
            1,
            10,
            len(self.coords["regions"]),
            len(self.coords["answers"]),
        )

        ndim_supp = v.owner.op.ndim_supp
        n_zerosum_axes = np.arange(-ndim_supp, 0)
        nonzero_axes = np.arange(v.ndim - ndim_supp)
        for samples in [
            s.posterior.v,
            random_samples,
        ]:
            self.assert_zerosum_axes(samples, n_zerosum_axes)
            self.assert_zerosum_axes(samples, nonzero_axes, check_zerosum_axes=False)

    @pytest.mark.parametrize(
        "error, match, shape, support_shape, n_zerosum_axes",
        [
            (
                ValueError,
                "Number of shape dimensions is too small for ndim_supp of 4",
                (3, 4, 5),
                None,
                4,
            ),
            (AssertionError, "does not match", (3, 4), (3,), None),  # support_shape should be 4
            (
                AssertionError,
                "does not match",
                (3, 4),
                (3, 4),
                None,
            ),  # doesn't work because n_zerosum_axes = 1 by default
        ],
    )
    def test_zsn_fail_axis(self, error, match, shape, support_shape, n_zerosum_axes):
        with pytest.raises(error, match=match):
            with pm.Model() as m:
                _ = pm.ZeroSumNormal(
                    "v", shape=shape, support_shape=support_shape, n_zerosum_axes=n_zerosum_axes
                )

    @pytest.mark.parametrize(
        "shape, support_shape",
        [
            (None, (3, 4)),
            ((3, 4), (3, 4)),
        ],
    )
    def test_zsn_support_shape(self, shape, support_shape):
        with pm.Model() as m:
            v = pm.ZeroSumNormal("v", shape=shape, support_shape=support_shape, n_zerosum_axes=2)

        random_samples = pm.draw(v, draws=10)
        n_zerosum_axes = np.arange(-2, 0)
        self.assert_zerosum_axes(random_samples, n_zerosum_axes)

    @pytest.mark.parametrize(
        "n_zerosum_axes",
        [1, 2],
    )
    def test_zsn_change_dist_size(self, n_zerosum_axes):
        base_dist = pm.ZeroSumNormal.dist(shape=(4, 9), n_zerosum_axes=n_zerosum_axes)
        random_samples = pm.draw(base_dist, draws=100)

        n_zerosum_axes = np.arange(-n_zerosum_axes, 0)
        self.assert_zerosum_axes(random_samples, n_zerosum_axes)

        new_dist = change_dist_size(base_dist, new_size=(5, 3), expand=False)
        try:
            assert new_dist.eval().shape == (5, 3, 9)
        except AssertionError:
            assert new_dist.eval().shape == (5, 3, 4, 9)
        random_samples = pm.draw(new_dist, draws=100)
        self.assert_zerosum_axes(random_samples, n_zerosum_axes)

        new_dist = change_dist_size(base_dist, new_size=(5, 3), expand=True)
        assert new_dist.eval().shape == (5, 3, 4, 9)
        random_samples = pm.draw(new_dist, draws=100)
        self.assert_zerosum_axes(random_samples, n_zerosum_axes)

    @pytest.mark.parametrize(
        "sigma, n",
        [
            (5, 3),
            (2, 6),
        ],
    )
    def test_zsn_variance(self, sigma, n):
        dist = pm.ZeroSumNormal.dist(sigma=sigma, shape=(100_000, n))
        random_samples = pm.draw(dist)

        empirical_var = random_samples.var(axis=0)
        theoretical_var = sigma**2 * (n - 1) / n

        np.testing.assert_allclose(empirical_var, theoretical_var, atol=0.4)

    @pytest.mark.parametrize(
        "sigma, shape, n_zerosum_axes, mvn_axes",
        [
            (5, 3, None, [-1]),
            (2, 6, None, [-1]),
            (5, (7, 3), None, [-1]),
            (5, (2, 7, 3), 2, [-2, -1]),
        ],
    )
    def test_zsn_logp(self, sigma, shape, n_zerosum_axes, mvn_axes):
        def logp_norm(value, sigma, axes):
            """
            Special case of the MvNormal, that's equivalent to the ZSN.
            Only to test the ZSN logp
            """
            axes = [ax if ax >= 0 else value.ndim + ax for ax in axes]
            if len(set(axes)) < len(axes):
                raise ValueError("Must specify unique zero sum axes")
            other_axes = [ax for ax in range(value.ndim) if ax not in axes]
            new_order = other_axes + axes
            reshaped_value = np.reshape(
                np.transpose(value, new_order), [value.shape[ax] for ax in other_axes] + [-1]
            )

            degrees_of_freedom = np.prod([value.shape[ax] - 1 for ax in axes])
            full_size = np.prod([value.shape[ax] for ax in axes])

            psdet = (0.5 * np.log(2 * np.pi) + np.log(sigma)) * degrees_of_freedom / full_size
            exp = 0.5 * (reshaped_value / sigma) ** 2
            inds = np.ones_like(value, dtype="bool")
            for ax in axes:
                inds = np.logical_and(inds, np.abs(np.mean(value, axis=ax, keepdims=True)) < 1e-9)
            inds = np.reshape(
                np.transpose(inds, new_order), [value.shape[ax] for ax in other_axes] + [-1]
            )[..., 0]

            return np.where(inds, np.sum(-psdet - exp, axis=-1), -np.inf)

        zsn_dist = pm.ZeroSumNormal.dist(sigma=sigma, shape=shape, n_zerosum_axes=n_zerosum_axes)
        zsn_draws = pm.draw(zsn_dist, 100)
        zsn_logp = pm.logp(zsn_dist, value=zsn_draws).eval()
        mvn_logp = logp_norm(value=zsn_draws, sigma=sigma, axes=mvn_axes)

        np.testing.assert_allclose(zsn_logp, mvn_logp)

    def test_does_not_upcast_to_float64(self):
        with pytensor.config.change_flags(floatX="float32", warn_float64="raise"):
            with pm.Model() as m:
                pm.ZeroSumNormal("b", sigma=1, shape=(2,))
            m.logp()


class TestMvStudentTCov(BaseTestDistributionRandom):
    def mvstudentt_rng_fn(self, size, nu, mu, scale, rng):
        mv_samples = rng.multivariate_normal(np.zeros_like(mu), scale, size=size)
        chi2_samples = rng.chisquare(nu, size=size)
        return (mv_samples / np.sqrt(chi2_samples[:, None] / nu)) + mu

    pymc_dist = pm.MvStudentT
    pymc_dist_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "scale": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "scale": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    sizes_to_check = [None, (1), (2, 3)]
    sizes_expected = [(2,), (1, 2), (2, 3, 2)]
    reference_dist_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "scale": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    reference_dist = lambda self: ft.partial(self.mvstudentt_rng_fn, rng=self.get_random_state())
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
        "check_errors",
        "check_mu_broadcast_helper",
    ]

    def check_errors(self):
        msg = "nu must be a scalar (ndim=0)."
        with pm.Model():
            with pytest.raises(ValueError, match=re.escape(msg)):
                mvstudentt = pm.MvStudentT(
                    "mvstudentt",
                    nu=np.array([1, 2]),
                    mu=np.ones(2),
                    scale=np.full((2, 2), np.ones(2)),
                )

    def check_mu_broadcast_helper(self):
        """Test that mu is broadcasted to the shape of cov"""
        x = pm.MvStudentT.dist(nu=4, mu=1, scale=np.eye(3))
        mu = x.owner.inputs[4]
        assert mu.eval().shape == (3,)

        x = pm.MvStudentT.dist(nu=4, mu=np.ones(1), scale=np.eye(3))
        mu = x.owner.inputs[4]
        assert mu.eval().shape == (3,)

        x = pm.MvStudentT.dist(nu=4, mu=np.ones((1, 1)), scale=np.eye(3))
        mu = x.owner.inputs[4]
        assert mu.eval().shape == (1, 3)

        x = pm.MvStudentT.dist(nu=4, mu=np.ones((10, 1)), scale=np.eye(3))
        mu = x.owner.inputs[4]
        assert mu.eval().shape == (10, 3)

        # Cov is artificially limited to being 2D
        # x = pm.MvStudentT.dist(nu=4, mu=np.ones((10, 1)), scale=np.full((2, 3, 3), np.eye(3)))
        # mu = x.owner.inputs[4]
        # assert mu.eval().shape == (10, 2, 3)


class TestMvStudentTChol(BaseTestDistributionRandom):
    pymc_dist = pm.MvStudentT
    pymc_dist_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "chol": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "scale": quaddist_matrix(chol=pymc_dist_params["chol"]).eval(),
    }
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestMvStudentTTau(BaseTestDistributionRandom):
    pymc_dist = pm.MvStudentT
    pymc_dist_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "tau": np.array([[2.0, 0.0], [0.0, 3.5]]),
    }
    expected_rv_op_params = {
        "nu": 5,
        "mu": np.array([1.0, 2.0]),
        "cov": quaddist_matrix(tau=pymc_dist_params["tau"]).eval(),
    }
    checks_to_run = ["check_pymc_params_match_rv_op"]


class TestDirichlet(BaseTestDistributionRandom):
    pymc_dist = pm.Dirichlet
    pymc_dist_params = {"a": np.array([1.0, 2.0])}
    expected_rv_op_params = {"a": np.array([1.0, 2.0])}
    sizes_to_check = [None, (1), (4,), (3, 4)]
    sizes_expected = [(2,), (1, 2), (4, 2), (3, 4, 2)]
    reference_dist_params = {"alpha": np.array([1.0, 2.0])}
    reference_dist = seeded_numpy_distribution_builder("dirichlet")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestMultinomial(BaseTestDistributionRandom):
    pymc_dist = pm.Multinomial
    pymc_dist_params = {"n": 85, "p": np.array([0.28, 0.62, 0.10])}
    expected_rv_op_params = {"n": 85, "p": np.array([0.28, 0.62, 0.10])}
    sizes_to_check = [None, (1), (4,), (3, 2)]
    sizes_expected = [(3,), (1, 3), (4, 3), (3, 2, 3)]
    reference_dist_params = {"n": 85, "pvals": np.array([0.28, 0.62, 0.10])}
    reference_dist = seeded_numpy_distribution_builder("multinomial")
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestDirichletMultinomial(BaseTestDistributionRandom):
    pymc_dist = pm.DirichletMultinomial

    pymc_dist_params = {"n": 85, "a": np.array([1.0, 2.0, 1.5, 1.5])}
    expected_rv_op_params = {"n": 85, "a": np.array([1.0, 2.0, 1.5, 1.5])}

    sizes_to_check = [None, 1, (4,), (3, 4)]
    sizes_expected = [(4,), (1, 4), (4, 4), (3, 4, 4)]

    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
        "check_random_draws",
    ]

    def check_random_draws(self):
        default_rng = pytensor.shared(np.random.default_rng(1234))
        draws = pm.DirichletMultinomial.dist(
            n=np.array([5, 100]),
            a=np.array([[0.001, 0.001, 0.001, 1000], [1000, 1000, 0.001, 0.001]]),
            size=(2, 3, 2),
            rng=default_rng,
        ).eval()
        assert np.all(draws.sum(-1) == np.array([5, 100]))
        assert np.all((draws.sum(-2)[:, :, 0] > 30) & (draws.sum(-2)[:, :, 0] <= 70))
        assert np.all((draws.sum(-2)[:, :, 1] > 30) & (draws.sum(-2)[:, :, 1] <= 70))
        assert np.all((draws.sum(-2)[:, :, 2] >= 0) & (draws.sum(-2)[:, :, 2] <= 2))
        assert np.all((draws.sum(-2)[:, :, 3] > 3) & (draws.sum(-2)[:, :, 3] <= 5))


class TestDirichletMultinomial_1D_n_2D_a(BaseTestDistributionRandom):
    pymc_dist = pm.DirichletMultinomial
    pymc_dist_params = {
        "n": np.array([23, 29]),
        "a": np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]),
    }
    sizes_to_check = [None, (1, 2), (4, 2), (3, 4, 2)]
    sizes_expected = [(2, 4), (1, 2, 4), (4, 2, 4), (3, 4, 2, 4)]
    checks_to_run = ["check_rv_size"]


class TestStickBreakingWeights(BaseTestDistributionRandom):
    pymc_dist = pm.StickBreakingWeights
    pymc_dist_params = {"alpha": 2.0, "K": 19}
    expected_rv_op_params = {"alpha": 2.0, "K": 19}
    sizes_to_check = [None, 17, (5,), (11, 5), (3, 13, 5)]
    sizes_expected = [
        (20,),
        (17, 20),
        (
            5,
            20,
        ),
        (11, 5, 20),
        (3, 13, 5, 20),
    ]
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
        "check_basic_properties",
    ]

    def check_basic_properties(self):
        default_rng = pytensor.shared(np.random.default_rng(1234))
        draws = pm.StickBreakingWeights.dist(
            alpha=3.5,
            K=19,
            size=(2, 3, 5),
            rng=default_rng,
        ).eval()

        assert np.allclose(draws.sum(-1), 1)
        assert np.all(draws >= 0)
        assert np.all(draws <= 1)


class TestStickBreakingWeights_1D_alpha(BaseTestDistributionRandom):
    pymc_dist = pm.StickBreakingWeights
    pymc_dist_params = {"alpha": [1.0, 2.0, 3.0], "K": 19}
    expected_rv_op_params = {"alpha": [1.0, 2.0, 3.0], "K": 19}
    sizes_to_check = [None, (3,), (5, 3)]
    sizes_expected = [(3, 20), (3, 20), (5, 3, 20)]
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
    ]


class TestWishart(BaseTestDistributionRandom):
    def wishart_rng_fn(self, size, nu, V, rng):
        return st.wishart.rvs(int(nu), V, size=size, random_state=rng)

    pymc_dist = pm.Wishart

    V = np.eye(3)
    pymc_dist_params = {"nu": 4, "V": V}
    reference_dist_params = {"nu": 4, "V": V}
    expected_rv_op_params = {"nu": 4, "V": V}
    sizes_to_check = [None, 1, (4, 5)]
    sizes_expected = [
        (3, 3),
        (1, 3, 3),
        (4, 5, 3, 3),
    ]
    reference_dist = lambda self: ft.partial(self.wishart_rng_fn, rng=self.get_random_state())
    checks_to_run = [
        "check_rv_size",
        "check_pymc_params_match_rv_op",
        "check_pymc_draws_match_reference",
        "check_rv_size_batched_params",
    ]

    def check_rv_size_batched_params(self):
        for size in (None, (2,), (1, 2), (4, 3, 2)):
            x = pm.Wishart.dist(nu=4, V=np.stack([np.eye(3), np.eye(3)]), size=size)

            if size is None:
                expected_shape = (2, 3, 3)
            else:
                expected_shape = size + (3, 3)

            assert tuple(x.shape.eval()) == expected_shape

            # RNG does not currently support batched parameters, whet it does this test
            # should be updated to check that draws also have the expected shape
            with pytest.raises(ValueError):
                x.eval()


class TestMatrixNormal(BaseTestDistributionRandom):
    pymc_dist = pm.MatrixNormal

    mu = np.random.random((3, 3))
    row_cov = np.eye(3)
    col_cov = np.eye(3)
    pymc_dist_params = {"mu": mu, "rowcov": row_cov, "colcov": col_cov}
    expected_rv_op_params = {"mu": mu, "rowcov": row_cov, "colcov": col_cov}

    sizes_to_check = (None, (1,), (2, 4))
    sizes_expected = [(3, 3), (1, 3, 3), (2, 4, 3, 3)]

    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
        "check_draws",
        "check_errors",
        "check_random_variable_prior",
    ]

    def check_draws(self):
        delta = 0.05  # limit for KS p-value
        n_fails = 10  # Allows the KS fails a certain number of times

        def ref_rand(mu, rowcov, colcov):
            return st.matrix_normal.rvs(mean=mu, rowcov=rowcov, colcov=colcov)

        matrixnormal = pm.MatrixNormal.dist(
            mu=np.random.random((3, 3)),
            rowcov=np.eye(3),
            colcov=np.eye(3),
        )

        p, f = delta, n_fails
        while p <= delta and f > 0:
            matrixnormal_smp = pm.draw(matrixnormal)
            ref_smp = ref_rand(mu=np.random.random((3, 3)), rowcov=np.eye(3), colcov=np.eye(3))

            p = np.min(
                [
                    st.ks_2samp(
                        matrixnormal_smp.flatten(),
                        ref_smp.flatten(),
                    )
                ]
            )
            f -= 1

        assert p > delta

    def check_errors(self):
        with pm.Model():
            matrixnormal = pm.MatrixNormal(
                "matnormal",
                mu=np.random.random((3, 3)),
                rowcov=np.eye(3),
                colcov=np.eye(3),
            )
            with pytest.raises(ValueError):
                logp(matrixnormal, pytensor.tensor.ones((3, 3, 3)))

    def check_random_variable_prior(self):
        """
        This test checks for shape correctness when using MatrixNormal distribution
        with parameters as random variables.
        Originally reported - https://github.com/pymc-devs/pymc/issues/3585
        """
        K = 3
        D = 15
        mu_0 = np.zeros((D, K))
        lambd = 1.0
        with pm.Model() as model:
            sd_dist = pm.HalfCauchy.dist(beta=2.5, size=D)
            packedL = pm.LKJCholeskyCov("packedL", eta=2, n=D, sd_dist=sd_dist, compute_corr=False)
            L = pm.expand_packed_triangular(D, packedL, lower=True)
            Sigma = pm.Deterministic("Sigma", L.dot(L.T))  # D x D covariance
            mu = pm.MatrixNormal(
                "mu", mu=mu_0, rowcov=(1 / lambd) * Sigma, colcov=np.eye(K), shape=(D, K)
            )
            prior = pm.sample_prior_predictive(2, return_inferencedata=False)

        assert prior["mu"].shape == (2, D, K)


class TestKroneckerNormal(BaseTestDistributionRandom):
    def kronecker_rng_fn(self, size, mu, covs=None, sigma=None, rng=None):
        cov = pm.math.kronecker(covs[0], covs[1]).eval()
        cov += sigma**2 * np.identity(cov.shape[0])
        return st.multivariate_normal.rvs(mean=mu, cov=cov, size=size, random_state=rng)

    pymc_dist = pm.KroneckerNormal

    n = 3
    N = n**2
    covs = [RandomPdMatrix(n), RandomPdMatrix(n)]
    mu = np.random.random(N) * 0.1
    sigma = 1

    pymc_dist_params = {"mu": mu, "covs": covs, "sigma": sigma}
    expected_rv_op_params = {"mu": mu, "covs": covs, "sigma": sigma}
    reference_dist_params = {"mu": mu, "covs": covs, "sigma": sigma}
    sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
    sizes_expected = [(N,), (N,), (1, N), (1, N), (5, N), (4, 5, N), (2, 4, 2, N)]

    reference_dist = lambda self: ft.partial(self.kronecker_rng_fn, rng=self.get_random_state())
    checks_to_run = [
        "check_pymc_draws_match_reference",
        "check_rv_size",
    ]


class TestOrderedMultinomial(BaseTestDistributionRandom):
    pymc_dist = _OrderedMultinomial
    pymc_dist_params = {"eta": 0, "cutpoints": np.array([-2, 0, 2]), "n": 1000}
    sizes_to_check = [None, (1), (4,), (3, 2)]
    sizes_expected = [(4,), (1, 4), (4, 4), (3, 2, 4)]
    expected_rv_op_params = {
        "n": 1000,
        "p": np.array([0.11920292, 0.38079708, 0.38079708, 0.11920292]),
    }
    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
    ]


class TestLKJCorr(BaseTestDistributionRandom):
    pymc_dist = pm.LKJCorr
    pymc_dist_params = {"n": 3, "eta": 1.0}
    expected_rv_op_params = {"n": 3, "eta": 1.0}

    sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
    sizes_expected = [
        (3,),
        (3,),
        (1, 3),
        (1, 3),
        (5, 3),
        (4, 5, 3),
        (2, 4, 2, 3),
    ]

    checks_to_run = [
        "check_pymc_params_match_rv_op",
        "check_rv_size",
        "check_draws_match_expected",
    ]

    def check_draws_match_expected(self):
        def ref_rand(size, n, eta):
            shape = int(n * (n - 1) // 2)
            beta = eta - 1 + n / 2
            return (st.beta.rvs(size=(size, shape), a=beta, b=beta) - 0.5) * 2

        continuous_random_tester(
            pm.LKJCorr,
            {
                "n": Domain([2, 10, 50], edges=(None, None)),
                "eta": Domain([1.0, 10.0, 100.0], edges=(None, None)),
            },
            ref_rand=ref_rand,
            size=1000,
        )


class TestLKJCholeskyCov(BaseTestDistributionRandom):
    pymc_dist = _LKJCholeskyCov
    pymc_dist_params = {"n": 3, "eta": 1.0, "sd_dist": pm.DiracDelta.dist([0.5, 1.0, 2.0])}
    expected_rv_op_params = {"n": 3, "eta": 1.0, "sd_dist": pm.DiracDelta.dist([0.5, 1.0, 2.0])}
    size = None

    sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
    sizes_expected = [
        (6,),
        (6,),
        (1, 6),
        (1, 6),
        (5, 6),
        (4, 5, 6),
        (2, 4, 2, 6),
    ]

    checks_to_run = [
        "check_rv_size",
        "check_draws_match_expected",
    ]

    def _instantiate_pymc_rv(self, dist_params=None):
        # RNG cannot be passed through the PyMC class
        params = dist_params if dist_params else self.pymc_dist_params
        self.pymc_rv = self.pymc_dist.dist(**params, size=self.size)

    def check_rv_size(self):
        for size, expected in zip(self.sizes_to_check, self.sizes_expected):
            sd_dist = pm.Exponential.dist(1, size=(*to_tuple(size), 3))
            pymc_rv = self.pymc_dist.dist(n=3, eta=1, sd_dist=sd_dist, size=size)
            expected_symbolic = tuple(pymc_rv.shape.eval())
            actual = pymc_rv.eval().shape
            assert actual == expected_symbolic == expected

    def check_draws_match_expected(self):
        # TODO: Find better comparison:
        rng = self.get_random_state(reset=True)
        x = _LKJCholeskyCov.dist(n=2, eta=10_000, sd_dist=pm.DiracDelta.dist([0.5, 2.0]))
        assert np.all(np.abs(draw(x, random_seed=rng) - np.array([0.5, 0, 2.0])) < 0.01)


class TestICAR(BaseTestDistributionRandom):
    pymc_dist = pm.ICAR
    pymc_dist_params = {"W": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), "sigma": 2}
    expected_rv_op_params = {
        "W": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]),
        "node1": np.array([1, 2, 2]),
        "node2": np.array([0, 0, 1]),
        "N": 3,
        "sigma": 2,
        "zero_sum_strength": 0.001,
    }
    checks_to_run = ["check_pymc_params_match_rv_op", "check_rv_inferred_size"]

    def check_rv_inferred_size(self):
        sizes_to_check = [None, (), 1, (1,), 5, (4, 5), (2, 4, 2)]
        sizes_expected = [(3,), (3,), (1, 3), (1, 3), (5, 3), (4, 5, 3), (2, 4, 2, 3)]
        for size, expected in zip(sizes_to_check, sizes_expected):
            pymc_rv = self.pymc_dist.dist(**self.pymc_dist_params, size=size)
            expected_symbolic = tuple(pymc_rv.shape.eval())
            assert expected_symbolic == expected

    def test_icar_logp(self):
        W = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

        with pm.Model() as m:
            RV = pm.ICAR("phi", W=W)

        assert pt.isclose(
            pm.logp(RV, np.array([0.01, -0.03, 0.02, 0.00])).eval(), np.array(4.60022238)
        ).eval(), "logp inaccuracy"

    def test_icar_rng_fn(self):
        W = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])

        RV = pm.ICAR.dist(W=W)

        with pytest.raises(NotImplementedError, match="Cannot sample from ICAR prior"):
            pm.draw(RV)

    @pytest.mark.parametrize(
        "W,msg",
        [
            (np.array([0, 1, 0, 0]), "W must be matrix with ndim=2"),
            (np.array([[0, 1, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1]]), "W must be a square matrix"),
            (
                np.array([[0, 1, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]),
                "W must be a symmetric matrix",
            ),
            (
                np.array([[0, 1, 1, 0], [1, 0, 0, 0.5], [1, 0, 0, 1], [0, 0.5, 1, 0]]),
                "W must be composed of only 1s and 0s",
            ),
        ],
    )
    def test_icar_matrix_checks(self, W, msg):
        with pytest.raises(ValueError, match=msg):
            with pm.Model():
                pm.ICAR("phi", W=W)


@pytest.mark.parametrize("sparse", [True, False])
def test_car_rng_fn(sparse):
    delta = 0.05  # limit for KS p-value
    n_fails = 20  # Allows the KS fails a certain number of times
    size = (100,)

    W = np.array(
        [[0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
    )

    tau = 2
    alpha = 0.5
    mu = np.array([1, 1, 1, 1])

    D = W.sum(axis=0)
    prec = tau * (np.diag(D) - alpha * W)
    cov = np.linalg.inv(prec)
    W = pytensor.tensor.as_tensor_variable(W)
    if sparse:
        W = pytensor.sparse.csr_from_dense(W)

    with pm.Model():
        car = pm.CAR("car", mu, W, alpha, tau, size=size)
        mn = pm.MvNormal("mn", mu, cov, size=size)
        check = pm.sample_prior_predictive(n_fails, return_inferencedata=False, random_seed=1)

    p, f = delta, n_fails
    while p <= delta and f > 0:
        car_smp, mn_smp = check["car"][f - 1, :, :], check["mn"][f - 1, :, :]
        p = min(
            st.ks_2samp(
                np.atleast_1d(car_smp[..., idx]).flatten(),
                np.atleast_1d(mn_smp[..., idx]).flatten(),
            )[1]
            for idx in range(car_smp.shape[-1])
        )
        f -= 1
    assert p > delta


@pytest.mark.parametrize(
    "matrix, result",
    [
        ([[1.0, 0], [0, 1]], True),
        ([[1.0, 2], [2, 1]], False),
        ([[1.0, 1], [1, 1]], False),
        ([[1, 0.99, 1], [0.99, 1, 0.999], [1, 0.999, 1]], False),
    ],
)
def test_posdef_symmetric(matrix, result):
    """The test returns 0 if the matrix has 0 eigenvalue.

    Is this correct?
    """
    data = np.array(matrix, dtype=pytensor.config.floatX)
    assert posdef(data) == result
