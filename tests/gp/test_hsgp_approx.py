#   Copyright 2024 - present The PyMC Developers
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

import arviz as az
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from scipy.spatial import distance

import pymc as pm

from pymc.gp.hsgp_approx import calc_eigenvalues, calc_eigenvectors

BOUNDARIES = ["dirichlet", "neumann", "dirichlet-neumann", "neumann-dirichlet"]


def build_mmd_func(sample1, sample2):
    """Build a PyTensor function that calculates the minimum mean discrepancy (MMD) statistic."""

    assert sample1.shape[1] == sample2.shape[1]

    s1 = pt.matrix(name="s1", shape=sample1.shape)
    s2 = pt.matrix(name="s2", shape=sample2.shape)

    X = np.concatenate((sample1, sample2), axis=0)
    test_ell = np.median(distance.pdist(X)) / 2

    K = pm.gp.cov.ExpQuad(sample1.shape[1], ls=test_ell)
    Kxx = K(s1)
    Kyy = K(s2)
    Kxy = K(s1, s2)

    n_x, n_y = s1.shape[0], s2.shape[0]
    mmd = (
        (pt.sum(Kxx) / (n_x * (n_x - 1)))
        + (pt.sum(Kyy) / (n_y * (n_y - 1)))
        - 2 * pt.sum(Kxy) / (n_x * n_y)
    )

    calc_mmd = pytensor.function(inputs=[s1, s2], outputs=mmd)
    return calc_mmd


def two_sample_test(sample1, sample2, n_sims=1000, alpha=0.05):
    """Calculate test whose null hypothesis is that two sets of samples were drawn from
    the same distribution.

    Largely taken from https://torchdrift.org/notebooks/note_on_mmd.html
    """
    # build function to calculate mmd
    calc_mmd = build_mmd_func(sample1, sample2)

    # simulate test statistic under null hypothesis
    X = np.concatenate((sample1, sample2), axis=0)
    half_N = int(X.shape[0] // 2)
    ix = np.arange(half_N * 2)

    h0 = []
    for i in range(n_sims):
        np.random.shuffle(ix)
        X = X[ix, :]
        h0.append(calc_mmd(X[:half_N, :], X[half_N:, :]))
    h0 = np.asarray(h0)
    critical_value = np.percentile(h0, 100 * (1 - alpha))
    mmd = calc_mmd(sample1, sample2)
    return h0, mmd, critical_value, mmd > critical_value


class _BaseFixtures:
    @pytest.fixture
    def rng(self):
        return np.random.RandomState(10)

    @pytest.fixture
    def data(self, rng):
        # 1D dataset
        X1 = np.linspace(-5, 5, 100)[:, None]

        # 3D dataset
        x1, x2, x3 = np.meshgrid(
            np.linspace(0, 10, 5), np.linspace(20, 30, 5), np.linspace(10, 20, 5)
        )
        X2 = np.vstack([x1.flatten(), x2.flatten(), x3.flatten()]).T
        return X1, X2

    @pytest.fixture
    def X1(self, data):
        return data[0]

    @pytest.fixture
    def X2(self, data):
        return data[1]

    @pytest.fixture
    def model(self):
        return pm.Model()


class TestHSGP(_BaseFixtures):
    @pytest.mark.parametrize("x_min, x_max", [(-5, 5), (-10, -1)])
    def test_set_boundaries_1d(self, x_min, x_max):
        X1 = np.linspace(x_min, x_max, 100)[:, None]
        X1s = X1 - np.mean(X1, axis=0)
        c = 2
        L = pm.gp.hsgp_approx.set_boundary(X1s, c=c)

        expected_L = np.abs(X1.max() - X1.min()) / 2 * c
        assert np.allclose(L, expected_L), f"Expected L to be close to {expected_L}, but got {L}"

    def test_set_boundaries_3d(self, X2):
        X2s = X2 - np.mean(X2, axis=0)
        L = pm.gp.hsgp_approx.set_boundary(X2s, c=2)
        assert np.all(L == 10)

    def test_mean_invariance(self):
        X = np.linspace(0, 10, 100)[:, None]
        original_center = (np.max(X, axis=0) - np.min(X, axis=0)) / 2

        with pm.Model() as model:
            _ = pm.Data("X", X)
            cov_func = pm.gp.cov.ExpQuad(1, ls=3)
            gp = pm.gp.HSGP(m=[20], L=[10], cov_func=cov_func)
            _ = gp.prior_linearized(X=X)

        x_new = np.linspace(-10, 20, 100)[:, None]
        with model:
            pm.set_data({"X": x_new})

        assert np.allclose(gp._X_center, original_center), (
            "gp._X_center should not change after updating data for out-of-sample predictions."
        )

    def test_parametrization(self):
        err_msg = (
            "`m` and `L`, if provided, must be sequences with one element per active dimension"
        )

        with pytest.raises(ValueError, match="Provide one of `c` or `L`"):
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
            pm.gp.HSGP(m=[500], c=2, L=[12], cov_func=cov_func)

        with pytest.raises(ValueError, match=err_msg):
            # m must be a list
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
            pm.gp.HSGP(m=500, c=2, cov_func=cov_func)

        with pytest.raises(ValueError, match=err_msg):
            # m must have same length as L
            cov_func = pm.gp.cov.ExpQuad(2, ls=[1, 2])
            pm.gp.HSGP(m=[500], L=[12, 12], cov_func=cov_func)

        with pytest.raises(ValueError, match=err_msg):
            # m must have same length as L, and match number of active dims of cov_func
            cov_func = pm.gp.cov.ExpQuad(1, ls=0.1)
            pm.gp.HSGP(m=[500], L=[12, 12], cov_func=cov_func)

        with pytest.raises(
            ValueError,
            match="`parametrization` must be either 'centered' or 'noncentered'.",
        ):
            cov_func = pm.gp.cov.ExpQuad(2, ls=[1, 2])
            pm.gp.HSGP(m=[50, 50], L=[12, 12], parametrization="wrong", cov_func=cov_func)

        # pass without error, cov_func has 2 active dimensions, c given as scalar
        cov_func = pm.gp.cov.ExpQuad(3, ls=[1, 2], active_dims=[0, 2])
        pm.gp.HSGP(m=[50, 50], c=2, cov_func=cov_func)

        # pass without error, all have two dimensions
        cov_func = pm.gp.cov.ExpQuad(2, ls=[1, 2])
        pm.gp.HSGP(m=[50, 50], L=[12, 12], cov_func=cov_func)

    @pytest.mark.parametrize("cov_func", [pm.gp.cov.ExpQuad(1, ls=1)])
    @pytest.mark.parametrize("drop_first", [True, False])
    def test_parametrization_drop_first(self, model, cov_func, X1, drop_first):
        n_basis = 100
        with model:
            gp = pm.gp.HSGP(m=[n_basis], c=4.0, cov_func=cov_func, drop_first=drop_first)
            gp.prior("f1", X1)

            n_coeffs = model.f1_hsgp_coeffs.type.shape[0]
            if drop_first:
                assert n_coeffs == n_basis - 1, (
                    f"one basis vector should have been dropped, {n_coeffs}"
                )
            else:
                assert n_coeffs == n_basis, "one was dropped when it shouldn't have been"

    @pytest.mark.parametrize("boundary", ["robin", ["neumann", "dirichlet"]])
    def test_boundary_validation(self, boundary):
        cov_func = pm.gp.cov.ExpQuad(1, ls=1)
        with pytest.raises(ValueError, match="`boundary` must be one of"):
            pm.gp.HSGP(m=[10], c=2.0, boundary=boundary, cov_func=cov_func)

    def test_default_boundary_is_dirichlet(self, X1):
        cov_func = pm.gp.cov.ExpQuad(1, ls=1)
        with pm.Model():
            phi_default, _ = pm.gp.HSGP(m=[10], c=2.0, cov_func=cov_func).prior_linearized(X1)
            phi_dirichlet, _ = pm.gp.HSGP(
                m=[10], c=2.0, boundary="dirichlet", cov_func=cov_func
            ).prior_linearized(X1)
        np.testing.assert_array_equal(phi_default.eval(), phi_dirichlet.eval())

    def test_boundary_with_active_dims(self):
        """The condition applies to every active dimension; at a box corner of a 2D Neumann
        basis the prior variance is 2 * 2 = 4, and inactive columns of X are ignored."""
        rng = np.random.default_rng(0)
        X = rng.uniform(-1.0, 1.0, size=(50, 3))
        X[0, 1:] = [-1.0, -1.0]  # a box corner in the two active dimensions
        X[1, 1:] = [0.0, 0.0]  # the centre
        X[2, 1:] = [1.0, 1.0]  # opposite corner, so the box is centred at exactly zero
        cov_func = pm.gp.cov.ExpQuad(3, ls=0.3, active_dims=[1, 2])
        with pm.Model():
            gp = pm.gp.HSGP(m=[60, 60], L=[1.0, 1.0], boundary="neumann", cov_func=cov_func)
            phi, sqrt_psd = gp.prior_linearized(X)
        var = ((phi**2) * sqrt_psd**2).sum(axis=1).eval()
        np.testing.assert_allclose(var[0], 4.0, atol=1e-6)
        np.testing.assert_allclose(var[1], 1.0, atol=1e-6)

    @pytest.mark.parametrize("boundary, var_at_L", [("dirichlet", 0.0), ("neumann", 2.0)])
    def test_conditional_past_training_range(self, boundary, var_at_L):
        """Predicting past the data (the forecasting use case): the conditional at the box edge
        has the prior variance dictated by the condition, and the prior is untouched."""
        X = np.linspace(0.0, 1.0, 30)[:, None]
        with pm.Model():
            gp = pm.gp.HSGP(
                m=[200], c=1.5, boundary=boundary, cov_func=pm.gp.cov.ExpQuad(1, ls=0.2)
            )
            gp.prior("f", X=X)
            fc = gp.conditional("fc", Xnew=np.array([[0.5 + 0.75], [0.5]]))  # box edge, centre
            draws = pm.draw(fc, draws=4000, random_seed=1)
        np.testing.assert_allclose(draws.var(axis=0), [var_at_L, 1.0], atol=0.15)

    @pytest.mark.parametrize(
        "boundary, var_left, var_right",
        [
            ("dirichlet", 0.0, 0.0),
            ("neumann", 2.0, 2.0),
            ("dirichlet-neumann", 0.0, 2.0),
            ("neumann-dirichlet", 2.0, 0.0),
        ],
    )
    def test_prior_variance_at_boundary(self, boundary, var_left, var_right):
        """A Dirichlet end pins f to 0; a Neumann end doubles the prior variance (even reflection)."""
        X = np.linspace(-5.0, 5.0, 11)[:, None]
        with pm.Model():
            gp = pm.gp.HSGP(
                m=[300], L=[5.0], boundary=boundary, cov_func=pm.gp.cov.ExpQuad(1, ls=1.0)
            )
            phi, sqrt_psd = gp.prior_linearized(X)
        var = ((phi**2) * sqrt_psd**2).sum(axis=1).eval()
        np.testing.assert_allclose(var[0], var_left, atol=1e-6)
        np.testing.assert_allclose(var[-1], var_right, atol=1e-6)
        np.testing.assert_allclose(var[5], 1.0, atol=1e-6)  # centre: unaffected

    def test_neumann_drop_first_removes_constant(self, X1):
        cov_func = pm.gp.cov.ExpQuad(1, ls=1)
        with pm.Model():
            phi_full, _ = pm.gp.HSGP(
                m=[10], c=2.0, boundary="neumann", cov_func=cov_func
            ).prior_linearized(X1)
            with pytest.warns(DeprecationWarning):
                gp = pm.gp.HSGP(
                    m=[10], c=2.0, boundary="neumann", drop_first=True, cov_func=cov_func
                )
            phi_drop, _ = gp.prior_linearized(X1)
        phi_full, phi_drop = phi_full.eval(), phi_drop.eval()
        assert np.ptp(phi_full[:, 0]) == 0.0  # first Neumann column is constant
        assert phi_drop.shape == (X1.shape[0], 9)
        np.testing.assert_allclose(phi_drop, phi_full[:, 1:])

    @pytest.mark.parametrize("boundary", BOUNDARIES)
    @pytest.mark.parametrize("parametrization", ["centered", "noncentered"])
    def test_conditional_equals_prior_at_training_inputs(self, X1, boundary, parametrization):
        """prior and conditional share the coefficients, so at Xnew = X they are the same function."""
        with pm.Model():
            gp = pm.gp.HSGP(
                m=[30],
                c=2.0,
                boundary=boundary,
                parametrization=parametrization,
                cov_func=pm.gp.cov.ExpQuad(1, ls=1.0),
            )
            f = gp.prior("f", X=X1)
            fc = gp.conditional("fc", Xnew=X1)
            f_draw, fc_draw = pm.draw([f, fc], random_seed=7)
        np.testing.assert_allclose(f_draw, fc_draw, rtol=1e-10)

    @pytest.mark.parametrize(
        "make_cov",
        [
            lambda ls: pm.gp.cov.ExpQuad(1, ls=ls),
            lambda ls: pm.gp.cov.Matern52(1, ls=ls),
            lambda ls: pm.gp.cov.Matern32(1, ls=ls),
            lambda ls: pm.gp.cov.RatQuad(1, alpha=2.0, ls=ls),
        ],
        ids=["ExpQuad", "Matern52", "Matern32", "RatQuad"],
    )
    def test_neumann_zero_frequency_is_differentiable(self, X1, make_cov):
        """The Neumann basis evaluates the PSD at omega=0; logp and its gradient must be finite,
        and must actually depend on the PSD (at the initial point the coefficients are zero and
        the PSD drops out, so evaluate at a perturbed point)."""
        rng = np.random.default_rng(0)
        y = rng.standard_normal(X1.shape[0])
        with pm.Model() as model:
            ls = pm.InverseGamma("ls", mu=2.0, sigma=1.0)
            gp = pm.gp.HSGP(m=[20], c=1.5, boundary="neumann", cov_func=make_cov(ls))
            f = gp.prior("f", X=X1)
            pm.Normal("y", mu=f, sigma=0.5, observed=y)
        point = model.initial_point()
        point["f_hsgp_coeffs"] = rng.standard_normal(20)
        logp_fn, dlogp_fn = model.compile_logp(), model.compile_dlogp()
        logp = logp_fn(point)
        grad = dlogp_fn(point)
        assert np.isfinite(logp)
        assert np.all(np.isfinite(grad))
        # the check is only meaningful if logp actually depends on the PSD (it does not at the
        # initial point, where all coefficients are zero): changing `ls` must change logp
        point_other_ls = dict(point, ls_log__=point["ls_log__"] + 0.5)
        assert logp_fn(point_other_ls) != logp

    @pytest.mark.parametrize("alpha, expected", [(0.5, np.isinf), (0.4, np.isnan)])
    def test_neumann_ratquad_small_alpha_documented_limitation(self, X1, alpha, expected):
        """RatQuad with alpha <= n_dims/2 has no finite spectral density at omega=0 (the kernel is
        not integrable): the formula gives inf at alpha == n_dims/2 and a negative value below,
        so the Neumann constant mode gets an inf or nan coefficient. Pins the documented
        behaviour so that a change in cov.py does not silently alter it."""
        with pm.Model():
            gp = pm.gp.HSGP(
                m=[10],
                c=1.5,
                boundary="neumann",
                cov_func=pm.gp.cov.RatQuad(1, alpha=alpha, ls=1.0),
            )
            _, sqrt_psd = gp.prior_linearized(X1)
        sqrt_psd = sqrt_psd.eval()
        assert expected(sqrt_psd[0])
        assert np.all(np.isfinite(sqrt_psd[1:]))

    @pytest.mark.parametrize(
        "cov_func,parametrization",
        [
            (pm.gp.cov.ExpQuad(1, ls=1), "centered"),
            (pm.gp.cov.ExpQuad(1, ls=1), "noncentered"),
        ],
    )
    def test_prior(self, model, cov_func, X1, parametrization, rng):
        """Compare HSGP prior to unapproximated GP prior, pm.gp.Latent.  Draw samples from the
        prior and compare them using MMD two sample test.  Tests both centered and non-centered
        parametrization.
        """
        with model:
            hsgp = pm.gp.HSGP(m=[200], c=2.0, parametrization=parametrization, cov_func=cov_func)
            f1 = hsgp.prior("f1", X=X1)

            gp = pm.gp.Latent(cov_func=cov_func)
            f2 = gp.prior("f2", X=X1)

            idata = pm.sample_prior_predictive(draws=1000, random_seed=rng)

        samples1 = az.extract(idata.prior["f1"]).values.T
        samples2 = az.extract(idata.prior["f2"]).values.T

        h0, mmd, critical_value, reject = two_sample_test(
            samples1, samples2, n_sims=500, alpha=0.01
        )
        assert not reject, "H0 was rejected, even though HSGP and GP priors should match."

    @pytest.mark.parametrize(
        "cov_func,parametrization",
        [
            (pm.gp.cov.ExpQuad(1, ls=1), "centered"),
            (pm.gp.cov.ExpQuad(1, ls=1), "noncentered"),
        ],
    )
    def test_conditional(self, model, cov_func, X1, parametrization):
        """Compare HSGP conditional to unapproximated GP prior, pm.gp.Latent.  Draw samples from the
        prior and compare them using MMD two sample test.  Tests both centered and non-centered
        parametrization.  The conditional should match the prior when no data is observed.
        """
        with model:
            hsgp = pm.gp.HSGP(m=[100], c=2.0, parametrization=parametrization, cov_func=cov_func)
            f = hsgp.prior("f", X=X1)
            fc = hsgp.conditional("fc", Xnew=X1)

            idata = pm.sample_prior_predictive(draws=1000)

        samples1 = az.extract(idata.prior["f"]).values.T
        samples2 = az.extract(idata.prior["fc"]).values.T

        h0, mmd, critical_value, reject = two_sample_test(
            samples1, samples2, n_sims=500, alpha=0.01
        )
        assert not reject, "H0 was rejected, even though HSGP prior and conditional should match."


class TestHSGPPeriodic(_BaseFixtures):
    def test_parametrization(self):
        err_msg = "`m` must be a positive integer as the `Periodic` kernel approximation is only implemented for 1-dimensional case."

        with pytest.raises(ValueError, match=err_msg):
            # `m` must be a positive integer, not a list
            cov_func = pm.gp.cov.Periodic(1, period=1, ls=0.1)
            pm.gp.HSGPPeriodic(m=[500], cov_func=cov_func)

        with pytest.raises(ValueError, match=err_msg):
            # `m`` must be a positive integer
            cov_func = pm.gp.cov.Periodic(1, period=1, ls=0.1)
            pm.gp.HSGPPeriodic(m=-1, cov_func=cov_func)

        with pytest.raises(
            ValueError,
            match="`cov_func` must be an instance of a `Periodic` kernel only. Use the `scale` parameter to control the variance.",
        ):
            # `cov_func` must be `Periodic` only
            cov_func = 5.0 * pm.gp.cov.Periodic(1, period=1, ls=0.1)
            pm.gp.HSGPPeriodic(m=500, cov_func=cov_func)

        with pytest.raises(
            ValueError,
            match="HSGP approximation for `Periodic` kernel only implemented for 1-dimensional case.",
        ):
            cov_func = pm.gp.cov.Periodic(2, period=1, ls=[1, 2])
            pm.gp.HSGPPeriodic(m=500, scale=0.5, cov_func=cov_func)

    @pytest.mark.parametrize("cov_func", [pm.gp.cov.Periodic(1, period=1, ls=1)])
    @pytest.mark.parametrize("eta", [100.0])
    @pytest.mark.xfail(
        reason="For `pm.gp.cov.Periodic`, this test does not pass.\
        The mmd is around `0.0468`.\
        The test passes more often when subtracting the mean from the mean from the samples.\
        It might be that the period is slightly off for the approximate power spectral density.\
        See https://github.com/pymc-devs/pymc/pull/6877/ for the full discussion."
    )
    def test_prior(self, model, cov_func, eta, X1, rng):
        """Compare HSGPPeriodic prior to unapproximated GP prior, pm.gp.Latent. Draw samples from the
        prior and compare them using MMD two sample test.
        """
        with model:
            hsgp = pm.gp.HSGPPeriodic(m=200, scale=eta, cov_func=cov_func)
            f1 = hsgp.prior("f1", X=X1)

            gp = pm.gp.Latent(cov_func=eta**2 * cov_func)
            f2 = gp.prior("f2", X=X1)

            idata = pm.sample_prior_predictive(draws=1000, random_seed=rng)

        samples1 = az.extract(idata.prior["f1"]).values.T
        samples2 = az.extract(idata.prior["f2"]).values.T

        h0, mmd, critical_value, reject = two_sample_test(
            samples1, samples2, n_sims=500, alpha=0.01
        )
        assert not reject, f"H0 was rejected, {mmd} even though HSGP and GP priors should match."

    @pytest.mark.parametrize("cov_func", [pm.gp.cov.Periodic(1, period=1, ls=1)])
    def test_conditional_periodic(self, model, cov_func, X1):
        """Compare HSGPPeriodic conditional to HSGPPeriodic prior. Draw samples
        from the prior and compare them using MMD two sample test. The conditional should match the
        prior when no data is observed.
        """
        with model:
            hsgp = pm.gp.HSGPPeriodic(m=100, cov_func=cov_func)
            f = hsgp.prior("f", X=X1)
            fc = hsgp.conditional("fc", Xnew=X1)

            idata = pm.sample_prior_predictive(draws=1000)

        samples1 = az.extract(idata.prior["f"]).values.T
        samples2 = az.extract(idata.prior["fc"]).values.T

        h0, mmd, critical_value, reject = two_sample_test(
            samples1, samples2, n_sims=500, alpha=0.01
        )
        assert not reject, "H0 was rejected, even though HSGP prior and conditional should match."


class TestLaplaceEigenbasis:
    @pytest.mark.parametrize(
        "boundary, j",
        [
            (None, [1, 2, 3, 4]),
            ("dirichlet", [1, 2, 3, 4]),
            ("neumann", [0, 1, 2, 3]),
            ("dirichlet-neumann", [0.5, 1.5, 2.5, 3.5]),
            ("neumann-dirichlet", [0.5, 1.5, 2.5, 3.5]),
        ],
    )
    def test_eigenvalue_index_sets(self, boundary, j):
        L = np.array([5.0])
        kwargs = {} if boundary is None else {"boundary": boundary}
        expected = ((np.pi * np.asarray(j) / (2 * L[0])) ** 2)[:, None]
        np.testing.assert_allclose(calc_eigenvalues(L, [4], **kwargs), expected)

    def test_eigenvalues_2d_shape_and_constant_mode(self):
        L = np.array([3.0, 4.0])
        m = [3, 2]
        ev = calc_eigenvalues(L, m, boundary="neumann")
        assert ev.shape == (6, 2)
        # exactly one basis vector is constant in every dimension, and it is the first one
        assert np.sum(np.all(ev == 0, axis=1)) == 1
        assert np.all(ev[0] == 0)
        for boundary in ["dirichlet", "dirichlet-neumann", "neumann-dirichlet"]:
            assert not np.any(calc_eigenvalues(L, m, boundary=boundary) == 0)

    def test_invalid_boundary(self):
        with pytest.raises(ValueError, match="`boundary` must be one of"):
            calc_eigenvalues(np.array([1.0]), [3], boundary="robin")

    @staticmethod
    def _gram(boundary, L, m, n_grid=4001):
        """Numerical L2 Gram matrix of the basis on the box [-L, L] (trapezoid rule)."""
        axes = [np.linspace(-Ld, Ld, n_grid) for Ld in L]
        grids = np.meshgrid(*axes, indexing="ij")
        Xs = np.vstack([g.ravel() for g in grids]).T
        eigvals = calc_eigenvalues(L, m, boundary=boundary)
        phi = calc_eigenvectors(Xs, L, eigvals, m, boundary=boundary).eval()
        w = np.ones(n_grid)
        w[[0, -1]] = 0.5
        weights = np.prod(
            np.meshgrid(*[w * (2 * Ld / (n_grid - 1)) for Ld in L], indexing="ij"), axis=0
        )
        return (phi * weights.ravel()[:, None]).T @ phi

    @pytest.mark.parametrize("boundary", BOUNDARIES)
    def test_orthonormal_1d(self, boundary):
        gram = self._gram(boundary, L=np.array([5.0]), m=[12])
        np.testing.assert_allclose(gram, np.eye(12), atol=1e-8)

    @pytest.mark.parametrize("boundary", BOUNDARIES)
    def test_orthonormal_2d(self, boundary):
        gram = self._gram(boundary, L=np.array([2.0, 3.0]), m=[3, 4], n_grid=201)
        np.testing.assert_allclose(gram, np.eye(12), atol=1e-8)

    @staticmethod
    def _phi_and_slope_at_ends(boundary, L=5.0, m=6, h=1e-6):
        """Basis values and central-difference slopes at x = -L and x = +L, shape (2, m) each."""
        Lv = np.array([L])
        ev = calc_eigenvalues(Lv, [m], boundary=boundary)
        X = np.array([[-L], [L], [-L + h], [-L - h], [L + h], [L - h]])
        phi = calc_eigenvectors(X, Lv, ev, [m], boundary=boundary).eval()
        slope = np.stack([(phi[2] - phi[3]) / (2 * h), (phi[4] - phi[5]) / (2 * h)])
        return phi[:2], slope

    @pytest.mark.parametrize(
        "boundary, left, right",
        [
            ("dirichlet", "value", "value"),
            ("neumann", "slope", "slope"),
            ("dirichlet-neumann", "value", "slope"),
            ("neumann-dirichlet", "slope", "value"),
        ],
    )
    def test_boundary_conditions_hold(self, boundary, left, right):
        phi, slope = self._phi_and_slope_at_ends(boundary)
        which = {"value": phi, "slope": slope}
        np.testing.assert_allclose(which[left][0], 0.0, atol=1e-12 if left == "value" else 1e-8)
        np.testing.assert_allclose(which[right][1], 0.0, atol=1e-12 if right == "value" else 1e-8)
        # the *other* quantity is not zero (otherwise the basis would be trivial)
        other = {"value": "slope", "slope": "value"}
        assert np.all(np.abs(which[other[left]][0][1:]) > 1e-3)
        assert np.all(np.abs(which[other[right]][1][1:]) > 1e-3)

    def test_neumann_constant_mode_normalisation(self):
        phi, _ = self._phi_and_slope_at_ends("neumann", L=5.0, m=6)
        j = np.arange(6)
        expected = np.stack([np.ones(6), (-1.0) ** j]) / np.sqrt(5.0)
        expected[:, 0] = 1 / np.sqrt(10.0)
        np.testing.assert_allclose(phi, expected, atol=1e-12)

    @staticmethod
    def _images_kernel_1d(x, L, ls, boundary, n_images=4):
        """m -> inf limit of the HSGP ExpQuad covariance on [-L, L] (method of images).

        Reflecting at -L and +L (odd for Dirichlet, even for Neumann) tiles the line with period
        4L; the sign alternates with the period when the two ends have different conditions.
        """

        def k(r):
            return np.exp(-0.5 * (r / ls) ** 2)

        left, _, right = boundary.partition("-")
        right = right or left
        sign_left = -1.0 if left == "dirichlet" else 1.0
        sign_right = -1.0 if right == "dirichlet" else 1.0
        K = np.zeros((len(x), len(x)))
        for n in range(-n_images, n_images + 1):
            s = (sign_left * sign_right) ** abs(n)
            K += s * k(x[:, None] - x[None, :] - 4 * n * L)
            K += s * sign_right * k(x[:, None] + x[None, :] - 2 * L - 4 * n * L)
        return K

    @pytest.mark.parametrize("boundary", BOUNDARIES)
    def test_covariance_matches_closed_form_1d(self, boundary):
        L, m, ls = np.array([5.0]), [400], 1.0
        x = np.linspace(-5.0, 5.0, 101)
        eigvals = calc_eigenvalues(L, m, boundary=boundary)
        phi = calc_eigenvectors(x[:, None], L, eigvals, m, boundary=boundary)
        psd = pm.gp.cov.ExpQuad(1, ls=ls).power_spectral_density(np.sqrt(eigvals))
        K_hsgp = ((phi * psd) @ phi.T).eval()
        np.testing.assert_allclose(K_hsgp, self._images_kernel_1d(x, L[0], ls, boundary), atol=1e-8)

    @pytest.mark.parametrize("boundary", BOUNDARIES)
    def test_covariance_matches_closed_form_2d(self, boundary):
        # ExpQuad is separable, so the 2D closed form is the Hadamard product of 1D ones.
        L, m, ls = np.array([2.0, 3.0]), [80, 80], 1.0
        x1 = np.linspace(-2.0, 2.0, 7)
        x2 = np.linspace(-3.0, 3.0, 9)
        g1, g2 = np.meshgrid(x1, x2, indexing="ij")
        Xs = np.vstack([g1.ravel(), g2.ravel()]).T
        eigvals = calc_eigenvalues(L, m, boundary=boundary)
        phi = calc_eigenvectors(Xs, L, eigvals, m, boundary=boundary)
        psd = pm.gp.cov.ExpQuad(2, ls=ls).power_spectral_density(np.sqrt(eigvals))
        K_hsgp = ((phi * psd) @ phi.T).eval()
        K1 = self._images_kernel_1d(Xs[:, 0], L[0], ls, boundary)
        K2 = self._images_kernel_1d(Xs[:, 1], L[1], ls, boundary)
        np.testing.assert_allclose(K_hsgp, K1 * K2, atol=1e-8)
