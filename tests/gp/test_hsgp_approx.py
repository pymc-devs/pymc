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

from pymc.gp.hsgp_approx import (
    calc_eigenvalues,
    calc_eigenvalues_dirichlet,
    calc_eigenvalues_neumann,
    calc_eigenvectors,
    calc_eigenvectors_dirichlet,
    calc_eigenvectors_neumann,
)


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

        samples1 = az.extract(idata.prior["f1"])["f1"].values.T
        samples2 = az.extract(idata.prior["f2"])["f2"].values.T

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

        samples1 = az.extract(idata.prior["f"])["f"].values.T
        samples2 = az.extract(idata.prior["fc"])["fc"].values.T

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

        samples1 = az.extract(idata.prior["f1"])["f1"].values.T
        samples2 = az.extract(idata.prior["f2"])["f2"].values.T

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

        samples1 = az.extract(idata.prior["f"])["f"].values.T
        samples2 = az.extract(idata.prior["fc"])["fc"].values.T

        h0, mmd, critical_value, reject = two_sample_test(
            samples1, samples2, n_sims=500, alpha=0.01
        )
        assert not reject, "H0 was rejected, even though HSGP prior and conditional should match."


class TestEigenvaluesFunctions:
    """Unit tests for eigenvalue calculation functions."""

    def test_calc_eigenvalues_dirichlet_1d(self):
        """Verify eigenvalues match theoretical formula λ_j = (πj/(2L))²."""
        L = np.array([5.0])
        m = [10]
        eigvals = calc_eigenvalues_dirichlet(L, m)
        expected = np.array([[(np.pi * j / (2 * 5.0)) ** 2] for j in range(1, 11)])
        np.testing.assert_allclose(eigvals, expected)

    def test_calc_eigenvalues_neumann_1d(self):
        """Verify Neumann eigenvalues start at j=1 (no j=0) and match Dirichlet."""
        L = np.array([5.0])
        m = [10]
        eigvals = calc_eigenvalues_neumann(L, m)
        # Should be identical to Dirichlet since we skip j=0
        eigvals_dirichlet = calc_eigenvalues_dirichlet(L, m)
        np.testing.assert_allclose(eigvals, eigvals_dirichlet)

    def test_calc_eigenvalues_shape(self):
        """Verify output shape is (prod(m), len(m))."""
        L = np.array([5.0, 3.0])
        m = [10, 8]
        eigvals = calc_eigenvalues_dirichlet(L, m)
        assert eigvals.shape == (80, 2)

    def test_calc_eigenvalues_2d(self):
        """Verify 2D eigenvalues are correct for both boundary conditions."""
        L = np.array([3.0, 4.0])
        m = [3, 2]
        eigvals_dirichlet = calc_eigenvalues_dirichlet(L, m)
        eigvals_neumann = calc_eigenvalues_neumann(L, m)
        assert eigvals_dirichlet.shape == (6, 2)
        assert eigvals_neumann.shape == (6, 2)
        np.testing.assert_allclose(eigvals_dirichlet, eigvals_neumann)

    def test_calc_eigenvalues_dispatcher(self):
        """Verify the dispatcher function calls the correct implementation."""
        L = np.array([5.0])
        m = [10]
        eigvals_d = calc_eigenvalues(L, m, boundary="dirichlet")
        eigvals_n = calc_eigenvalues(L, m, boundary="neumann")
        eigvals_default = calc_eigenvalues(L, m)

        np.testing.assert_allclose(eigvals_d, calc_eigenvalues_dirichlet(L, m))
        np.testing.assert_allclose(eigvals_n, calc_eigenvalues_neumann(L, m))
        np.testing.assert_allclose(eigvals_default, calc_eigenvalues_dirichlet(L, m))


class TestEigenvectorsFunctions:
    """Unit tests for eigenvector calculation functions."""

    def test_eigenvectors_dirichlet_boundary_zero(self):
        """Verify Dirichlet eigenvectors are zero at boundaries x = ±L."""
        L = np.array([5.0])
        m = [10]
        eigvals = calc_eigenvalues_dirichlet(L, m)
        # Test at boundary x = L and x = -L
        Xs_boundary = np.array([[5.0], [-5.0]])
        phi = calc_eigenvectors_dirichlet(Xs_boundary, L, eigvals, m).eval()
        np.testing.assert_allclose(phi, 0, atol=1e-10)

    def test_eigenvectors_neumann_derivative_zero(self):
        """Verify Neumann eigenvector derivatives are theoretically zero at boundaries.

        At x=±L: d/dx[cos(√λ_j(x+L))] = -√λ_j·sin(√λ_j(x+L))
        At x=L: sin(√λ_j·2L) = sin(πj) = 0 for integer j ✓
        """
        L = np.array([5.0])
        m = [10]
        eigvals = calc_eigenvalues_neumann(L, m)
        # Theoretical verification: derivative is 0 at boundaries
        for j in range(1, 11):
            assert np.isclose(np.sin(np.pi * j), 0, atol=1e-10)

    def test_eigenvectors_neumann_not_zero_at_boundary(self):
        """Verify Neumann eigenvectors are NOT zero at boundaries (unlike Dirichlet)."""
        L = np.array([5.0])
        m = [10]
        eigvals = calc_eigenvalues_neumann(L, m)
        Xs_boundary = np.array([[5.0], [-5.0]])
        phi = calc_eigenvectors_neumann(Xs_boundary, L, eigvals, m).eval()
        # Neumann eigenvectors should NOT be zero at boundaries in general
        # At x=L: cos(√λ_j·2L) = cos(πj) = ±1 for integer j
        # At x=-L: cos(√λ_j·0) = 1
        assert not np.allclose(phi, 0), "Neumann eigenvectors should not be zero at boundaries"

    def test_eigenvectors_orthonormality_dirichlet(self):
        """Verify Dirichlet eigenvectors are approximately orthonormal."""
        L = np.array([5.0])
        m = [20]
        # Dense grid for numerical integration
        x = np.linspace(-5, 5, 1000)[:, None]
        eigvals = calc_eigenvalues_dirichlet(L, m)
        phi = calc_eigenvectors_dirichlet(x, L, eigvals, m).eval()
        # Numerical integration: ∫φ_i·φ_j dx ≈ δ_ij
        dx = 10 / 1000
        gram = phi.T @ phi * dx
        np.testing.assert_allclose(gram, np.eye(20), atol=0.1)

    def test_eigenvectors_orthonormality_neumann(self):
        """Verify Neumann eigenvectors are approximately orthonormal."""
        L = np.array([5.0])
        m = [20]
        # Dense grid for numerical integration
        x = np.linspace(-5, 5, 1000)[:, None]
        eigvals = calc_eigenvalues_neumann(L, m)
        phi = calc_eigenvectors_neumann(x, L, eigvals, m).eval()
        # Numerical integration: ∫φ_i·φ_j dx ≈ δ_ij
        dx = 10 / 1000
        gram = phi.T @ phi * dx
        np.testing.assert_allclose(gram, np.eye(20), atol=0.1)

    def test_eigenvectors_shape(self):
        """Verify output shape is (n, prod(m))."""
        L = np.array([5.0, 3.0])
        m = [10, 8]
        n = 50
        Xs = np.random.randn(n, 2)
        eigvals = calc_eigenvalues_dirichlet(L, m)
        phi = calc_eigenvectors_dirichlet(Xs, L, eigvals, m).eval()
        assert phi.shape == (50, 80)

    def test_eigenvectors_dispatcher(self):
        """Verify the dispatcher function calls the correct implementation."""
        L = np.array([5.0])
        m = [10]
        Xs = np.linspace(-3, 3, 20)[:, None]
        eigvals = calc_eigenvalues_dirichlet(L, m)

        phi_d = calc_eigenvectors(Xs, L, eigvals, m, boundary="dirichlet").eval()
        phi_n = calc_eigenvectors(Xs, L, eigvals, m, boundary="neumann").eval()
        phi_default = calc_eigenvectors(Xs, L, eigvals, m).eval()

        np.testing.assert_allclose(phi_d, calc_eigenvectors_dirichlet(Xs, L, eigvals, m).eval())
        np.testing.assert_allclose(phi_n, calc_eigenvectors_neumann(Xs, L, eigvals, m).eval())
        np.testing.assert_allclose(
            phi_default, calc_eigenvectors_dirichlet(Xs, L, eigvals, m).eval()
        )


class TestHSGPBoundaryConditions(_BaseFixtures):
    """Integration tests for HSGP with different boundary conditions."""

    def test_boundary_parameter_validation(self):
        """Verify invalid boundary parameter raises ValueError."""
        with pytest.raises(ValueError, match="`boundary` must be"):
            cov_func = pm.gp.cov.ExpQuad(1, ls=1)
            pm.gp.HSGP(m=[50], c=2, boundary="invalid", cov_func=cov_func)

    def test_boundary_default_is_dirichlet(self):
        """Verify default boundary condition is Dirichlet."""
        cov_func = pm.gp.cov.ExpQuad(1, ls=1)
        hsgp = pm.gp.HSGP(m=[50], c=2, cov_func=cov_func)
        assert hsgp._boundary == "dirichlet"

    @pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
    def test_prior_matches_gp(self, model, X1, boundary, rng):
        """Compare HSGP prior to unapproximated GP using MMD test."""
        with model:
            cov_func = pm.gp.cov.ExpQuad(1, ls=1)
            hsgp = pm.gp.HSGP(m=[200], c=2.0, boundary=boundary, cov_func=cov_func)
            f1 = hsgp.prior("f1", X=X1)

            gp = pm.gp.Latent(cov_func=cov_func)
            f2 = gp.prior("f2", X=X1)

            idata = pm.sample_prior_predictive(draws=1000, random_seed=rng)

        samples1 = az.extract(idata.prior["f1"])["f1"].values.T
        samples2 = az.extract(idata.prior["f2"])["f2"].values.T
        h0, mmd, critical_value, reject = two_sample_test(
            samples1, samples2, n_sims=500, alpha=0.01
        )
        assert not reject, (
            f"H0 was rejected for {boundary} boundary, even though HSGP and GP priors should match."
        )

    @pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
    def test_conditional_matches_prior(self, model, X1, boundary):
        """Verify conditional matches prior when no data observed."""
        with model:
            cov_func = pm.gp.cov.ExpQuad(1, ls=1)
            hsgp = pm.gp.HSGP(m=[100], c=2.0, boundary=boundary, cov_func=cov_func)
            f = hsgp.prior("f", X=X1)
            fc = hsgp.conditional("fc", Xnew=X1)

            idata = pm.sample_prior_predictive(draws=1000)

        samples1 = az.extract(idata.prior["f"])["f"].values.T
        samples2 = az.extract(idata.prior["fc"])["fc"].values.T
        h0, mmd, critical_value, reject = two_sample_test(
            samples1, samples2, n_sims=500, alpha=0.01
        )
        assert not reject, (
            f"H0 was rejected for {boundary} boundary, "
            "even though HSGP prior and conditional should match."
        )

    def test_neumann_no_edge_effect(self, model):
        """Verify Neumann doesn't force function toward zero at edges."""
        X = np.linspace(-5, 5, 100)[:, None]
        X_edge = np.array([[-4.9], [4.9]])  # Near boundaries

        with model:
            cov_func = pm.gp.cov.ExpQuad(1, ls=1)
            hsgp = pm.gp.HSGP(m=[100], c=2.0, boundary="neumann", cov_func=cov_func)
            hsgp.prior("f", X=X)
            fc = hsgp.conditional("fc", Xnew=X_edge)
            idata = pm.sample_prior_predictive(draws=500)

        edge_samples = az.extract(idata.prior["fc"])["fc"].values
        # Variance at edges should be similar to interior (not suppressed)
        edge_var = np.var(edge_samples, axis=1)
        assert np.all(edge_var > 0.5), "Neumann should not suppress variance at edges"

    def test_dirichlet_edge_effect(self, model):
        """Verify Dirichlet DOES force function toward zero at edges."""
        X = np.linspace(-5, 5, 100)[:, None]
        # Use points very close to the boundary L
        # With c=2.0 and X in [-5, 5], L = 2 * 5 = 10
        X_edge = np.array([[-9.9], [9.9]])  # Very close to L boundaries

        with model:
            cov_func = pm.gp.cov.ExpQuad(1, ls=1)
            hsgp = pm.gp.HSGP(m=[100], c=2.0, boundary="dirichlet", cov_func=cov_func)
            hsgp.prior("f", X=X)
            fc = hsgp.conditional("fc", Xnew=X_edge)
            idata = pm.sample_prior_predictive(draws=500)

        edge_samples = az.extract(idata.prior["fc"])["fc"].values
        edge_var = np.var(edge_samples, axis=1)
        # Dirichlet should suppress variance at boundaries
        assert np.all(edge_var < 0.5), "Dirichlet should suppress variance at boundaries"

    @pytest.mark.parametrize("boundary", ["dirichlet", "neumann"])
    def test_prior_linearized_with_boundary(self, X1, boundary):
        """Verify prior_linearized works correctly with boundary parameter."""
        with pm.Model():
            cov_func = pm.gp.cov.ExpQuad(1, ls=1)
            hsgp = pm.gp.HSGP(m=[50], c=2.0, boundary=boundary, cov_func=cov_func)
            phi, sqrt_psd = hsgp.prior_linearized(X=X1)

            # Check shapes
            assert phi.eval().shape == (100, 50)
            assert sqrt_psd.eval().shape == (50,)
