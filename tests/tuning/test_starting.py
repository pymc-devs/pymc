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
import re

import numpy as np
import pytest
import xarray as xr

from numpy.testing import assert_allclose
from scipy.optimize import LbfgsInvHessProduct, OptimizeResult
from scipy.sparse.linalg import LinearOperator

import pymc as pm

from pymc.exceptions import ImputationWarning, SamplingError
from pymc.step_methods.metropolis import tune
from pymc.testing import select_by_precision
from pymc.tuning import find_MAP
from pymc.tuning.starting import _optimizer_result_to_dataset
from tests import models
from tests.models import non_normal, simple_arbitrary_det, simple_model


@pytest.fixture
def normal_model():
    rng = np.random.default_rng(sum(map(ord, "find_MAP")))
    with pm.Model() as m:
        mu = pm.Normal("mu")
        sigma = pm.Exponential("sigma", 1)
        pm.Normal("y_hat", mu=mu, sigma=sigma, observed=rng.normal(loc=3, scale=1.5, size=10))
    return m


@pytest.mark.parametrize("bounded", [False, True])
def test_mle_jacobian(bounded):
    """Test MAP / MLE estimation for distributions with flat priors."""
    truth = 10.0  # Simple normal model should give mu=10.0
    rtol = 1e-4  # this rtol should work on both floatX precisions

    _, model, _ = models.simple_normal(bounded_prior=bounded)
    map_estimate = find_MAP(
        method="BFGS", model=model, return_inferencedata=False, progressbar=False
    )
    assert_allclose(map_estimate["mu_i"], truth, rtol=rtol)


def test_tune_not_inplace():
    orig_scaling = np.array([0.001, 0.1])
    returned_scaling = tune(orig_scaling, acc_rate=0.6)
    assert returned_scaling is not orig_scaling
    assert np.all(orig_scaling == np.array([0.001, 0.1]))


def test_accuracy_normal():
    _, model, (mu, _) = simple_model()
    with model:
        newstart = find_MAP(
            initvals=pm.Point(x=[-10.5, 100.5]), return_inferencedata=False, progressbar=False
        )
        assert_allclose(
            newstart["x"], [mu, mu], atol=select_by_precision(float64=1e-5, float32=1e-4)
        )


def test_accuracy_non_normal():
    _, model, (mu, _) = non_normal(4)
    with model:
        newstart = find_MAP(
            initvals=pm.Point(x=[0.5, 0.01, 0.95, 0.99]),
            jitter=False,
            return_inferencedata=False,
            progressbar=False,
        )
        assert_allclose(newstart["x"], mu, atol=select_by_precision(float64=1e-5, float32=1e-4))


def test_find_MAP_discrete():
    tol1 = 2.0**-11
    tol2 = 2.0**-6
    alpha = 4
    beta = 4
    n = 20
    yes = 15

    with pm.Model() as model:
        p = pm.Beta("p", alpha, beta)
        pm.Binomial("ss", n=n, p=p)
        pm.Binomial("s", n=n, p=p, observed=yes)

        map_est1 = find_MAP(return_inferencedata=False, progressbar=False)
        with pytest.warns(UserWarning, match="Discrete variables are being optimized"):
            map_est2 = find_MAP(
                vars=model.value_vars, return_inferencedata=False, progressbar=False
            )

    # ss is held fixed at its (jittered-p dependent) initial value; conjugate MAP given ss
    ss0 = map_est1["ss"]
    assert_allclose(
        map_est1["p"], (alpha + yes + ss0 - 1) / (alpha + beta + 2 * n - 2), atol=tol1, rtol=0
    )

    assert_allclose(map_est2["p"], 0.695642178810167, atol=tol2, rtol=0)
    assert map_est2["ss"] == 14


def test_find_MAP_no_gradient():
    _, model = simple_arbitrary_det()
    with pytest.warns(UserWarning, match="Gradient not available"):
        find_MAP(model=model, progressbar=False)
    with pytest.raises(Exception):
        find_MAP(model=model, use_grad=True, progressbar=False)


def test_find_MAP():
    tol = 2.0**-11  # 16 bit machine epsilon, a low bar
    data = np.random.randn(100)
    # data should be roughly mean 0, std 1, but let's
    # normalize anyway to get it really close
    data = (data - np.mean(data)) / np.std(data)

    with pm.Model():
        mu = pm.Uniform("mu", -1, 1)
        sigma = pm.Uniform("sigma", 0.5, 1.5)
        pm.Normal("y", mu=mu, tau=sigma**-2, observed=data)

        # Test gradient minimization
        map_est1 = find_MAP(progressbar=False, return_inferencedata=False)
        # Test non-gradient minimization, with case-insensitive method name
        map_est2 = find_MAP(progressbar=False, method="Powell", return_inferencedata=False)

    assert_allclose(map_est1["mu"], 0, atol=tol)
    assert_allclose(map_est1["sigma"], 1, atol=tol)

    assert_allclose(map_est2["mu"], 0, atol=tol)
    assert_allclose(map_est2["sigma"], 1, atol=tol)


def test_find_MAP_issue_5923():
    # Test that gradient-based minimization works well regardless of the order
    # of variables in `vars`, and even when starting a reasonable distance from
    # the MAP.
    tol = 2.0**-11  # 16 bit machine epsilon, a low bar
    data = np.random.randn(100)
    # data should be roughly mean 0, std 1, but let's
    # normalize anyway to get it really close
    data = (data - np.mean(data)) / np.std(data)

    with pm.Model():
        mu = pm.Uniform("mu", -1, 1)
        sigma = pm.Uniform("sigma", 0.5, 1.5)
        pm.Normal("y", mu=mu, tau=sigma**-2, observed=data)

        start = {"mu": -0.5, "sigma": 1.25}
        kwargs = {"progressbar": False, "initvals": start, "return_inferencedata": False}
        map_est1 = find_MAP(vars=[mu, sigma], **kwargs)
        map_est2 = find_MAP(vars=[sigma, mu], **kwargs)

    assert_allclose(map_est1["mu"], 0, atol=tol)
    assert_allclose(map_est1["sigma"], 1, atol=tol)

    assert_allclose(map_est2["mu"], 0, atol=tol)
    assert_allclose(map_est2["sigma"], 1, atol=tol)


def test_find_MAP_issue_4488():
    # Test for https://github.com/pymc-devs/pymc/issues/4488
    with pm.Model() as m:
        with pytest.warns(ImputationWarning):
            x = pm.Gamma("x", alpha=3, beta=10, observed=np.array([1, np.nan]))
        y = pm.Deterministic("y", x + 1)
        map_estimate = find_MAP(
            include_transformed=True, return_inferencedata=False, progressbar=False
        )

    assert not set.difference({"x_unobserved", "x_unobserved_log__", "y"}, set(map_estimate.keys()))
    assert_allclose(map_estimate["x_unobserved"], 0.2, rtol=1e-4, atol=1e-4)
    assert_allclose(map_estimate["y"], [2.0, map_estimate["x_unobserved"][0] + 1])


def test_find_MAP_warning_non_free_RVs():
    with pm.Model() as m:
        x = pm.Normal("x")
        y = pm.Normal("y")
        det = pm.Deterministic("det", x + y)
        pm.Normal("z", det, 1e-5, observed=100)

        msg = "Intermediate variables (such as Deterministic or Potential) were passed"
        with pytest.warns(UserWarning, match=re.escape(msg)):
            r = pm.find_MAP(vars=[det], jitter=False, return_inferencedata=False, progressbar=False)
        assert_allclose([r["x"], r["y"], r["det"]], [50, 50, 100])


def test_find_MAP_vars_subset_holds_others_fixed(normal_model):
    with normal_model:
        r = find_MAP(
            vars=[normal_model["mu"]],
            initvals={"sigma": 2.0},
            return_inferencedata=False,
            progressbar=False,
        )
    assert_allclose(r["sigma"], 2.0)


@pytest.mark.parametrize(
    "method, use_grad, use_hess, use_hessp",
    [
        ("Newton-CG", True, True, False),
        ("Newton-CG", True, False, True),
        ("BFGS", True, False, False),
        ("L-BFGS-B", True, False, False),
        ("trust-exact", True, True, False),
        ("powell", False, False, False),
    ],
)
@pytest.mark.parametrize("include_transformed, compute_hessian", [(True, True), (False, False)])
def test_find_MAP_inferencedata(
    normal_model, method, use_grad, use_hess, use_hessp, include_transformed, compute_hessian
):
    idata = find_MAP(
        method=method,
        model=normal_model,
        use_grad=use_grad,
        use_hess=use_hess,
        use_hessp=use_hessp,
        progressbar=False,
        include_transformed=include_transformed,
        compute_hessian=compute_hessian,
    )
    for group in ["posterior", "fit", "optimizer_result", "observed_data"]:
        assert group in idata.children

    posterior = idata.posterior.dataset.squeeze(["chain", "draw"])
    assert posterior["mu"].shape == () and posterior["sigma"].shape == ()
    assert ("sigma_log__" in posterior) == include_transformed
    assert ("covariance_matrix" in idata.fit) == compute_hessian
    assert idata.fit.rows.values.tolist() == ["mu", "sigma_log__"]
    assert idata.optimizer_result["method"].item() == method
    assert ("hess_inv" in idata.optimizer_result) == (method == "BFGS")
    assert ("hess_inv_sk" in idata.optimizer_result) == (method == "L-BFGS-B")
    for key in ("hess", "hess_inv"):
        if key in idata.optimizer_result:
            assert idata.optimizer_result[key].dims == ("variables", "variables_aux")


@pytest.mark.parametrize("gradient_backend", ["jax", "pytensor"])
def test_find_MAP_jax_backend(normal_model, gradient_backend):
    pytest.importorskip("jax")
    idata = find_MAP(
        model=normal_model,
        backend="jax",
        compile_kwargs={"gradient_backend": gradient_backend},
        compute_hessian=True,
        progressbar=False,
    )
    assert idata.fit.covariance_matrix.shape == (2, 2)
    assert_allclose(idata.posterior["mu"].item(), 3.0, atol=1.0)


def test_find_MAP_return_inferencedata_consistent(normal_model):
    kwargs = {
        "model": normal_model,
        "include_transformed": True,
        "progressbar": False,
        "random_seed": 1,
    }
    idata = find_MAP(**kwargs)
    point = find_MAP(return_inferencedata=False, **kwargs)
    assert set(point) == {"mu", "sigma", "sigma_log__"}
    for name, value in point.items():
        assert_allclose(idata.posterior[name].values.squeeze(), value)


def test_find_MAP_idata_kwargs(normal_model):
    idata = find_MAP(model=normal_model, idata_kwargs={"log_likelihood": True}, progressbar=False)
    assert "log_likelihood" in idata.children
    assert idata.log_likelihood["y_hat"].shape == (1, 1, 10)


def test_find_MAP_shared_variables():
    x_val = np.linspace(-1, 1, 20)
    with pm.Model() as m:
        x = pm.Data("x", x_val)
        beta = pm.Normal("beta")
        sigma = pm.Exponential("sigma", 1)
        pm.Normal(
            "y", beta * x, sigma, observed=2 * x_val + np.random.default_rng(0).normal(0, 0.1, 20)
        )

    idata = find_MAP(model=m, progressbar=False)
    assert "x" in idata.constant_data
    assert "y" in idata.observed_data
    assert_allclose(idata.posterior["beta"].item(), 2.0, atol=0.1)


@pytest.mark.parametrize("use_hess, use_hessp", [(True, False), (False, True)])
def test_find_MAP_basinhopping(normal_model, use_hess, use_hessp):
    idata = find_MAP(
        method="basinhopping",
        model=normal_model,
        use_hess=use_hess,
        use_hessp=use_hessp,
        progressbar=False,
        random_seed=1,
        minimizer_kwargs={"method": "Newton-CG"},
        niter=1,
    )
    assert idata.posterior["mu"].shape == (1, 1)
    assert idata.optimizer_result["method"].item() == "basinhopping"


def test_find_MAP_with_coords():
    with pm.Model(coords={"group": [1, 2, 3, 4, 5]}) as m:
        mu_loc = pm.Normal("mu_loc", 0, 1)
        mu_scale = pm.HalfNormal("mu_scale", 1)
        mu = pm.Normal("mu", mu_loc, mu_scale, dims=["group"])
        sigma = pm.HalfNormal("sigma", 1, dims=["group"])
        pm.Normal("obs", mu=mu, sigma=sigma, observed=np.random.normal(size=(10, 5)))

    idata = find_MAP(model=m, progressbar=False, include_transformed=True)
    posterior = idata.posterior.dataset.squeeze(["chain", "draw"])
    assert posterior["mu"].dims == ("group",)
    assert posterior["sigma"].shape == (5,)
    assert posterior["sigma_log__"].dims == ("group",)
    assert idata.fit.rows.values.tolist() == [
        "mu_loc",
        "mu_scale_log__",
        *[f"mu[{i}]" for i in range(1, 6)],
        *[f"sigma_log__[{i}]" for i in range(1, 6)],
    ]


def test_find_MAP_nonscalar_rv_without_dims():
    with pm.Model(coords={"test": ["A", "B", "C"]}) as model:
        x_loc = pm.Normal("x_loc", mu=0, sigma=1, dims=["test"])
        x = pm.Normal("x", mu=x_loc, sigma=1, shape=(2, 3))
        pm.Normal("y", mu=x, sigma=1, observed=np.random.randn(10, 2, 3))

    idata = find_MAP(model=model, progressbar=False)
    assert idata.posterior["x"].shape == (1, 1, 2, 3)
    assert idata.fit.rows.values.tolist() == [
        "x_loc[A]",
        "x_loc[B]",
        "x_loc[C]",
        *[f"x[{i},{j}]" for i in range(2) for j in range(3)],
    ]


def test_find_MAP_jitter_escapes_saddle():
    # https://github.com/pymc-devs/pymc-extras/issues/687
    with pm.Model() as m:
        w = pm.Normal("w")
        z = pm.Normal("z")
        pm.Normal("y", mu=w * z, sigma=0.1, observed=1.0)

    kwargs = {"model": m, "progressbar": False, "return_inferencedata": False}
    stuck = find_MAP(jitter=False, **kwargs)
    assert_allclose([stuck["w"], stuck["z"]], 0.0)
    r1 = find_MAP(random_seed=11, **kwargs)
    r2 = find_MAP(random_seed=11, **kwargs)
    assert_allclose(r1["w"] * r1["z"], 1.0, atol=0.05)
    assert_allclose(r1["w"], r2["w"])


def test_find_MAP_invalid_start_raises():
    with pm.Model() as m:
        pm.Uniform("x", 0, 1, default_transform=None)
    with pytest.raises(SamplingError, match="Initial evaluation of model at starting point failed"):
        find_MAP(model=m, initvals={"x": 2.0}, progressbar=False)


def test_find_MAP_unknown_method(normal_model):
    with pytest.raises(ValueError, match="Unknown method"):
        find_MAP(method="gradient-descent", model=normal_model, progressbar=False)


def test_find_MAP_legacy_kwargs(normal_model):
    kwargs = {
        "model": normal_model,
        "progressbar": False,
        "return_inferencedata": False,
        "random_seed": 1,
    }
    with pytest.warns(FutureWarning, match="`start` is deprecated"):
        r1 = find_MAP({"mu": 1.0}, **kwargs)
    with pytest.warns(FutureWarning, match="`start` is deprecated"):
        r2 = find_MAP(start={"mu": 1.0}, **kwargs)
    assert_allclose(r1["mu"], r2["mu"])
    with pytest.warns(FutureWarning, match="`seed` is deprecated"):
        find_MAP(seed=1, **kwargs)
    with pytest.warns(FutureWarning, match="`maxeval` is deprecated"):
        find_MAP(maxeval=10, **kwargs)
    with pytest.warns(FutureWarning, match="`return_raw` is deprecated"):
        point, res = find_MAP(return_raw=True, **kwargs)
    assert isinstance(res, OptimizeResult)
    assert set(point) == {"mu", "sigma"}


class TestOptimizerResultToDataset:
    names = ["mu", "sigma_log__"]

    def test_basic(self):
        result = OptimizeResult(
            x=np.array([1.0, 2.0]),
            fun=0.5,
            success=True,
            message="done",
            jac=np.array([0.1, 0.2]),
            nit=5,
            custom_stat=np.array([42, 43]),
        )
        ds = _optimizer_result_to_dataset(result, "BFGS", self.names)
        assert isinstance(ds, xr.Dataset)
        assert ds["x"].coords["variables"].values.tolist() == self.names
        assert ds["jac"].dims == ("variables",)
        assert ds["message"].item() == "done" and ds["method"].item() == "BFGS"
        assert ds["custom_stat"].dims == ("custom_stat_dim_0",)
        assert "variables_aux" not in ds.coords

    def test_hess_inv_linear_operator(self):
        result = OptimizeResult(
            x=np.ones(2), hess_inv=LinearOperator((2, 2), matvec=lambda x: 2 * x)
        )
        ds = _optimizer_result_to_dataset(result, "L-BFGS-B", self.names)
        assert ds["hess_inv"].dims == ("variables", "variables_aux")
        assert ds["hess_inv"].coords["variables_aux"].values.tolist() == self.names
        assert_allclose(ds["hess_inv"].values, 2 * np.eye(2))

    def test_lbfgs_hess_inv_kept_low_rank(self):
        rng = np.random.default_rng(0)
        n, m = 2, 3
        sk, yk = rng.normal(size=(m, n)), rng.normal(size=(m, n))
        yk = yk + np.sign(np.sum(sk * yk, axis=1))[:, None] * sk  # keep s.y > 0
        result = OptimizeResult(x=np.ones(n), hess_inv=LbfgsInvHessProduct(sk, yk))
        ds = _optimizer_result_to_dataset(result, "L-BFGS-B", self.names)
        assert "hess_inv" not in ds
        for key, pairs in (("hess_inv_sk", sk), ("hess_inv_yk", yk)):
            assert ds[key].dims == ("lbfgs_corrections", "variables")
            assert ds[key].coords["variables"].values.tolist() == self.names
            assert_allclose(ds[key].values, pairs)

    def test_basinhopping_nested_result(self):
        result = OptimizeResult(
            x=np.ones(2),
            lowest_optimization_result=OptimizeResult(x=np.zeros(2), hess_inv=3 * np.eye(2)),
        )
        ds = _optimizer_result_to_dataset(result, "basinhopping", self.names)
        assert_allclose(ds["hess_inv"].values, 3 * np.eye(2))
        assert_allclose(ds["x"].values, 0.0)
        assert "lowest_optimization_result" not in ds
