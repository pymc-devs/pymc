#   Copyright 2025 - present The PyMC Developers
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
import numpy as np
import pytest

from pytensor.xtensor.type import XTensorConstant, XTensorSharedVariable, XTensorType
from xarray import DataArray

import pymc as pm

from pymc import dims as pmd
from pymc import observe
from pymc.model.transform.optimization import freeze_dims_and_data

pytestmark = pytest.mark.filterwarnings("error")


def test_data():
    x_np = np.random.randn(10, 2, 3)
    coords = {"a": range(10), "b": range(2), "c": range(3)}
    with pm.Model(coords=coords) as m:
        x_data = pmd.Data("x", x_np, dims=("a", "b", "c"))
        assert isinstance(x_data, XTensorSharedVariable)
        assert isinstance(x_data.type, XTensorType)
        assert x_data.type.dims == ("a", "b", "c")

    assert m["x"] is x_data

    np.testing.assert_allclose(pm.draw(x_data), x_np)

    with m:
        pm.set_data({"x": x_np * 2})

    np.testing.assert_allclose(pm.draw(x_data), x_np * 2)


def test_simple_model():
    coords = {"a": range(3), "b": range(5)}

    with pm.Model(coords=coords) as model:
        x = pmd.Normal("x", mu=1, dims=("a", "b"))
        sigma = pmd.HalfNormal("sigma", dims=("a",))
        y = pmd.Normal("y", mu=x.T * 2, sigma=sigma, dims=("b", "a"))

    with pm.Model(coords=coords) as xmodel:
        x = pmd.Normal("x", mu=1, dims=("a", "b"))
        sigma = pmd.HalfNormal("sigma", dims=("a",))
        # Imply a transposition
        y = pmd.Normal("y", mu=x * 2, sigma=sigma, dims=("b", "a"))

        assert x.type.dims == ("a", "b")
        assert sigma.type.dims == ("a",)
        assert y.type.dims == ("b", "a")

    ip = model.initial_point()
    xip = xmodel.initial_point()
    assert ip.keys() == xip.keys()
    for value, xvalue in zip(ip.values(), xip.values()):
        np.testing.assert_allclose(value, xvalue)

    rv_shapes = model.eval_rv_shapes()
    assert rv_shapes == {
        "x": (3, 5),
        "sigma_log__": (3,),
        "sigma": (3,),
        "y": (5, 3),
    }

    logp = model.compile_logp()(ip)
    xlogp = xmodel.compile_logp()(xip)
    np.testing.assert_allclose(logp, xlogp)

    dlogp = model.compile_dlogp()(ip)
    xdlogp = xmodel.compile_dlogp()(xip)
    np.testing.assert_allclose(dlogp, xdlogp)

    draw = pm.draw(xmodel["y"], random_seed=1)
    draw_same = pm.draw(xmodel["y"], random_seed=1)
    draw_diff = pm.draw(xmodel["y"], random_seed=2)
    assert draw.shape == (5, 3)
    np.testing.assert_allclose(draw, draw_same)
    assert not np.allclose(draw, draw_diff)

    observed_values = DataArray(np.ones((3, 5)), dims=("a", "b"))
    with observe(xmodel, {"y": observed_values}):
        pm.sample_prior_predictive()
        idata = pm.sample(
            tune=200, chains=2, draws=50, compute_convergence_checks=False, progressbar=False
        )
        pm.sample_posterior_predictive(idata, progressbar=False)


def test_complex_model():
    N = 100
    rng = np.random.default_rng(4)
    x_np = np.linspace(0, 10, N)
    y_np = np.piecewise(
        x_np,
        [x_np <= 3, (x_np > 3) & (x_np <= 7), x_np > 7],
        [lambda x: 0.5 * x, lambda x: 1.5 + 0.2 * (x - 3), lambda x: 2.3 - 0.1 * (x - 7)],
    )
    y_np += rng.normal(0, 0.2, size=N)
    group_idx_np = rng.choice(3, size=N)
    N_knots = 13
    knots_np = np.linspace(0, 10, num=N_knots)

    coords = {
        "group": range(3),
        "knot": range(N_knots),
        "obs": range(N),
    }

    with pm.Model(coords=coords) as model:
        x = pm.Data("x", x_np, dims="obs")
        knots = pm.Data("knots", knots_np, dims="knot")

        sigma = pm.HalfCauchy("sigma", beta=1)
        sigma_beta0 = pm.HalfNormal("sigma_beta0", sigma=10)
        beta0 = pm.HalfNormal("beta_0", sigma=sigma_beta0, dims="group")
        z = pm.Normal("z", dims=("group", "knot"))

        delta_factors = pm.math.softmax(z, axis=-1)  # (groups, knot)
        slope_factors = 1 - delta_factors[:, :-1].cumsum(axis=-1)  # (groups, knot-1)
        spline_slopes = pm.math.concatenate(
            [beta0[:, None], beta0[:, None] * slope_factors], axis=-1
        )  # (groups, knot-1)
        beta = pm.math.concatenate(
            [beta0[:, None], pm.math.diff(spline_slopes)], axis=-1
        )  # (groups, knot)

        beta = pm.Deterministic("beta", beta, dims=("group", "knot"))

        X = pm.math.maximum(0, x[:, None] - knots[None, :])  # (n, knot)
        mu = (X * beta[group_idx_np]).sum(-1)  # ((n, knots) * (n, knots)).sum(-1) = (n,)
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_np, dims="obs")

    with pm.Model(coords=coords) as xmodel:
        x = pmd.Data("x", x_np, dims="obs")
        y = pmd.Data("y", y_np, dims="obs")
        knots = pmd.Data("knots", knots_np, dims=("knot",))
        group_idx = pmd.math.as_xtensor(group_idx_np, dims=("obs",))

        sigma = pmd.HalfCauchy("sigma", beta=1)
        sigma_beta0 = pmd.HalfNormal("sigma_beta0", sigma=10)
        beta0 = pmd.HalfNormal("beta_0", sigma=sigma_beta0, dims=("group",))
        z = pmd.Normal("z", dims=("group", "knot"))

        delta_factors = pmd.math.softmax(z, dim="knot")
        slope_factors = 1 - delta_factors.isel(knot=slice(None, -1)).cumsum("knot")
        spline_slopes = pmd.concat([beta0, beta0 * slope_factors], dim="knot")
        beta = pmd.concat([beta0, spline_slopes.diff("knot")], dim="knot")

        beta = pm.Deterministic("beta", beta)

        X = pmd.math.maximum(0, x - knots)
        mu = (X * beta.isel(group=group_idx)).sum("knot")
        y_obs = pmd.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

    # Test initial point
    model_ip = model.initial_point()
    xmodel_ip = xmodel.initial_point()
    assert model_ip.keys() == xmodel_ip.keys()
    for value, xvalue in zip(model_ip.values(), xmodel_ip.values()):
        np.testing.assert_allclose(value, xvalue)

    # Test logp
    model_logp = model.compile_logp()(model_ip)
    xmodel_logp = xmodel.compile_logp()(xmodel_ip)
    np.testing.assert_allclose(model_logp, xmodel_logp)

    # Test random draws
    model_draw = pm.draw(model["y_obs"], random_seed=1)
    xmodel_draw = pm.draw(xmodel["y_obs"], random_seed=1)
    np.testing.assert_allclose(model_draw, xmodel_draw)
    np.testing.assert_allclose(model_draw, xmodel_draw)

    with xmodel:
        pm.sample_prior_predictive()
        idata = pm.sample(
            tune=200, chains=2, draws=50, compute_convergence_checks=False, progressbar=False
        )
        pm.sample_posterior_predictive(idata, progressbar=False)


def test_zerosumnormal_model():
    coords = {"time": range(5), "item": range(3)}

    with pm.Model(coords=coords) as model:
        zsn_item = pmd.ZeroSumNormal("zsn_item", core_dims="item", dims=("time", "item"))
        zsn_time = pmd.ZeroSumNormal("zsn_time", core_dims="time", dims=("time", "item"))
        zsn_item_time = pmd.ZeroSumNormal("zsn_item_time", core_dims=("item", "time"))
    assert zsn_item.type.dims == ("time", "item")
    assert zsn_time.type.dims == ("time", "item")
    assert zsn_item_time.type.dims == ("item", "time")

    zsn_item_draw, zsn_time_draw, zsn_item_time_draw = pm.draw(
        [zsn_item, zsn_time, zsn_item_time], random_seed=1
    )
    assert zsn_item_draw.shape == (5, 3)
    np.testing.assert_allclose(zsn_item_draw.mean(-1), 0, atol=1e-13)
    assert not np.allclose(zsn_item_draw.mean(0), 0, atol=1e-13)

    assert zsn_time_draw.shape == (5, 3)
    np.testing.assert_allclose(zsn_time_draw.mean(0), 0, atol=1e-13)
    assert not np.allclose(zsn_time_draw.mean(-1), 0, atol=1e-13)

    assert zsn_item_time_draw.shape == (3, 5)
    np.testing.assert_allclose(zsn_item_time_draw.mean(), 0, atol=1e-13)

    with pm.Model(coords=coords) as ref_model:
        # Check that the ZeroSumNormal can be used in a model
        pm.ZeroSumNormal("zsn_item", dims=("time", "item"))
        pm.ZeroSumNormal("zsn_time", dims=("item", "time"))
        pm.ZeroSumNormal("zsn_item_time", n_zerosum_axes=2, dims=("item", "time"))

    # Check initial_point and logp
    ip = model.initial_point()
    ref_ip = ref_model.initial_point()
    assert ip.keys() == ref_ip.keys()
    for i, (ip_value, ref_ip_value) in enumerate(zip(ip.values(), ref_ip.values())):
        if i == 1:
            # zsn_time is actually transposed in the original model
            ip_value = ip_value.T
        np.testing.assert_allclose(ip_value, ref_ip_value)

    logp_fn = model.compile_logp()
    ref_logp_fn = ref_model.compile_logp()
    np.testing.assert_allclose(logp_fn(ip), ref_logp_fn(ref_ip))

    # Test a new point
    rng = np.random.default_rng(68)
    new_ip = ip.copy()
    for key in new_ip:
        new_ip[key] += rng.uniform(size=new_ip[key].shape)
    np.testing.assert_allclose(logp_fn(new_ip), ref_logp_fn(new_ip))


def test_freeze_dims_and_data():
    coords = {"time": range(5), "item": range(3)}
    with pm.Model(coords=coords) as m:
        x = pmd.Data("x", np.zeros((5, 3)), dims=("time", "item"))
        y = pmd.Normal("y", x)

    assert isinstance(m["x"], XTensorSharedVariable)
    assert m["x"].type.shape == (None, None)
    assert m["y"].type.shape == (None, None)

    frozen_m = freeze_dims_and_data(m)
    assert isinstance(frozen_m["x"], XTensorConstant)
    assert frozen_m["x"].type.shape == (5, 3)
    assert frozen_m["y"].type.shape == (5, 3)
