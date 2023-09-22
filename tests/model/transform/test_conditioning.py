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
import arviz as az
import numpy as np
import pytest

from pytensor import config

import pymc as pm

from pymc.distributions.transforms import logodds
from pymc.model.transform.conditioning import (
    change_value_transforms,
    do,
    observe,
    remove_value_transforms,
)
from pymc.variational.minibatch_rv import create_minibatch_rv


def test_observe():
    with pm.Model() as m_old:
        x = pm.Normal("x")
        y = pm.Normal("y", x)
        z = pm.Normal("z", y)

    m_new = observe(m_old, {y: 0.5})

    assert len(m_new.free_RVs) == 2
    assert len(m_new.observed_RVs) == 1
    assert m_new["x"] in m_new.free_RVs
    assert m_new["y"] in m_new.observed_RVs
    assert m_new["z"] in m_new.free_RVs

    np.testing.assert_allclose(
        m_old.compile_logp()({"x": 0.9, "y": 0.5, "z": 1.4}),
        m_new.compile_logp()({"x": 0.9, "z": 1.4}),
    )

    # Test two substitutions
    m_new = observe(m_old, {y: 0.5, z: 1.4})

    assert len(m_new.free_RVs) == 1
    assert len(m_new.observed_RVs) == 2
    assert m_new["x"] in m_new.free_RVs
    assert m_new["y"] in m_new.observed_RVs
    assert m_new["z"] in m_new.observed_RVs

    np.testing.assert_allclose(
        m_old.compile_logp()({"x": 0.9, "y": 0.5, "z": 1.4}),
        m_new.compile_logp()({"x": 0.9}),
    )


def test_observe_minibatch():
    data = np.zeros((100,), dtype=config.floatX)
    batch_size = 10
    with pm.Model() as m_old:
        x = pm.Normal("x")
        y = pm.Normal("y", x)
        # Minibatch RVs are usually created with `total_size` kwarg
        z_raw = pm.Normal.dist(y, shape=batch_size)
        mb_z = create_minibatch_rv(z_raw, total_size=data.shape)
        m_old.register_rv(mb_z, name="mb_z")

    mb_data = pm.Minibatch(data, batch_size=batch_size)
    m_new = observe(m_old, {mb_z: mb_data})

    assert len(m_new.free_RVs) == 2
    assert len(m_new.observed_RVs) == 1
    assert m_new["x"] in m_new.free_RVs
    assert m_new["y"] in m_new.free_RVs
    assert m_new["mb_z"] in m_new.observed_RVs

    np.testing.assert_allclose(
        m_old.compile_logp()({"x": 0.9, "y": 0.5, "mb_z": np.zeros(10)}),
        m_new.compile_logp()({"x": 0.9, "y": 0.5}),
    )


def test_observe_deterministic():
    y_censored_obs = np.array([0.9, 0.5, 0.3, 1, 1], dtype=config.floatX)

    with pm.Model() as m_old:
        x = pm.Normal("x")
        y = pm.Normal.dist(x, shape=(5,))
        y_censored = pm.Deterministic("y_censored", pm.math.clip(y, -1, 1))

    m_new = observe(m_old, {y_censored: y_censored_obs})

    with pm.Model() as m_ref:
        x = pm.Normal("x")
        pm.Censored("y_censored", pm.Normal.dist(x), lower=-1, upper=1, observed=y_censored_obs)


def test_observe_dims():
    with pm.Model(coords={"test_dim": range(5)}) as m_old:
        x = pm.Normal("x", dims="test_dim")

    m_new = observe(m_old, {x: np.arange(5, dtype=config.floatX)})
    assert m_new.named_vars_to_dims["x"] == ["test_dim"]


def test_do():
    rng = np.random.default_rng(seed=435)
    with pm.Model() as m_old:
        x = pm.Normal("x", 0, 1e-3)
        y = pm.Normal("y", x, 1e-3)
        z = pm.Normal("z", y + x, 1e-3)

    assert -5 < pm.draw(z, random_seed=rng) < 5

    m_new = do(m_old, {y: x + 100})

    assert len(m_new.free_RVs) == 2
    assert m_new["x"] in m_new.free_RVs
    assert m_new["y"] in m_new.deterministics
    assert m_new["z"] in m_new.free_RVs

    assert 95 < pm.draw(m_new["z"], random_seed=rng) < 105

    # Test two substitutions
    with m_old:
        switch = pm.MutableData("switch", 1)
    m_new = do(m_old, {y: 100 * switch, x: 100 * switch})

    assert len(m_new.free_RVs) == 1
    assert m_new["y"] not in m_new.deterministics
    assert m_new["x"] not in m_new.deterministics
    assert m_new["z"] in m_new.free_RVs

    assert 195 < pm.draw(m_new["z"], random_seed=rng) < 205
    with m_new:
        pm.set_data({"switch": 0})
    assert -5 < pm.draw(m_new["z"], random_seed=rng) < 5


def test_do_posterior_predictive():
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", x, 1)
        z = pm.Normal("z", y + x, 1e-3)

    # Dummy posterior
    idata_m = az.from_dict(
        {
            "x": np.full((2, 500), 25),
            "y": np.full((2, 500), np.nan),
            "z": np.full((2, 500), np.nan),
        }
    )

    # Replace `y` by a constant `100.0`
    m_do = do(m, {y: 100.0})
    with m_do:
        idata_do = pm.sample_posterior_predictive(idata_m, var_names="z")

    assert 120 < idata_do.posterior_predictive["z"].mean() < 130


@pytest.mark.parametrize("mutable", (False, True))
def test_do_constant(mutable):
    rng = np.random.default_rng(seed=122)
    with pm.Model() as m:
        x = pm.Data("x", 0, mutable=mutable)
        y = pm.Normal("y", x, 1e-3)

    do_m = do(m, {x: 105})
    assert pm.draw(do_m["y"], random_seed=rng) > 100


def test_do_deterministic():
    rng = np.random.default_rng(seed=435)
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1e-3)
        y = pm.Deterministic("y", x + 105)
        z = pm.Normal("z", y, 1e-3)

    do_m = do(m, {"z": x - 105})
    assert pm.draw(do_m["z"], random_seed=rng) < 100


def test_do_dims():
    coords = {"test_dim": range(10)}
    with pm.Model(coords=coords) as m:
        x = pm.Normal("x", dims="test_dim")
        y = pm.Deterministic("y", x + 5, dims="test_dim")

    do_m = do(
        m,
        {"x": np.zeros(10, dtype=config.floatX)},
    )
    assert do_m.named_vars_to_dims["x"] == ["test_dim"]

    do_m = do(
        m,
        {"y": np.zeros(10, dtype=config.floatX)},
    )
    assert do_m.named_vars_to_dims["y"] == ["test_dim"]


@pytest.mark.parametrize("prune", (False, True))
def test_do_prune(prune):
    with pm.Model() as m:
        x0 = pm.ConstantData("x0", 0)
        x1 = pm.ConstantData("x1", 0)
        y = pm.Normal("y")
        y_det = pm.Deterministic("y_det", y + x0)
        z = pm.Normal("z", y_det)
        llike = pm.Normal("llike", z + x1, observed=0)

    orig_named_vars = {"x0", "x1", "y", "y_det", "z", "llike"}
    assert set(m.named_vars) == orig_named_vars

    do_m = do(m, {y_det: x0 + 5}, prune_vars=prune)
    if prune:
        assert set(do_m.named_vars) == {"x0", "x1", "y_det", "z", "llike"}
    else:
        assert set(do_m.named_vars) == orig_named_vars

    do_m = do(m, {z: 0.5}, prune_vars=prune)
    if prune:
        assert set(do_m.named_vars) == {"x1", "z", "llike"}
    else:
        assert set(do_m.named_vars) == orig_named_vars


def test_do_self_reference():
    """Check we can replace a variable by an expression that refers to the same variable."""
    with pm.Model() as m:
        x = pm.Normal("x", 0, 1)

    with pytest.warns(
        UserWarning,
        match="Intervention expression references the variable that is being intervened",
    ):
        new_m = do(m, {x: x + 100})

    x = new_m["x"]
    do_x = new_m["do_x"]
    draw_x, draw_do_x = pm.draw([x, do_x], draws=5)
    np.testing.assert_allclose(draw_x + 100, draw_do_x)


def test_change_value_transforms():
    with pm.Model() as base_m:
        p = pm.Uniform("p", 0, 1, transform=None)
        w = pm.Binomial("w", n=9, p=p, observed=6)
        assert base_m.rvs_to_transforms[p] is None
        assert base_m.rvs_to_values[p].name == "p"

    with change_value_transforms(base_m, {"p": logodds}) as transformed_p:
        new_p = transformed_p["p"]
        assert transformed_p.rvs_to_transforms[new_p] == logodds
        assert transformed_p.rvs_to_values[new_p].name == "p_logodds__"
        mean_q = pm.find_MAP(progressbar=False)

    with change_value_transforms(transformed_p, {"p": None}) as untransformed_p:
        new_p = untransformed_p["p"]
        assert untransformed_p.rvs_to_transforms[new_p] is None
        assert untransformed_p.rvs_to_values[new_p].name == "p"
        std_q = ((1 / pm.find_hessian(mean_q, vars=[new_p])) ** 0.5)[0]

    np.testing.assert_allclose(np.round(mean_q["p"], 2), 0.67)
    np.testing.assert_allclose(np.round(std_q[0], 2), 0.16)


def test_change_value_transforms_error():
    with pm.Model() as m:
        x = pm.Uniform("x", observed=5.0)

    with pytest.raises(ValueError, match="All keys must be free variables in the model"):
        change_value_transforms(m, {x: logodds})


def test_remove_value_transforms():
    with pm.Model() as base_m:
        p = pm.Uniform("p", transform=logodds)
        q = pm.Uniform("q", transform=logodds)

    new_m = remove_value_transforms(base_m)
    new_p = new_m["p"]
    new_q = new_m["q"]
    assert new_m.rvs_to_transforms == {new_p: None, new_q: None}

    new_m = remove_value_transforms(base_m, [p, q])
    new_p = new_m["p"]
    new_q = new_m["q"]
    assert new_m.rvs_to_transforms == {new_p: None, new_q: None}

    new_m = remove_value_transforms(base_m, [p])
    new_p = new_m["p"]
    new_q = new_m["q"]
    assert new_m.rvs_to_transforms == {new_p: None, new_q: logodds}

    new_m = remove_value_transforms(base_m, ["q"])
    new_p = new_m["p"]
    new_q = new_m["q"]
    assert new_m.rvs_to_transforms == {new_p: logodds, new_q: None}
