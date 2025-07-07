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

import numpy as np
import numpy.testing as npt
import pytest

import pymc as pm


@pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
def test_external_nuts_sampler(recwarn, nuts_sampler):
    if nuts_sampler != "pymc":
        pytest.importorskip(nuts_sampler)

    with pm.Model():
        x = pm.Normal("x", 100, 5)
        y = pm.Data("y", [1, 2, 3, 4])
        pm.Data("z", [100, 190, 310, 405])

        pm.Normal("L", mu=x, sigma=0.1, observed=y)

        kwargs = {
            "nuts_sampler": nuts_sampler,
            "random_seed": 123,
            "chains": 2,
            "tune": 500,
            "draws": 500,
            "progressbar": False,
            "initvals": {"x": 0.0},
        }

        idata1 = pm.sample(**kwargs)
        idata2 = pm.sample(**kwargs)

        reference_kwargs = kwargs.copy()
        reference_kwargs["nuts_sampler"] = "pymc"
        idata_reference = pm.sample(**reference_kwargs)

    warns = {
        (warn.category, warn.message.args[0])
        for warn in recwarn
        if warn.category not in (FutureWarning, DeprecationWarning, RuntimeWarning)
    }
    expected = set()
    if nuts_sampler == "nutpie":
        expected.add(
            (
                UserWarning,
                "`initvals` are currently not passed to nutpie sampler. "
                "Use `init_mean` kwarg following nutpie specification instead.",
            )
        )
    assert warns == expected
    assert "y" in idata1.constant_data
    assert "z" in idata1.constant_data
    assert "L" in idata1.observed_data
    assert idata1.posterior.chain.size == 2
    assert idata1.posterior.draw.size == 500
    assert idata1.posterior.tuning_steps == 500
    np.testing.assert_array_equal(idata1.posterior.x, idata2.posterior.x)

    assert idata_reference.posterior.attrs.keys() == idata1.posterior.attrs.keys()


def test_step_args():
    with pm.Model() as model:
        a = pm.Normal("a")
        idata = pm.sample(
            nuts_sampler="numpyro",
            target_accept=0.5,
            nuts={"max_treedepth": 10},
            random_seed=1411,
            progressbar=False,
        )

    npt.assert_almost_equal(idata.sample_stats.acceptance_rate.mean(), 0.5, decimal=1)


@pytest.mark.parametrize("nuts_sampler", ["pymc", "nutpie", "blackjax", "numpyro"])
def test_sample_var_names(nuts_sampler):
    kwargs = {
        "nuts_sampler": nuts_sampler,
        "chains": 1,
    }

    # Generate data
    seed = 1234
    rng = np.random.default_rng(seed)

    group = rng.choice(list("ABCD"), size=100)
    x = rng.normal(size=100)
    y = rng.normal(size=100)

    group_values, group_idx = np.unique(group, return_inverse=True)

    coords = {"group": group_values}

    # Create model
    with pm.Model(coords=coords) as model:
        b_group = pm.Normal("b_group", dims="group")
        b_x = pm.Normal("b_x")
        mu = pm.Deterministic("mu", b_group[group_idx] + b_x * x)
        sigma = pm.HalfNormal("sigma")
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)

    # Sample with and without var_names, but always with the same seed
    with model:
        idata_1 = pm.sample(tune=100, draws=100, random_seed=seed, **kwargs)
        idata_2 = pm.sample(
            tune=100, draws=100, var_names=["b_group", "b_x", "sigma"], random_seed=seed, **kwargs
        )

    assert "mu" in idata_1.posterior
    assert "mu" not in idata_2.posterior

    assert np.all(idata_1.posterior["b_group"] == idata_2.posterior["b_group"]).item()
    assert np.all(idata_1.posterior["b_x"] == idata_2.posterior["b_x"]).item()
    assert np.all(idata_1.posterior["sigma"] == idata_2.posterior["sigma"]).item()
