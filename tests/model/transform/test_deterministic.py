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
import pytensor
import pytest

import pymc as pm

from pymc.model.transform.deterministic import (
    extract_deterministics,
    insert_deterministics,
)
from pymc.testing import assert_equivalent_model


@pytest.mark.parametrize(
    "var_names, remaining",
    [
        (None, set()),
        ("mu", {"sigma", "mu2"}),  # mu2 depends on extracted mu and stays
        (["mu", "mu2"], {"sigma"}),
        (["sigma"], {"mu", "mu2"}),
    ],
)
def test_extract_insert_roundtrip(var_names, remaining):
    # `sigma` is independent; `mu` and `mu2` are chained (mu2 depends on mu)
    with pm.Model(coords={"obs": range(5), "feat": range(3)}) as m:
        x = pm.Data("x", np.ones((5, 3)), dims=("obs", "feat"))
        beta = pm.Normal("beta", dims="feat")
        pm.Deterministic("sigma", pm.math.exp(pm.Normal("log_sigma")))
        mu = pm.Deterministic("mu", x @ beta, dims="obs")
        pm.Deterministic("mu2", mu * 3, dims="obs")
        pm.Normal("y", m["mu2"], m["sigma"], observed=np.ones(5), dims="obs")

    m_extracted, deterministics = extract_deterministics(m, var_names)
    assert {d.name for d in m_extracted.deterministics} == remaining

    m_inserted = insert_deterministics(m_extracted, deterministics)
    assert_equivalent_model(m, m_inserted)

    # And the inserted Deterministics are wired correctly: with x=ones and beta=pi,
    # mu = 3 * pi, mu2 = 3 * mu = 9 * pi, sigma = exp(log_sigma)
    fn = pytensor.function(
        [m_inserted["beta"], m_inserted["log_sigma"]],
        [m_inserted["mu"], m_inserted["mu2"], m_inserted["sigma"]],
    )
    mu, mu2, sigma = fn(np.full(3, np.pi), np.log(2.0))
    np.testing.assert_allclose(mu, 3 * np.pi)
    np.testing.assert_allclose(mu2, 9 * np.pi)
    np.testing.assert_allclose(sigma, 2.0)


def test_no_deterministics():
    with pm.Model() as m:
        pm.Normal("a")

    m_extracted, deterministics = extract_deterministics(m)
    assert deterministics == []
    assert_equivalent_model(m, m_extracted)


def test_insert_into_recovered_latent_model():
    """Detach a deterministic before marginalizing a latent, then reinsert it onto the model
    where that latent has been recovered as its conditional (the marginalize/conditional
    workflow, with the recovered model built by hand).
    """
    # Measurement-error model: a latent log-rate (to marginalize) reported on the natural scale
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 1)
        log_rate = pm.Normal("log_rate", mu, 1)
        rate = pm.Deterministic("rate", pm.math.exp(log_rate))
        pm.Normal("measured", log_rate, 0.5, observed=0.8)

    _, deterministics = extract_deterministics(m, ["rate"])

    # The conjugate conditional `conditional` produces: `mu` at its posterior and `log_rate`
    # recovered as the posterior depending on `mu` and the measurement.
    with pm.Model() as recovered_m:
        mu = pm.Normal("mu", 0.356, 0.745)
        measured = pm.Data("measured", 0.8)
        log_rate = pm.Normal("log_rate", (mu + 4 * measured) / 5, 0.447)

    recovered_m = insert_deterministics(recovered_m, deterministics)
    assert "rate" in recovered_m.named_vars

    # `rate` is wired to the recovered `log_rate`, with the correct conditional mean
    log_rate_draws, rate_draws = pm.draw(
        [recovered_m["log_rate"], recovered_m["rate"]], draws=4000, random_seed=1
    )
    np.testing.assert_allclose(rate_draws, np.exp(log_rate_draws))
    np.testing.assert_allclose(log_rate_draws.mean(), 0.711, atol=0.05)
