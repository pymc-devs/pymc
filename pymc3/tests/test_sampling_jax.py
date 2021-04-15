import aesara
import numpy as np

import pymc3 as pm

from pymc3.sampling_jax import sample_numpyro_nuts


def test_transform_samples():
    aesara.config.on_opt_error = "raise"
    np.random.seed(13244)

    obs = np.random.normal(10, 2, size=100)
    obs_at = aesara.shared(obs, borrow=True, name="obs")
    with pm.Model() as model:
        a = pm.Uniform("a", -20, 20)
        sigma = pm.HalfNormal("sigma")
        b = pm.Normal("b", a, sigma=sigma, observed=obs_at)

        trace = sample_numpyro_nuts(chains=1, random_seed=1322, keep_untransformed=True)

    log_vals = trace.posterior["sigma_log__"].values

    trans_vals = trace.posterior["sigma"].values
    assert np.allclose(np.exp(log_vals), trans_vals)

    assert 8 < trace.posterior["a"].mean() < 11
    assert 1.5 < trace.posterior["sigma"].mean() < 2.5

    obs_at.set_value(-obs)
    with model:
        trace = sample_numpyro_nuts(chains=1, random_seed=1322, keep_untransformed=False)

    assert -11 < trace.posterior["a"].mean() < -8
    assert 1.5 < trace.posterior["sigma"].mean() < 2.5
