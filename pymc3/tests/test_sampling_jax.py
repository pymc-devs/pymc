import numpy as np

import pymc3 as pm

from pymc3.sampling_jax import sample_numpyro_nuts


def test_transform_samples():

    with pm.Model() as model:

        sigma = pm.HalfNormal("sigma")
        b = pm.Normal("b", sigma=sigma)
        trace = sample_numpyro_nuts(keep_untransformed=True)

    log_vals = trace.posterior["sigma_log__"].values
    trans_vals = trace.posterior["sigma"].values

    assert np.allclose(np.exp(log_vals), trans_vals)
