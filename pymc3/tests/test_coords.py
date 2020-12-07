import numpy as np

import pymc3 as pm


def test_coords():
    chains = 2
    n_features = 3
    n_samples = 10

    coords = {"features": np.arange(n_features)}

    with pm.Model(coords=coords):
        a = pm.Uniform("a", -100, 100, dims="features")
        b = pm.Uniform("b", -100, 100, dims="features")
        tr = pm.sample(n_samples, chains=chains, return_inferencedata=True)

    assert "features" in tr.posterior.a.coords.dims
    assert "features" in tr.posterior.b.coords.dims
