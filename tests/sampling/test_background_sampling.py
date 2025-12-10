import pymc as pm
import pytest


def test_background_sampling_happy_path():
    with pm.Model():
        pm.Normal("x", 0, 1)
        handle = pm.sample(
            draws=20,
            tune=10,
            chains=1,
            cores=1,
            background=True,
            progressbar=False,
        )
    idata = handle.result()
    assert hasattr(idata, "posterior")
    assert idata.posterior.sizes["chain"] >= 1


def test_background_sampling_raises():
    with pm.Model():
        pm.Normal("x", 0, sigma=-1)
        handle = pm.sample(
            draws=10,
            tune=5,
            chains=1,
            cores=1,
            background=True,
            progressbar=False,
        )
    with pytest.raises(Exception):
        handle.result()