import numpy as np
import pymc as pm
import pytest
import pytensor


@pytest.mark.parametrize("dtype_str", ["float32", "float64"])
def test_vi_float_precision(dtype_str):
    """
    Test that Variational Inference (ADVI) works under float32 and float64 precision.
    """
    rng = np.random.default_rng(123)

    # Temporarily override floatX only within this test scope
    with pytensor.config.change_flags(floatX=dtype_str):
        data = rng.normal(loc=1.0, scale=1.0, size=50).astype(dtype_str)

        with pm.Model():
            mu = pm.Normal(
                "mu",
                mu=np.array(0.0, dtype=dtype_str),
                sigma=np.array(1.0, dtype=dtype_str),
            )
            sigma = pm.HalfNormal(
                "sigma",
                sigma=np.array(1.0, dtype=dtype_str),
            )
            pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

            approx = pm.fit(
                n=100,
                method="advi",
                progressbar=False,
                random_seed=123,
            )

        assert approx is not None

        # Request MultiTrace so dtype checks are straightforward
        trace = approx.sample(
            draws=10,
            return_inferencedata=False,
            random_seed=123,
        )

        assert trace.get_values("mu").dtype == np.dtype(dtype_str)
        assert trace.get_values("sigma").dtype == np.dtype(dtype_str)
