import aesara
import aesara.tensor as at
import numpy as np
import pytest

from aesara.compile import SharedVariable
from aesara.graph import graph_inputs

import pymc as pm

from pymc.sampling_jax import (
    _get_batched_jittered_initial_points,
    _get_log_likelihood,
    _replace_shared_variables,
    get_jaxified_graph,
    get_jaxified_logp,
    sample_blackjax_nuts,
    sample_numpyro_nuts,
)


@pytest.mark.parametrize(
    "sampler",
    [
        sample_blackjax_nuts,
        sample_numpyro_nuts,
    ],
)
@pytest.mark.parametrize("postprocessing_backend", [None, "cpu"])
def test_transform_samples(sampler, postprocessing_backend):
    aesara.config.on_opt_error = "raise"
    np.random.seed(13244)

    obs = np.random.normal(10, 2, size=100)
    obs_at = aesara.shared(obs, borrow=True, name="obs")
    with pm.Model() as model:
        a = pm.Uniform("a", -20, 20)
        sigma = pm.HalfNormal("sigma")
        b = pm.Normal("b", a, sigma=sigma, observed=obs_at)

        trace = sampler(
            chains=1,
            random_seed=1322,
            keep_untransformed=True,
            postprocessing_backend=postprocessing_backend,
        )

    log_vals = trace.posterior["sigma_log__"].values

    trans_vals = trace.posterior["sigma"].values
    assert np.allclose(np.exp(log_vals), trans_vals)

    assert 8 < trace.posterior["a"].mean() < 11
    assert 1.5 < trace.posterior["sigma"].mean() < 2.5

    obs_at.set_value(-obs)
    with model:
        trace = sampler(
            chains=2,
            random_seed=1322,
            keep_untransformed=False,
            postprocessing_backend=postprocessing_backend,
        )

    assert -11 < trace.posterior["a"].mean() < -8
    assert 1.5 < trace.posterior["sigma"].mean() < 2.5


@pytest.mark.parametrize(
    "sampler",
    [
        sample_blackjax_nuts,
        sample_numpyro_nuts,
    ],
)
def test_deterministic_samples(sampler):
    aesara.config.on_opt_error = "raise"
    np.random.seed(13244)

    obs = np.random.normal(10, 2, size=100)
    obs_at = aesara.shared(obs, borrow=True, name="obs")
    with pm.Model() as model:
        a = pm.Uniform("a", -20, 20)
        b = pm.Deterministic("b", a / 2.0)
        c = pm.Normal("c", a, sigma=1.0, observed=obs_at)

        trace = sampler(chains=2, random_seed=1322, keep_untransformed=True)

    assert 8 < trace.posterior["a"].mean() < 11
    assert np.allclose(trace.posterior["b"].values, trace.posterior["a"].values / 2)


def test_get_jaxified_graph():
    # Check that jaxifying a graph does not emmit the Supervisor Warning. This test can
    # be removed once https://github.com/aesara-devs/aesara/issues/637 is sorted.
    x = at.scalar("x")
    y = at.exp(x)
    with pytest.warns(None) as record:
        fn = get_jaxified_graph(inputs=[x], outputs=[y])
    assert not record
    assert np.isclose(fn(0), 1)


def test_get_log_likelihood():
    obs = np.random.normal(10, 2, size=100)
    obs_at = aesara.shared(obs, borrow=True, name="obs")
    with pm.Model() as model:
        a = pm.Normal("a", 0, 2)
        sigma = pm.HalfNormal("sigma")
        b = pm.Normal("b", a, sigma=sigma, observed=obs_at)

        trace = pm.sample(tune=10, draws=10, chains=2, random_seed=1322)

    b_true = trace.log_likelihood.b.values
    a = np.array(trace.posterior.a)
    sigma_log_ = np.log(np.array(trace.posterior.sigma))
    b_jax = _get_log_likelihood(model, [a, sigma_log_])["b"]

    assert np.allclose(b_jax.reshape(-1), b_true.reshape(-1))


def test_replace_shared_variables():
    x = aesara.shared(5, name="shared_x")

    new_x = _replace_shared_variables([x])
    shared_variables = [var for var in graph_inputs(new_x) if isinstance(var, SharedVariable)]
    assert not shared_variables

    x.default_update = x + 1
    with pytest.raises(ValueError, match="shared variables with default_update"):
        _replace_shared_variables([x])


def test_get_jaxified_logp():
    with pm.Model() as m:
        x = pm.Flat("x")
        y = pm.Flat("y")
        pm.Potential("pot", at.log(at.exp(x) + at.exp(y)))

    jax_fn = get_jaxified_logp(m)
    # This would underflow if not optimized
    assert not np.isinf(jax_fn((np.array(5000.0), np.array(5000.0))))


@pytest.mark.parametrize(
    "sampler",
    [
        sample_blackjax_nuts,
        sample_numpyro_nuts,
    ],
)
@pytest.mark.parametrize(
    "idata_kwargs",
    [
        dict(),
        dict(log_likelihood=False),
    ],
)
@pytest.mark.parametrize("postprocessing_backend", [None, "cpu"])
def test_idata_kwargs(sampler, idata_kwargs, postprocessing_backend):
    with pm.Model() as m:
        x = pm.Normal("x")
        z = pm.Normal("z")
        y = pm.Normal("y", x, observed=0)
        idata = sampler(
            tune=50,
            draws=50,
            chains=1,
            idata_kwargs=idata_kwargs,
            postprocessing_backend=postprocessing_backend,
        )

    if idata_kwargs.get("log_likelihood", True):
        assert "log_likelihood" in idata
    else:
        assert "log_likelihood" not in idata


def test_get_batched_jittered_initial_points():
    with pm.Model() as model:
        x = pm.MvNormal("x", mu=np.zeros(3), cov=np.eye(3), shape=(2, 3), initval=np.zeros((2, 3)))

    # No jitter
    ips = _get_batched_jittered_initial_points(
        model=model, chains=1, random_seed=1, initvals=None, jitter=False
    )
    assert np.all(ips[0] == 0)

    # Single chain
    ips = _get_batched_jittered_initial_points(model=model, chains=1, random_seed=1, initvals=None)

    assert ips[0].shape == (2, 3)
    assert np.all(ips[0] != 0)

    # Multiple chains
    ips = _get_batched_jittered_initial_points(model=model, chains=2, random_seed=1, initvals=None)

    assert ips[0].shape == (2, 2, 3)
    assert np.all(ips[0][0] != ips[0][1])


@pytest.mark.parametrize("random_seed", (None, 123))
@pytest.mark.parametrize("chains", (1, 2))
def test_seeding(chains, random_seed):
    sample_kwargs = dict(
        tune=100,
        draws=5,
        chains=chains,
        random_seed=random_seed,
    )

    with pm.Model(rng_seeder=456) as m:
        pm.Normal("x", mu=0, sigma=1)
        result1 = sample_numpyro_nuts(**sample_kwargs)

    with pm.Model(rng_seeder=456) as m:
        pm.Normal("x", mu=0, sigma=1)
        result2 = sample_numpyro_nuts(**sample_kwargs)
        result3 = sample_numpyro_nuts(**sample_kwargs)

    assert np.all(result1.posterior["x"] == result2.posterior["x"])
    expected_equal_result3 = random_seed is not None
    assert np.all(result2.posterior["x"] == result3.posterior["x"]) == expected_equal_result3

    if chains > 1:
        assert np.all(result1.posterior["x"].sel(chain=0) != result1.posterior["x"].sel(chain=1))
        assert np.all(result2.posterior["x"].sel(chain=0) != result2.posterior["x"].sel(chain=1))
        assert np.all(result3.posterior["x"].sel(chain=0) != result3.posterior["x"].sel(chain=1))
