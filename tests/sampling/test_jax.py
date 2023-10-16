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
import warnings

from typing import Any, Callable, Dict, Optional
from unittest import mock

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpyro.infer import MCMC
from pytensor.compile import SharedVariable
from pytensor.graph import graph_inputs

import pymc as pm

from pymc.distributions.multivariate import PosDefMatrix
from pymc.sampling.jax import (
    _get_batched_jittered_initial_points,
    _get_log_likelihood,
    _numpyro_nuts_defaults,
    _replace_shared_variables,
    _update_numpyro_nuts_kwargs,
    get_jaxified_graph,
    get_jaxified_logp,
    sample_blackjax_nuts,
    sample_numpyro_nuts,
)


def test_old_import_route():
    import pymc.sampling.jax as new_sj
    import pymc.sampling_jax as old_sj

    assert set(new_sj.__all__) <= set(dir(old_sj))


def test_jax_PosDefMatrix():
    x = pt.tensor(name="x", shape=(2, 2), dtype="float32")
    matrix_pos_def = PosDefMatrix()
    x_is_pos_def = matrix_pos_def(x)
    f = pytensor.function(inputs=[x], outputs=[x_is_pos_def], mode="JAX")

    test_cases = [
        (jnp.eye(2), True),
        (jnp.zeros(shape=(2, 2)), False),
        (jnp.array([[1, -1.5], [0, 1.2]], dtype="float32"), True),
        (-1 * jnp.array([[1, -1.5], [0, 1.2]], dtype="float32"), False),
        (jnp.array([[1, -1.5], [0, -1.2]], dtype="float32"), False),
    ]

    for input, expected in test_cases:
        actual = f(input)[0]
        assert jnp.array_equal(a1=actual, a2=expected)


@pytest.mark.parametrize(
    "sampler",
    [
        sample_blackjax_nuts,
        sample_numpyro_nuts,
    ],
)
@pytest.mark.parametrize("postprocessing_backend", [None, "cpu"])
@pytest.mark.parametrize(
    "chains",
    [
        pytest.param(1),
        pytest.param(
            2, marks=pytest.mark.skipif(len(jax.devices()) < 2, reason="not enough devices")
        ),
    ],
)
@pytest.mark.parametrize("postprocessing_vectorize", ["scan", "vmap"])
def test_transform_samples(sampler, postprocessing_backend, chains, postprocessing_vectorize):
    pytensor.config.on_opt_error = "raise"
    np.random.seed(13244)

    obs = np.random.normal(10, 2, size=100)
    obs_at = pytensor.shared(obs, borrow=True, name="obs")
    with pm.Model() as model:
        a = pm.Uniform("a", -20, 20)
        sigma = pm.HalfNormal("sigma", shape=(2,))
        b = pm.Normal("b", a, sigma=sigma.mean(), observed=obs_at)

        trace = sampler(
            chains=chains,
            random_seed=1322,
            keep_untransformed=True,
            postprocessing_backend=postprocessing_backend,
            postprocessing_vectorize=postprocessing_vectorize,
        )

    log_vals = trace.posterior["sigma_log__"].values

    trans_vals = trace.posterior["sigma"].values
    assert np.allclose(np.exp(log_vals), trans_vals)

    assert 8 < trace.posterior["a"].mean() < 11
    assert 1.5 < trace.posterior["sigma"].mean() < 2.5

    obs_at.set_value(-obs)
    with model:
        trace = sampler(
            chains=chains,
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
@pytest.mark.skipif(len(jax.devices()) < 2, reason="not enough devices")
def test_deterministic_samples(sampler):
    pytensor.config.on_opt_error = "raise"
    np.random.seed(13244)

    obs = np.random.normal(10, 2, size=100)
    obs_at = pytensor.shared(obs, borrow=True, name="obs")
    with pm.Model() as model:
        a = pm.Uniform("a", -20, 20)
        b = pm.Deterministic("b", a / 2.0)
        c = pm.Normal("c", a, sigma=1.0, observed=obs_at)

        trace = sampler(chains=2, random_seed=1322, keep_untransformed=True)

    assert 8 < trace.posterior["a"].mean() < 11
    assert np.allclose(trace.posterior["b"].values, trace.posterior["a"].values / 2)


def test_get_jaxified_graph():
    # Check that jaxifying a graph does not emit the Supervisor Warning. This test can
    # be removed once https://github.com/aesara-devs/aesara/issues/637 is sorted.
    x = pt.scalar("x")
    y = pt.exp(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fn = get_jaxified_graph(inputs=[x], outputs=[y])
    assert np.isclose(fn(0), 1)


def test_get_log_likelihood():
    obs = np.random.normal(10, 2, size=100)
    obs_at = pytensor.shared(obs, borrow=True, name="obs")
    with pm.Model() as model:
        a = pm.Normal("a", 0, 2)
        sigma = pm.HalfNormal("sigma")
        b = pm.Normal("b", a, sigma=sigma, observed=obs_at)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            trace = pm.sample(
                tune=10,
                draws=10,
                chains=2,
                random_seed=1322,
                idata_kwargs=dict(log_likelihood=True),
            )

    b_true = trace.log_likelihood.b.values
    a = np.array(trace.posterior.a)
    sigma_log_ = np.log(np.array(trace.posterior.sigma))
    b_jax = _get_log_likelihood(model, [a, sigma_log_])["b"]

    assert np.allclose(b_jax.reshape(-1), b_true.reshape(-1))


def test_replace_shared_variables():
    x = pytensor.shared(5, name="shared_x")

    new_x = _replace_shared_variables([x])
    shared_variables = [var for var in graph_inputs(new_x) if isinstance(var, SharedVariable)]
    assert not shared_variables

    x.default_update = x + 1
    with pytest.raises(ValueError, match="shared variables with default_update"):
        _replace_shared_variables([x])

    shared_rng = pytensor.shared(np.random.default_rng(), name="shared_rng")
    x = pytensor.tensor.random.normal(rng=shared_rng)
    with pytest.raises(ValueError, match="Graph contains shared RandomType variables"):
        _replace_shared_variables([x])


def test_get_jaxified_logp():
    with pm.Model() as m:
        x = pm.Flat("x")
        y = pm.Flat("y")
        pm.Potential("pot", pt.log(pt.exp(x) + pt.exp(y)))

    jax_fn = get_jaxified_logp(m)
    # This would underflow if not optimized
    assert not np.isinf(jax_fn((np.array(5000.0), np.array(5000.0))))


@pytest.fixture(scope="module")
def model_test_idata_kwargs() -> pm.Model:
    with pm.Model(
        coords={"x_coord": ["a", "b"], "x_coord2": [1, 2], "z_coord": ["apple", "banana", "orange"]}
    ) as m:
        x = pm.Normal("x", shape=(2,), dims=["x_coord"])
        _ = pm.Normal("y", x, observed=[0, 0])
        _ = pm.Normal("z", 0, 1, dims="z_coord")
        pm.ConstantData("constantdata", [1, 2, 3])
        pm.MutableData("mutabledata", 2)
    return m


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
        dict(log_likelihood=True),
        # Overwrite models coords
        dict(coords={"x_coord": ["x1", "x2"]}),
        # Overwrite dims from dist specification in model
        dict(dims={"x": ["x_coord2"]}),
        # Overwrite both coords and dims
        dict(coords={"x_coord3": ["A", "B"]}, dims={"x": ["x_coord3"]}),
    ],
)
@pytest.mark.parametrize("postprocessing_backend", [None, "cpu"])
def test_idata_kwargs(
    model_test_idata_kwargs: pm.Model,
    sampler: Callable[..., az.InferenceData],
    idata_kwargs: Dict[str, Any],
    postprocessing_backend: Optional[str],
):
    idata: Optional[az.InferenceData] = None
    with model_test_idata_kwargs:
        idata = sampler(
            tune=50,
            draws=50,
            chains=1,
            idata_kwargs=idata_kwargs,
            postprocessing_backend=postprocessing_backend,
        )
    assert idata is not None
    const_data = idata.get("constant_data")
    assert const_data is not None
    assert "constantdata" in const_data
    assert "mutabledata" in const_data

    if idata_kwargs.get("log_likelihood", False):
        assert "log_likelihood" in idata
    else:
        assert "log_likelihood" not in idata

    posterior = idata.get("posterior")
    assert posterior is not None
    x_dim_expected = idata_kwargs.get("dims", model_test_idata_kwargs.named_vars_to_dims)["x"][0]
    assert x_dim_expected is not None
    assert posterior["x"].dims[-1] == x_dim_expected

    x_coords_expected = idata_kwargs.get("coords", model_test_idata_kwargs.coords)[x_dim_expected]
    assert x_coords_expected is not None
    assert list(x_coords_expected) == list(posterior["x"].coords[x_dim_expected].values)

    assert posterior["z"].dims[2] == "z_coord"
    assert np.all(
        posterior["z"].coords["z_coord"].values == np.array(["apple", "banana", "orange"])
    )


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


@pytest.mark.parametrize(
    "sampler",
    [
        sample_blackjax_nuts,
        sample_numpyro_nuts,
    ],
)
@pytest.mark.parametrize("random_seed", (None, 123))
@pytest.mark.parametrize(
    "chains",
    [
        pytest.param(1),
        pytest.param(
            2, marks=pytest.mark.skipif(len(jax.devices()) < 2, reason="not enough devices")
        ),
    ],
)
def test_seeding(chains, random_seed, sampler):
    sample_kwargs = dict(
        tune=100,
        draws=5,
        chains=chains,
        random_seed=random_seed,
    )

    with pm.Model() as m:
        pm.Normal("x", mu=0, sigma=1)
        result1 = sampler(**sample_kwargs)
        result2 = sampler(**sample_kwargs)

    all_equal = np.all(result1.posterior["x"] == result2.posterior["x"])
    if random_seed is None:
        assert not all_equal
    else:
        assert all_equal

    if chains > 1:
        assert np.all(result1.posterior["x"].sel(chain=0) != result1.posterior["x"].sel(chain=1))
        assert np.all(result2.posterior["x"].sel(chain=0) != result2.posterior["x"].sel(chain=1))


@pytest.mark.parametrize(
    "nuts_kwargs",
    [
        {"adapt_step_size": False},
        {"adapt_mass_matrix": True},
        {"dense_mass": True},
        {"adapt_step_size": False, "adapt_mass_matrix": True, "dense_mass": True},
        {"fake-key": "fake-value"},
    ],
)
def test_update_numpyro_nuts_kwargs(nuts_kwargs: Dict[str, Any]):
    original_kwargs = nuts_kwargs.copy()
    new_kwargs = _update_numpyro_nuts_kwargs(nuts_kwargs)

    # Maintains original key-value pairs.
    for k, v in original_kwargs.items():
        assert new_kwargs[k] == v

    for k, v in _numpyro_nuts_defaults().items():
        if k not in original_kwargs:
            assert new_kwargs[k] == v


@mock.patch("numpyro.infer.MCMC")
def test_numpyro_nuts_kwargs_are_used(mocked: mock.MagicMock):
    mocked.side_effect = MCMC

    step_size = 0.13
    dense_mass = True
    adapt_step_size = False
    target_accept = 0.78

    with pm.Model():
        pm.Normal("a")
        sample_numpyro_nuts(
            10,
            tune=10,
            chains=1,
            target_accept=target_accept,
            nuts_kwargs={
                "step_size": step_size,
                "dense_mass": dense_mass,
                "adapt_step_size": adapt_step_size,
            },
        )
    mocked.assert_called_once()
    nuts_sampler = mocked.call_args.args[0]
    assert nuts_sampler._step_size == step_size
    assert nuts_sampler._dense_mass == dense_mass
    assert nuts_sampler._adapt_step_size == adapt_step_size
    assert nuts_sampler._adapt_mass_matrix
    assert nuts_sampler._target_accept_prob == target_accept


@pytest.mark.parametrize(
    "sampler_name",
    [
        "sample_blackjax_nuts",
        "sample_numpyro_nuts",
    ],
)
def test_idata_contains_stats(sampler_name: str):
    """Tests whether sampler statistics were written to sample_stats
    group of InferenceData"""
    if sampler_name == "sample_blackjax_nuts":
        sampler = sample_blackjax_nuts
    elif sampler_name == "sample_numpyro_nuts":
        sampler = sample_numpyro_nuts

    with pm.Model():
        pm.Normal("a")
        idata = sampler(tune=50, draws=50)

    stats = idata.get("sample_stats")
    assert stats is not None
    n_chains = stats.dims["chain"]
    n_draws = stats.dims["draw"]

    # Stats vars expected for both samplers
    expected_stat_vars = {
        "acceptance_rate": (n_chains, n_draws),
        "diverging": (n_chains, n_draws),
        "energy": (n_chains, n_draws),
        "tree_depth": (n_chains, n_draws),
        "lp": (n_chains, n_draws),
    }
    # Stats only expected for blackjax nuts
    if sampler_name == "sample_blackjax_nuts":
        blackjax_special_vars = {}
        stat_vars = expected_stat_vars | blackjax_special_vars
    # Stats only expected for numpyro nuts
    elif sampler_name == "sample_numpyro_nuts":
        numpyro_special_vars = {
            "step_size": (n_chains, n_draws),
            "n_steps": (n_chains, n_draws),
        }
        stat_vars = expected_stat_vars | numpyro_special_vars
    # test existence and dimensionality
    for stat_var, stat_var_dims in stat_vars.items():
        assert stat_var in stats.variables
        assert stats.get(stat_var).values.shape == stat_var_dims
