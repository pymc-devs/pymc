#   Copyright 2024 The PyMC Developers
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
import itertools

from dataclasses import asdict

import numpy as np
import pytest
import zarr

import pymc as pm

from pymc.backends.zarr import ZarrTrace
from pymc.stats.convergence import SamplerWarning
from pymc.step_methods import NUTS, CompoundStep, Metropolis
from pymc.step_methods.state import equal_dataclass_values
from tests.helpers import equal_sampling_states


@pytest.fixture(scope="module")
def model():
    time_int = np.array([np.timedelta64(np.timedelta64(i, "h"), "ns") for i in range(25)])
    coords = {
        "dim_int": range(3),
        "dim_str": ["A", "B"],
        "dim_time": np.datetime64("2024-10-16") + time_int,
        "dim_interval": time_int,
    }
    rng = np.random.default_rng(42)
    with pm.Model(coords=coords) as model:
        data1 = pm.Data("data1", np.ones(3, dtype="bool"), dims=["dim_int"])
        data2 = pm.Data("data2", np.ones(3, dtype="bool"))
        time = pm.Data("time", time_int / np.timedelta64(1, "h"), dims="dim_time")

        a = pm.Normal("a", shape=(len(coords["dim_int"]), len(coords["dim_str"])))
        b = pm.Normal("b", dims=["dim_int", "dim_str"])
        c = pm.Deterministic("c", a + b, dims=["dim_int", "dim_str"])

        d = pm.LogNormal("d", dims="dim_time")
        e = pm.Deterministic("e", (time + d)[:, None] + c[0], dims=["dim_interval", "dim_str"])

        obs = pm.Normal(
            "obs",
            mu=e,
            observed=rng.normal(size=(len(coords["dim_time"]), len(coords["dim_str"]))),
            dims=["dim_time", "dim_str"],
        )

    return model


@pytest.fixture(params=[True, False])
def include_transformed(request):
    return request.param


@pytest.fixture(params=["frequent_writes", "sparse_writes"])
def draws_per_chunk(request):
    spec = {
        "frequent_writes": 1,
        "sparse_writes": 7,
    }
    return spec[request.param]


@pytest.fixture(params=["single_step", "compound_step"])
def model_step(request, model):
    rng = np.random.default_rng(42)
    with model:
        if request.param == "single_step":
            step = NUTS(rng=rng)
        else:
            rngs = rng.spawn(2)
            step = CompoundStep(
                [
                    Metropolis(vars=model["a"], rng=rngs[0]),
                    NUTS(vars=[rv for rv in model.value_vars if rv.name != "a"], rng=rngs[1]),
                ]
            )
    return step


def test_record(model, model_step, include_transformed, draws_per_chunk):
    store = zarr.MemoryStore()
    trace = ZarrTrace(
        store=store, include_transformed=include_transformed, draws_per_chunk=draws_per_chunk
    )
    draws = 5
    tune = 5
    trace.init_trace(chains=1, draws=draws, tune=tune, model=model, step=model_step)

    # Assert that init was successful
    expected_groups = {
        "_sampling_state",
        "sample_stats",
        "posterior",
        "constant_data",
        "observed_data",
    }
    if include_transformed:
        expected_groups.add("unconstrained_posterior")
    assert {group_name for group_name, _ in trace.root.groups()} == expected_groups

    # Record samples from the ZarrChain
    manually_collected_warmup_draws = []
    manually_collected_warmup_stats = []
    manually_collected_draws = []
    manually_collected_stats = []
    point = model.initial_point()
    for draw in range(tune + draws):
        tuning = draw < tune
        if not tuning:
            model_step.stop_tuning()
        point, stats = model_step.step(point)
        if tuning:
            manually_collected_warmup_draws.append(point)
            manually_collected_warmup_stats.append(stats)
        else:
            manually_collected_draws.append(point)
            manually_collected_stats.append(stats)
        trace.straces[0].record(point, stats)
    trace.straces[0].record_sampling_state(model_step)
    assert {group_name for group_name, _ in trace.root.groups()} == expected_groups

    # Assert split warmup
    trace.split_warmup("posterior")
    trace.split_warmup("sample_stats")
    expected_groups = {
        "_sampling_state",
        "sample_stats",
        "posterior",
        "warmup_sample_stats",
        "warmup_posterior",
        "constant_data",
        "observed_data",
    }
    if include_transformed:
        trace.split_warmup("unconstrained_posterior")
        expected_groups.add("unconstrained_posterior")
        expected_groups.add("warmup_unconstrained_posterior")
    assert {group_name for group_name, _ in trace.root.groups()} == expected_groups
    # trace.consolidate()

    # Assert observed data is correct
    assert set(dict(trace.observed_data.arrays())) == {"obs", "dim_time", "dim_str"}
    assert list(trace.observed_data.obs.attrs["_ARRAY_DIMENSIONS"]) == ["dim_time", "dim_str"]
    np.testing.assert_array_equal(trace.observed_data.dim_time[:], model.coords["dim_time"])
    np.testing.assert_array_equal(trace.observed_data.dim_str[:], model.coords["dim_str"])

    # Assert constant data is correct
    assert set(dict(trace.constant_data.arrays())) == {
        "data1",
        "data2",
        "data2_dim_0",
        "time",
        "dim_time",
        "dim_int",
    }
    assert list(trace.constant_data.data1.attrs["_ARRAY_DIMENSIONS"]) == ["dim_int"]
    assert list(trace.constant_data.data2.attrs["_ARRAY_DIMENSIONS"]) == ["data2_dim_0"]
    assert list(trace.constant_data.time.attrs["_ARRAY_DIMENSIONS"]) == ["dim_time"]
    np.testing.assert_array_equal(trace.constant_data.dim_time[:], model.coords["dim_time"])
    np.testing.assert_array_equal(trace.constant_data.dim_int[:], model.coords["dim_int"])

    # Assert unconstrained posterior has correct shapes and kinds
    assert {rv.name for rv in model.free_RVs + model.deterministics} <= set(
        dict(trace.posterior.arrays())
    )
    if include_transformed:
        assert {"d_log__", "chain", "draw", "d_log___dim_0"} == set(
            dict(trace.unconstrained_posterior.arrays())
        )
        assert list(trace.unconstrained_posterior.d_log__.attrs["_ARRAY_DIMENSIONS"]) == [
            "chain",
            "draw",
            "d_log___dim_0",
        ]
        assert trace.unconstrained_posterior.d_log__.attrs["kind"] == "freeRV"
        np.testing.assert_array_equal(trace.unconstrained_posterior.chain, np.arange(1))
        np.testing.assert_array_equal(trace.unconstrained_posterior.draw, np.arange(draws))
        np.testing.assert_array_equal(
            trace.unconstrained_posterior.d_log___dim_0, np.arange(len(model.coords["dim_time"]))
        )

    # Assert posterior has correct shapes and kinds
    posterior_dims = set()
    for kind, rv_name in [
        (kind, rv.name)
        for kind, rv in itertools.chain(
            itertools.zip_longest([], model.free_RVs, fillvalue="freeRV"),
            itertools.zip_longest([], model.deterministics, fillvalue="deterministic"),
        )
    ]:
        if rv_name == "a":
            expected_dims = ["a_dim_0", "a_dim_1"]
        else:
            expected_dims = model.named_vars_to_dims[rv_name]
        posterior_dims |= set(expected_dims)
        assert list(trace.posterior[rv_name].attrs["_ARRAY_DIMENSIONS"]) == [
            "chain",
            "draw",
            *expected_dims,
        ]
        assert trace.posterior[rv_name].attrs["kind"] == kind
    for posterior_dim in posterior_dims:
        try:
            model_coord = model.coords[posterior_dim]
        except KeyError:
            model_coord = {
                "a_dim_0": np.arange(len(model.coords["dim_int"])),
                "a_dim_1": np.arange(len(model.coords["dim_str"])),
                "chain": np.arange(1),
                "draw": np.arange(draws),
            }[posterior_dim]
        np.testing.assert_array_equal(trace.posterior[posterior_dim][:], model_coord)

    # Assert sample stats have correct shape
    stats_bijection = trace.straces[0].stats_bijection
    for draw_idx, (draw, stat) in enumerate(
        zip(manually_collected_draws, manually_collected_stats)
    ):
        stat = stats_bijection.map(stat)
        for var, value in draw.items():
            if var in trace.posterior.arrays():
                assert np.array_equal(trace.posterior[var][0, draw_idx], value)
        for var, value in stat.items():
            sample_stats = trace.root["sample_stats"]
            stat_val = sample_stats[var][0, draw_idx]
            if not isinstance(stat_val, SamplerWarning):
                unequal_stats = stat_val != value
            else:
                unequal_stats = not equal_dataclass_values(asdict(stat_val), asdict(value))
            if unequal_stats and not (np.isnan(stat_val) and np.isnan(value)):
                raise AssertionError(f"{var} value does not match: {stat_val} != {value}")

    # Assert manually collected warmup samples match
    for draw_idx, (draw, stat) in enumerate(
        zip(manually_collected_warmup_draws, manually_collected_warmup_stats)
    ):
        stat = stats_bijection.map(stat)
        for var, value in draw.items():
            if var == "d_log__":
                if not include_transformed:
                    continue
                posterior = trace.root["warmup_unconstrained_posterior"]
            else:
                posterior = trace.root["warmup_posterior"]
            if var in posterior.arrays():
                assert np.array_equal(posterior[var][0, draw_idx], value)
        for var, value in stat.items():
            sample_stats = trace.root["warmup_sample_stats"]
            stat_val = sample_stats[var][0, draw_idx]
            if not isinstance(stat_val, SamplerWarning):
                unequal_stats = stat_val != value
            else:
                unequal_stats = not equal_dataclass_values(asdict(stat_val), asdict(value))
            if unequal_stats and not (np.isnan(stat_val) and np.isnan(value)):
                raise AssertionError(f"{var} value does not match: {stat_val} != {value}")

    # Assert manually collected posterior samples match
    for draw_idx, (draw, stat) in enumerate(
        zip(manually_collected_draws, manually_collected_stats)
    ):
        stat = stats_bijection.map(stat)
        for var, value in draw.items():
            if var == "d_log__":
                if not include_transformed:
                    continue
                posterior = trace.root["unconstrained_posterior"]
            else:
                posterior = trace.root["posterior"]
            if var in posterior.arrays():
                assert np.array_equal(posterior[var][0, draw_idx], value)
        for var, value in stat.items():
            sample_stats = trace.root["sample_stats"]
            stat_val = sample_stats[var][0, draw_idx]
            if not isinstance(stat_val, SamplerWarning):
                unequal_stats = stat_val != value
            else:
                unequal_stats = not equal_dataclass_values(asdict(stat_val), asdict(value))
            if unequal_stats and not (np.isnan(stat_val) and np.isnan(value)):
                raise AssertionError(f"{var} value does not match: {stat_val} != {value}")

    # Assert sampling_state is correct
    assert list(trace._sampling_state.draw_idx[:]) == [draws + tune]
    assert equal_sampling_states(
        trace._sampling_state.sampling_state[0],
        model_step.sampling_state,
    )

    # Assert to inference data returns the expected groups
    idata = trace.to_inferencedata(save_warmup=True)
    expected_groups = {
        "posterior",
        "constant_data",
        "observed_data",
        "sample_stats",
        "warmup_posterior",
        "warmup_sample_stats",
    }
    if include_transformed:
        expected_groups.add("unconstrained_posterior")
        expected_groups.add("warmup_unconstrained_posterior")
    assert set(idata.groups()) == expected_groups
    for group in idata.groups():
        for name, value in itertools.chain(
            idata[group].data_vars.items(), idata[group].coords.items()
        ):
            try:
                array = getattr(trace, group)[name][:]
            except AttributeError:
                array = trace.root[group][name][:]
            if "sample_stats" in group and "warning" in name:
                continue
            np.testing.assert_array_equal(array, value)


@pytest.mark.parametrize("tune", [0, 5, 10])
def test_split_warmup(tune, model, model_step, include_transformed):
    store = zarr.MemoryStore()
    trace = ZarrTrace(store=store, include_transformed=include_transformed)
    draws = 10 - tune
    trace.init_trace(chains=1, draws=draws, tune=tune, model=model, step=model_step)

    trace.split_warmup("posterior")
    trace.split_warmup("sample_stats")
    assert len(trace.root.posterior.draw) == draws
    assert len(trace.root.sample_stats.draw) == draws
    if tune == 0:
        with pytest.raises(KeyError):
            trace.root["warmup_posterior"]
    else:
        assert len(trace.root["warmup_posterior"].draw) == tune
        assert len(trace.root["warmup_sample_stats"].draw) == tune

        with pytest.raises(RuntimeError):
            trace.split_warmup("posterior")

        for var_name, posterior_array in trace.posterior.arrays():
            dims = posterior_array.attrs["_ARRAY_DIMENSIONS"]
            if len(dims) >= 2 and dims[1] == "draw":
                assert posterior_array.shape[1] == draws
                assert trace.root["warmup_posterior"][var_name].shape[1] == tune
        for var_name, sample_stats_array in trace.sample_stats.arrays():
            dims = sample_stats_array.attrs["_ARRAY_DIMENSIONS"]
            if len(dims) >= 2 and dims[1] == "draw":
                assert sample_stats_array.shape[1] == draws
                assert trace.root["warmup_sample_stats"][var_name].shape[1] == tune
