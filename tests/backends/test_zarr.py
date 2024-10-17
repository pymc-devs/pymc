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
import numpy as np
import pytest
import zarr

import pymc as pm

from pymc.backends.zarr import ZarrTrace
from pymc.step_methods import NUTS, CompoundStep, Metropolis
from tests.helpers import equal_sampling_states


@pytest.fixture(scope="module")
def model():
    time_int = np.array([np.timedelta64(i, "h") for i in range(25)])
    coords = {
        "dim_int": range(3),
        "dim_str": ["A", "B"],
        "dim_time": np.datetime64("2024-10-16") + time_int,
        "dim_interval": time_int,
    }
    rng = np.random.default_rng(42)
    with pm.Model(coords=coords) as model:
        data1 = pm.Data("data1", np.ones(3, dtype="bool"), dims=["dim_int"])
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


def test_record(model, model_step, include_transformed):
    store = zarr.MemoryStore()
    trace = ZarrTrace(store=store, model=model, include_transformed=include_transformed)
    draws = 10
    trace.init_trace(chains=1, draws=draws, step=model_step)

    # Assert that init was successful
    expected_groups = {
        "_sampling_state",
        "sample_stats",
        "posterior",
        "constant_data",
        "observed_data",
    }
    assert {group_name for group_name, _ in trace.root.groups()} == expected_groups

    # Record samples from the ZarrChain
    manually_collected_draws = []
    manually_collected_stats = []
    point = model.initial_point()
    for draw in range(draws):
        point, stats = model_step.step(point)
        manually_collected_draws.append(point)
        manually_collected_stats.append(stats)
        trace.straces[0].record(point, stats)
    trace.straces[0].record_sampling_state(model_step)
    assert {group_name for group_name, _ in trace.root.groups()} == expected_groups
    # trace.consolidate()

    # Assert observed data is correct
    assert set(dict(trace.observed_data.arrays())) == {"obs", "dim_time", "dim_str"}
    assert list(trace.observed_data.obs.attrs["_ARRAY_DIMENSIONS"]) == ["dim_time", "dim_str"]
    np.testing.assert_array_equal(trace.observed_data.dim_time[:], model.coords["dim_time"])
    np.testing.assert_array_equal(trace.observed_data.dim_str[:], model.coords["dim_str"])

    # Assert constant data is correct
    assert set(dict(trace.constant_data.arrays())) == {"data1", "time", "dim_time", "dim_int"}
    assert list(trace.constant_data.data1.attrs["_ARRAY_DIMENSIONS"]) == ["dim_int"]
    assert list(trace.constant_data.time.attrs["_ARRAY_DIMENSIONS"]) == ["dim_time"]
    np.testing.assert_array_equal(trace.constant_data.dim_time[:], model.coords["dim_time"])
    np.testing.assert_array_equal(trace.constant_data.dim_int[:], model.coords["dim_int"])

    # Assert posterior and sample_stats are correct
    if include_transformed:
        assert {rv.name for rv in model.unobserved_value_vars + model.deterministics} <= set(
            dict(trace.posterior.arrays())
        )
    else:
        assert {rv.name for rv in model.free_RVs + model.deterministics} <= set(
            dict(trace.posterior.arrays())
        )
    posterior_dims = set()
    for rv_name in [rv.name for rv in model.free_RVs + model.deterministics]:
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

    stats_bijection = trace.straces[0].stats_bijection
    for draw_idx, (draw, stat) in enumerate(
        zip(manually_collected_draws, manually_collected_stats)
    ):
        stat = stats_bijection.map(stat)
        for var, value in draw.items():
            if var in trace.posterior.arrays():
                assert np.array_equal(trace.posterior[var][0, draw_idx], value)
        for var, value in stat.items():
            stat_val = trace.sample_stats[var][0, draw_idx]
            if stat_val != value:
                if not (np.isnan(stat_val) and np.isnan(value)):
                    raise AssertionError(f"{stat_val} != {value}")

    # Assert sampling_state is correct
    assert list(trace._sampling_state.draw_idx[:]) == [draws]
    assert equal_sampling_states(
        trace._sampling_state.sampling_state[0],
        model_step.sampling_state,
    )

    idata = trace.to_inferencedata()
    assert set(idata.groups()) == {"posterior", "constant_data", "observed_data", "sample_stats"}
    for group in idata.groups():
        for name, value in idata[group].data_vars.items():
            np.testing.assert_array_equal(getattr(trace, group)[name][:], value)
