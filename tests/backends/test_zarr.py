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
import itertools

from dataclasses import asdict

import numpy as np
import pytest
import xarray as xr
import zarr

from arviz import InferenceData

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


@pytest.fixture(params=["include_transformed", "discard_transformed"])
def include_transformed(request):
    return request.param == "include_transformed"


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


@pytest.fixture(scope="function", params=["mid_tuning", "finished_tuning"])
def populated_trace(model, request):
    tune = 5
    draws = 5
    chains = 1
    if request.param == "mid_tuning":
        total_steps = 2
    else:
        total_steps = 7
    trace = ZarrTrace(
        draws_per_chunk=1,
        include_transformed=True,
    )
    with model:
        rng = np.random.default_rng(42)
        stepper = NUTS(rng=rng)
    trace.init_trace(
        chains=chains,
        draws=draws,
        tune=tune,
        step=stepper,
        model=model,
    )
    point = model.initial_point()
    for draw in range(total_steps):
        tuning = draw < tune
        if not tuning:
            stepper.stop_tuning()
        point, stats = stepper.step(point)
        trace.straces[0].record(point, stats)
    trace.straces[0].record_sampling_state(stepper)
    return trace, total_steps, tune, draws


def test_record(model, model_step, include_transformed, draws_per_chunk):
    store = zarr.TempStore()
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
        "warmup_sample_stats",
        "warmup_posterior",
        "constant_data",
        "observed_data",
    }
    if include_transformed:
        expected_groups.add("unconstrained_posterior")
        expected_groups.add("warmup_unconstrained_posterior")
    assert {group_name for group_name, _ in trace.root.groups()} == expected_groups

    # Record samples from the ZarrChain
    manually_collected_warmup_draws = []
    manually_collected_warmup_stats = []
    manually_collected_draws = []
    manually_collected_stats = []
    point = model.initial_point()
    divergences = 0
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
        for step_stats in stats:
            divergences += sum(
                int(step_stats[key] and not step_stats["tune"])
                for key in step_stats
                if "diverging" in key
            )
    assert trace.straces[0].completed_draws_and_divergences() == (draw + 1, divergences)
    last_point = point
    trace.straces[0].record_sampling_state(model_step)
    assert {group_name for group_name, _ in trace.root.groups()} == expected_groups

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
                np.testing.assert_array_equal(trace.posterior[var][0, draw_idx], value)
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
                np.testing.assert_array_equal(posterior[var][0, draw_idx], value)
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
                np.testing.assert_array_equal(posterior[var][0, draw_idx], value)
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
    assert set(last_point) == set(trace._sampling_state.mcmc_point.array_keys())
    for var_name, value in trace._sampling_state.mcmc_point.arrays():
        np.testing.assert_array_equal(last_point[var_name][None, ...], value)

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

    assert len(trace.root.posterior.draw) == draws
    assert len(trace.root.sample_stats.draw) == draws
    assert len(trace.root["warmup_posterior"].draw) == tune
    assert len(trace.root["warmup_sample_stats"].draw) == tune

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


@pytest.mark.parametrize(
    "desired_tune_and_draws",
    [
        [None, 1],
        [3, None],
        [10, None],
        [None, 10],
    ],
)
def test_resize(populated_trace, desired_tune_and_draws):
    desired_tune, desired_draws = desired_tune_and_draws
    trace, total_steps, tune, draws = populated_trace
    expect_to_fail = False
    failure_message = ""
    if desired_tune is not None:
        if total_steps > tune:
            expect_to_fail = True
            failure_message = (
                "Cannot change the number of tuning steps in the trace. "
                "Some chains have finished their tuning phase and have "
                "already performed steps in the posterior sampling regime."
            )
        elif total_steps > desired_tune:
            expect_to_fail = True
            failure_message = (
                "Cannot change the number of tuning steps in the trace. "
                "Some chains have already taken more steps than the desired number "
                "of tuning steps. Please increase the desired number of tuning "
                f"steps to at least {total_steps}."
            )
    if desired_draws is not None and total_steps > (desired_draws + tune):
        expect_to_fail = True
        failure_message = (
            "Cannot change the number of draws in the trace. "
            "Some chains have already taken more steps than the desired number "
            "of draws. Please increase the desired number of draws "
            f"to at least {total_steps - tune}."
        )
    if expect_to_fail:
        with pytest.raises(ValueError, match=failure_message):
            trace.resize(tune=desired_tune, draws=desired_draws)
    else:
        trace.resize(tune=desired_tune, draws=desired_draws)
        result_tune = desired_tune or tune
        result_draws = desired_draws or draws
        assert trace.tuning_steps == result_tune
        assert trace.draws == result_draws
        posterior_groups = ["posterior", "sample_stats", "unconstrained_posterior"]
        warmup_groups = [f"warmup_{name}" for name in posterior_groups]
        for group_set, expected_size in zip(
            [posterior_groups, warmup_groups], [result_draws, result_tune]
        ):
            for group in group_set:
                zarr_group = getattr(trace, group)
                for name, values in zarr_group.arrays():
                    if values.ndim > 1:  # Quick and dirty hack to filter out coordinate arrays
                        assert values.shape[1] == expected_size
                    elif name == "draw":
                        assert values.shape[0] == expected_size


@pytest.fixture(scope="function", params=["discard_tuning", "keep_tuning"])
def discard_tuned_samples(request):
    return request.param == "discard_tuning"


@pytest.fixture(scope="function", params=["return_idata", "return_zarr"])
def return_inferencedata(request):
    return request.param == "return_idata"


@pytest.fixture(
    scope="function", params=[True, False], ids=["keep_warning_stat", "discard_warning_stat"]
)
def keep_warning_stat(request):
    return request.param


@pytest.fixture(
    scope="function", params=[True, False], ids=["parallel_sampling", "sequential_sampling"]
)
def parallel(request):
    return request.param


@pytest.fixture(scope="function", params=[True, False], ids=["compute_loglike", "no_loglike"])
def log_likelihood(request):
    return request.param


def test_sample(
    model,
    model_step,
    include_transformed,
    discard_tuned_samples,
    return_inferencedata,
    keep_warning_stat,
    parallel,
    log_likelihood,
    draws_per_chunk,
):
    if not return_inferencedata and not log_likelihood:
        pytest.skip(
            reason="log_likelihood is only computed if an inference data object is returned"
        )
    store = zarr.TempStore()
    trace = ZarrTrace(
        store=store, include_transformed=include_transformed, draws_per_chunk=draws_per_chunk
    )
    tune = 2
    draws = 3
    if parallel:
        chains = 2
        cores = 2
    else:
        chains = 1
        cores = 1
    with model:
        out_trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            trace=trace,
            step=model_step,
            discard_tuned_samples=discard_tuned_samples,
            return_inferencedata=return_inferencedata,
            keep_warning_stat=keep_warning_stat,
            idata_kwargs={"log_likelihood": log_likelihood},
        )

    if not return_inferencedata:
        assert isinstance(out_trace, ZarrTrace)
        assert out_trace.root.store is trace.root.store
    else:
        assert isinstance(out_trace, InferenceData)

    expected_groups = {"posterior", "constant_data", "observed_data", "sample_stats"}
    if include_transformed:
        expected_groups |= {"unconstrained_posterior"}
    if not return_inferencedata or not discard_tuned_samples:
        expected_groups |= {"warmup_posterior", "warmup_sample_stats"}
        if include_transformed:
            expected_groups |= {"warmup_unconstrained_posterior"}
    if not return_inferencedata:
        expected_groups |= {"_sampling_state"}
    elif log_likelihood:
        expected_groups |= {"log_likelihood"}
    assert set(out_trace.groups()) == expected_groups

    if return_inferencedata:
        warning_stat = (
            "sampler_1__warning" if isinstance(model_step, CompoundStep) else "sampler_0__warning"
        )
        if keep_warning_stat:
            assert warning_stat in out_trace.sample_stats
        else:
            assert warning_stat not in out_trace.sample_stats

    # Assert that all variables have non empty samples (not NaNs)
    if return_inferencedata:
        assert all(
            (not np.any(np.isnan(v))) and v.shape[:2] == (chains, draws)
            for v in out_trace.posterior.data_vars.values()
        )
    else:
        dimensions = {*model.coords, "a_dim_0", "a_dim_1", "chain", "draw"}
        assert all(
            (not np.any(np.isnan(v[:]))) and v.shape[:2] == (chains, draws)
            for name, v in out_trace.posterior.arrays()
            if name not in dimensions
        )

    # Assert that the trace has valid sampling state stored for each chain
    for step_method_state in trace._sampling_state.sampling_state[:]:
        # We have no access to the actual step method that was using by each chain in pymc.sample
        # The best way to see if the step method state is valid is by trying to set
        # the model_step sampling state to the one stored in the trace.
        model_step.sampling_state = step_method_state


def test_sampling_consistency(
    model,
    model_step,
    draws_per_chunk,
):
    # Test that pm.sample will generate the same posterior and sampling state
    # regardless of whether sampling was done in parallel or not.
    store1 = zarr.TempStore()
    parallel_trace = ZarrTrace(
        store=store1, include_transformed=include_transformed, draws_per_chunk=draws_per_chunk
    )
    store2 = zarr.TempStore()
    sequential_trace = ZarrTrace(
        store=store2, include_transformed=include_transformed, draws_per_chunk=draws_per_chunk
    )
    tune = 2
    draws = 3
    chains = 2
    random_seed = 12345
    initial_step_state = model_step.sampling_state
    with model:
        parallel_idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            trace=parallel_trace,
            step=model_step,
            discard_tuned_samples=True,
            return_inferencedata=True,
            keep_warning_stat=False,
            idata_kwargs={"log_likelihood": False},
            random_seed=random_seed,
        )
        model_step.sampling_state = initial_step_state
        sequential_idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            trace=sequential_trace,
            step=model_step,
            discard_tuned_samples=True,
            return_inferencedata=True,
            keep_warning_stat=False,
            idata_kwargs={"log_likelihood": False},
            random_seed=random_seed,
        )
    for chain in range(chains):
        assert equal_sampling_states(
            parallel_trace._sampling_state.sampling_state[chain],
            sequential_trace._sampling_state.sampling_state[chain],
        )
    xr.testing.assert_equal(parallel_idata.posterior, sequential_idata.posterior)
