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
import pickle
import tempfile

from dataclasses import asdict

import numpy as np
import pytest
import xarray as xr
import zarr

from zarr.dtype import Struct

import pymc as pm

from pymc.backends.zarr import (
    OBJECT_CODEC_ATTR,
    ZarrTrace,
    decode_object_value,
    encode_object_value,
)
from pymc.pytensorf import make_shared_replacements
from pymc.stats.convergence import SamplerWarning, WarningType
from pymc.step_methods import NUTS, CompoundStep, Metropolis
from pymc.step_methods.arraystep import ArrayStepShared
from pymc.step_methods.state import equal_dataclass_values
from tests.helpers import equal_sampling_states


def dims(array) -> list:
    """Return the dimension names stored in the zarr v3 metadata of an array."""
    return list(array.metadata.dimension_names)


def assert_stat_value_matches(sample_stats, var, draw_idx, value):
    """Assert that a recorded stat matches its stored value.

    Warning stats are stored as typed columns derived from the warning objects
    (kind, message, level, step); the reconstruction is compared field-wise, so
    untypeable payloads (``extra``, divergence points) are not compared.
    """
    if f"{var}_type" in sample_stats:
        kind_code = int(sample_stats[f"{var}_type"][0, draw_idx])
        if value is None:
            assert kind_code == 0
            return
        step_code = int(sample_stats[f"{var}_step"][0, draw_idx])
        assert kind_code == value.kind.value
        assert str(sample_stats[f"{var}_message"][0, draw_idx]) == value.message
        assert str(sample_stats[f"{var}_level"][0, draw_idx]) == value.level
        expected_step = -1 if value.step is None else value.step
        assert step_code == expected_step
        return
    stat_val = sample_stats[var][0, draw_idx]
    if sample_stats[var].attrs.get(OBJECT_CODEC_ATTR):
        stat_val = decode_object_value(stat_val)
    if not isinstance(stat_val, SamplerWarning):
        unequal_stats = stat_val != value
    else:
        unequal_stats = not equal_dataclass_values(asdict(stat_val), asdict(value))
    if unequal_stats and not (np.isnan(stat_val) and np.isnan(value)):
        raise AssertionError(f"{var} value does not match: {stat_val} != {value}")


_temp_dirs: list[tempfile.TemporaryDirectory] = []


def make_store():
    """Create a temporary disk backed zarr store.

    A disk backed store is required for parallel sampling, where worker
    processes record their draws and sampling states to the shared store.
    """
    tmp = tempfile.TemporaryDirectory()
    _temp_dirs.append(tmp)
    return zarr.storage.LocalStore(tmp.name)


class WarnStepper(ArrayStepShared):
    """Step method that passes points through and always emits a fixed warning."""

    name = "warn_stepper"
    stats_dtypes_shapes = {
        "accepted": (bool, []),
        "warning": (SamplerWarning, None),
    }

    def __init__(self, vars, warning, shared, **kwargs):
        super().__init__(vars, shared=shared, **kwargs)
        self._warning = warning

    def astep(self, apoint, *args):
        return apoint, [{"accepted": True, "warning": self._warning}]


def test_warning_stat_stored_as_typed_columns():
    warning = SamplerWarning(
        WarningType.BAD_ACCEPTANCE,
        "The acceptance probability does not match the target.",
        "warn",
        step=3,
    )
    with pm.Model() as model:
        a = pm.Normal("a")
        ip = model.initial_point()
        shared = make_shared_replacements(ip, [a], model)
        step = WarnStepper([a], warning, shared)

        trace = ZarrTrace(store=make_store(), draws_per_chunk=1)
        trace.init_trace(chains=1, draws=3, tune=0, model=model, step=step)

        point = ip
        for _ in range(3):
            point, stats = step.step(point)
            trace.straces[0].record(point, stats, in_warmup=False)
        trace.straces[0].record_sampling_state(step)

    # The warning stat is stored as typed columns, not as a pickled object
    sample_stats = trace.root["sample_stats"]
    expected_arrays = {
        "chain",
        "draw",
        "sampler_0__accepted",
        "in_warmup",
        "sampler_0__warning_type",
        "sampler_0__warning_message",
        "sampler_0__warning_level",
        "sampler_0__warning_step",
    }
    assert set(dict(sample_stats.arrays())) == expected_arrays
    for name in expected_arrays:
        assert not sample_stats[name].attrs.get(OBJECT_CODEC_ATTR)
    np.testing.assert_array_equal(
        sample_stats["sampler_0__warning_type"][:][0],
        [int(WarningType.BAD_ACCEPTANCE.value)] * 3,
    )
    np.testing.assert_array_equal(sample_stats["sampler_0__warning_step"][:][0], [3] * 3)

    # The trace reconstructs the original warnings from the typed columns
    assert trace.warnings == [[warning, warning, warning]]


def test_to_datatree(model, model_step):
    trace = ZarrTrace(store=make_store())
    draws, tune, chains = 4, 2, 1
    trace.init_trace(chains=chains, draws=draws, tune=tune, model=model, step=model_step)

    point = model.initial_point()
    for draw in range(tune + draws):
        tuning = draw < tune
        if not tuning:
            model_step.stop_tuning()
        point, stats = model_step.step(point)
        trace.straces[0].record(point, stats, in_warmup=tuning)
    trace.straces[0].record_sampling_state(model_step)
    trace.sampling_time = 12.0

    dt = trace.to_datatree()
    assert isinstance(dt, xr.DataTree)
    # The tree mirrors the zarr group hierarchy, minus internal groups
    assert set(dt.children) == {"posterior", "sample_stats", "constant_data", "observed_data"}
    # Global sampling metadata is attached to each group
    for node in dt.children.values():
        assert node.attrs["tuning_steps"] == tune
        assert node.attrs["sampling_time"] == 12.0
    # Data is readable
    for var_name, var in dt["posterior"].data_vars.items():
        assert var.shape[:2] == (chains, draws)
    # And warmup groups are only attached if requested
    assert "warmup_posterior" not in dt.children
    dt = trace.to_datatree(save_warmup=True)
    assert "warmup_posterior" in dt.children
    assert dt["warmup_posterior"]["draw"].shape[0] == tune


def test_sampling_state_stored_as_struct():
    with pm.Model() as model:
        a = pm.Normal("a")
        ip = model.initial_point()
        rng = np.random.default_rng(1)
        step = NUTS(vars=[a], rng=rng)

        trace = ZarrTrace(store=make_store(), draws_per_chunk=1)
        trace.init_trace(chains=1, draws=2, tune=0, model=model, step=step)
        chain = trace.straces[0]
        chain.link_stepper(step)

        point = ip
        for _ in range(2):
            point, stats = step.step(point)
            chain.record(point, stats, in_warmup=False)
        chain.record_sampling_state(step)

    state_array = trace.root["_sampling_state"]["sampling_state"]
    # Not a pickled utf8 array, but a native zarr struct array
    assert not state_array.attrs.get(OBJECT_CODEC_ATTR)
    assert isinstance(state_array.metadata.data_type, Struct)

    # And the state roundtrips losslessly
    # (generator resume behavior is covered in tests/step_methods/test_state.py)
    assert equal_sampling_states(chain.sampling_state, step.sampling_state)


def test_pickle_protocol_stored_in_root_attrs():
    trace = ZarrTrace(store=make_store())
    # The root group records the pickle protocol used for object encoding
    assert trace.root.attrs["pymc_pickle_protocol"] == pickle.HIGHEST_PROTOCOL

    # decode_object_value honors an explicit protocol, defaulting to the
    # module-level one
    encoded = encode_object_value("some object")
    assert decode_object_value(encoded, protocol=pickle.HIGHEST_PROTOCOL) == "some object"
    assert decode_object_value(encoded) == "some object"


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


def test_record(model, model_step, include_transformed, draws_per_chunk):
    store = make_store()
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
        trace.straces[0].record(point, stats, in_warmup=tuning)
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
    assert list(dims(trace.observed_data["obs"])) == ["dim_time", "dim_str"]
    np.testing.assert_array_equal(trace.observed_data["dim_time"][:], model.coords["dim_time"])
    np.testing.assert_array_equal(trace.observed_data["dim_str"][:], model.coords["dim_str"])

    # Assert constant data is correct
    assert set(dict(trace.constant_data.arrays())) == {
        "data1",
        "data2",
        "data2_dim_0",
        "time",
        "dim_time",
        "dim_int",
    }
    assert list(dims(trace.constant_data["data1"])) == ["dim_int"]
    assert list(dims(trace.constant_data["data2"])) == ["data2_dim_0"]
    assert list(dims(trace.constant_data["time"])) == ["dim_time"]
    np.testing.assert_array_equal(trace.constant_data["dim_time"][:], model.coords["dim_time"])
    np.testing.assert_array_equal(trace.constant_data["dim_int"][:], model.coords["dim_int"])

    # Assert unconstrained posterior has correct shapes and kinds
    assert {rv.name for rv in model.free_RVs + model.deterministics} <= set(
        dict(trace.posterior.arrays())
    )
    if include_transformed:
        assert {"d_log__", "chain", "draw", "d_log___dim_0"} == set(
            dict(trace.unconstrained_posterior.arrays())
        )
        assert list(dims(trace.unconstrained_posterior["d_log__"])) == [
            "chain",
            "draw",
            "d_log___dim_0",
        ]
        assert trace.unconstrained_posterior["d_log__"].attrs["kind"] == "freeRV"
        np.testing.assert_array_equal(trace.unconstrained_posterior["chain"], np.arange(1))
        np.testing.assert_array_equal(trace.unconstrained_posterior["draw"], np.arange(draws))
        np.testing.assert_array_equal(
            trace.unconstrained_posterior["d_log___dim_0"],
            np.arange(len(model.coords["dim_time"])),
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
        assert list(dims(trace.posterior[rv_name])) == [
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
            assert_stat_value_matches(trace.root["sample_stats"], var, draw_idx, value)

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
            assert_stat_value_matches(trace.root["warmup_sample_stats"], var, draw_idx, value)

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
            assert_stat_value_matches(trace.root["sample_stats"], var, draw_idx, value)

    # Assert sampling_state is correct
    assert list(trace._sampling_state["draw_idx"][:]) == [draws + tune]
    assert equal_sampling_states(
        trace.straces[0].sampling_state,
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
    assert set(idata.children) == expected_groups
    for group in idata.children:
        for name, value in itertools.chain(
            idata[group].data_vars.items(), idata[group].coords.items()
        ):
            try:
                array = getattr(trace, group[1:])[name][:]
            except AttributeError:
                array = trace.root[group][name][:]
            if "sample_stats" in group and "warning" in name:
                continue
            np.testing.assert_array_equal(array, value)


@pytest.mark.parametrize("tune", [0, 5, 10])
def test_split_warmup(tune, model, model_step, include_transformed):
    store = make_store()
    trace = ZarrTrace(store=store, include_transformed=include_transformed)
    draws = 10 - tune
    trace.init_trace(chains=1, draws=draws, tune=tune, model=model, step=model_step)

    trace.split_warmup("posterior")
    trace.split_warmup("sample_stats")
    assert trace.root["posterior"]["draw"].shape[0] == draws
    assert trace.root["sample_stats"]["draw"].shape[0] == draws
    if tune == 0:
        with pytest.raises(KeyError):
            trace.root["warmup_posterior"]
    else:
        assert trace.root["warmup_posterior"]["draw"].shape[0] == tune
        assert trace.root["warmup_sample_stats"]["draw"].shape[0] == tune

        with pytest.raises(RuntimeError):
            trace.split_warmup("posterior")

        for var_name, posterior_array in trace.posterior.arrays():
            dims = posterior_array.metadata.dimension_names
            if len(dims) >= 2 and dims[1] == "draw":
                assert posterior_array.shape[1] == draws
                assert trace.root["warmup_posterior"][var_name].shape[1] == tune
        for var_name, sample_stats_array in trace.sample_stats.arrays():
            dims = sample_stats_array.metadata.dimension_names
            if len(dims) >= 2 and dims[1] == "draw":
                assert sample_stats_array.shape[1] == draws
                assert trace.root["warmup_sample_stats"][var_name].shape[1] == tune


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
    store = make_store()
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
        assert isinstance(out_trace, xr.DataTree)

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

    if return_inferencedata:
        expected_groups = {"/" + g for g in expected_groups} | {"/"}

        assert set(out_trace.groups) == expected_groups
    else:
        assert set(out_trace.groups()) == expected_groups

    if return_inferencedata:
        warning_stat = (
            "sampler_1__warning" if isinstance(model_step, CompoundStep) else "sampler_0__warning"
        )
        warning_stat = f"{warning_stat}_type"
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
    for strace in trace.straces:
        step_method_state = strace.sampling_state
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
    store1 = make_store()
    parallel_trace = ZarrTrace(
        store=store1, include_transformed=include_transformed, draws_per_chunk=draws_per_chunk
    )
    store2 = make_store()
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
            parallel_trace.straces[chain].sampling_state,
            sequential_trace.straces[chain].sampling_state,
        )
    xr.testing.assert_equal(parallel_idata.posterior, sequential_idata.posterior)
