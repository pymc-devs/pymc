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
import logging

import arviz
import numpy as np
import pytest

import pymc as pm

from pymc.backends import init_traces
from pymc.step_methods.arraystep import ArrayStepShared

try:
    import mcbackend as mcb

    from mcbackend.npproto.utils import ndarray_to_numpy
except ImportError:
    pytest.skip("Requires McBackend to be installed.")

from pymc.backends.mcbackend import (
    ChainRecordAdapter,
    find_data,
    get_variables_and_point_fn,
    make_runmeta_and_point_fn,
)


@pytest.fixture
def simple_model():
    seconds = np.linspace(0, 5)
    observations = np.random.normal(0.5 + np.random.uniform(size=3)[:, None] * seconds[None, :])
    with pm.Model(
        coords={
            "condition": ["A", "B", "C"],
        }
    ) as pmodel:
        x = pm.ConstantData("seconds", seconds, dims="time")
        a = pm.Normal("scalar")
        b = pm.Uniform("vector", dims="condition")
        pm.Deterministic("matrix", a + b[:, None] * x[None, :], dims=("condition", "time"))
        pm.Bernoulli("integer", p=0.5)
        obs = pm.MutableData("obs", observations, dims=("condition", "time"))
        pm.Normal("L", pmodel["matrix"], observed=obs, dims=("condition", "time"))
    return pmodel


def test_find_data(simple_model):
    dvars = find_data(simple_model)
    dvardict = {d.name: d for d in dvars}
    assert set(dvardict) == {"seconds", "obs"}

    secs = dvardict["seconds"]
    assert isinstance(secs, mcb.DataVariable)
    assert secs.dims == ["time"]
    assert not secs.is_observed
    np.testing.assert_array_equal(ndarray_to_numpy(secs.value), simple_model["seconds"].data)

    obs = dvardict["obs"]
    assert isinstance(obs, mcb.DataVariable)
    assert obs.dims == ["condition", "time"]
    assert obs.is_observed
    np.testing.assert_array_equal(ndarray_to_numpy(obs.value), simple_model["obs"].get_value())


def test_find_data_skips_deterministics():
    data = np.array([0, 1], dtype="float32")
    with pm.Model() as pmodel:
        a = pm.ConstantData("a", data, dims="item")
        b = pm.Normal("b")
        pm.Deterministic("c", a + b, dims="item")
    assert "c" in pmodel.named_vars
    dvars = find_data(pmodel)
    assert len(dvars) == 1
    assert dvars[0].name == "a"
    assert dvars[0].dims == ["item"]
    np.testing.assert_array_equal(ndarray_to_numpy(dvars[0].value), data)
    assert not dvars[0].is_observed


def test_get_variables_and_point_fn(simple_model):
    ip = simple_model.initial_point()
    variables, point_fn = get_variables_and_point_fn(simple_model, ip)
    assert isinstance(variables, list)
    assert callable(point_fn)
    vdict = {v.name: v for v in variables}
    assert set(vdict) == {"integer", "scalar", "vector", "vector_interval__", "matrix"}
    point = point_fn(ip)
    assert len(point) == len(variables)
    for v, p in zip(variables, point):
        assert str(p.dtype) == v.dtype


def test_make_runmeta_and_point_fn(simple_model):
    with simple_model:
        step = pm.DEMetropolisZ()
    rmeta, point_fn = make_runmeta_and_point_fn(
        initial_point=simple_model.initial_point(),
        step=step,
        model=simple_model,
    )
    assert isinstance(rmeta, mcb.RunMeta)
    assert callable(point_fn)
    vars = {v.name: v for v in rmeta.variables}
    assert set(vars.keys()) == {"scalar", "vector", "vector_interval__", "matrix", "integer"}
    # NOTE: Technically the "vector" is deterministic, but from the user perspective it is not.
    #       This is merely a matter of which version of transformed variables should be traced.
    assert not vars["vector"].is_deterministic
    assert not vars["vector_interval__"].is_deterministic
    assert vars["matrix"].is_deterministic
    assert len(rmeta.sample_stats) == len(step.stats_dtypes[0])

    with simple_model:
        step = pm.NUTS()
    rmeta, point_fn = make_runmeta_and_point_fn(
        initial_point=simple_model.initial_point(),
        step=step,
        model=simple_model,
    )
    assert isinstance(rmeta, mcb.RunMeta)
    svars = {s.name: s for s in rmeta.sample_stats}
    # Unbeknownst to McBackend, object stats are pickled to str
    assert "sampler_0__warning" in svars
    assert svars["sampler_0__warning"].dtype == "str"
    pass


def test_init_traces(simple_model):
    with simple_model:
        step = pm.DEMetropolisZ()
    run, traces = init_traces(
        backend=mcb.NumPyBackend(),
        chains=2,
        expected_length=70,
        step=step,
        initial_point=simple_model.initial_point(),
        model=simple_model,
    )
    assert isinstance(run, mcb.backends.numpy.NumPyRun)
    assert isinstance(traces, list)
    assert len(traces) == 2
    assert isinstance(traces[0], ChainRecordAdapter)
    assert isinstance(traces[0]._chain, mcb.backends.numpy.NumPyChain)
    pass


class ToyStepper(ArrayStepShared):
    stats_dtypes_shapes = {
        "accepted": (bool, []),
        "tune": (bool, []),
        "s1": (np.float64, []),
    }

    def astep(self, *args, **kwargs):
        raise NotImplementedError()


class ToyStepperWithOtherStats(ToyStepper):
    stats_dtypes_shapes = {
        "accepted": (bool, []),
        "tune": (bool, []),
        "s2": (np.float64, []),
    }


class TestChainRecordAdapter:
    def test_get_sampler_stats(self):
        # Initialize a very simply toy model
        N = 45
        with pm.Model() as pmodel:
            a = pm.Normal("a")
            b = pm.Uniform("b")
            c = pm.Deterministic("c", a + b)
            ip = pmodel.initial_point()
            shared = pm.make_shared_replacements(ip, [a, b], pmodel)
            run, traces = init_traces(
                backend=mcb.NumPyBackend(),
                chains=1,
                expected_length=N,
                step=ToyStepper([a, b], shared),
                initial_point=pmodel.initial_point(),
                model=pmodel,
            )
        cra = traces[0]
        assert isinstance(run, mcb.backends.numpy.NumPyRun)
        assert isinstance(cra, ChainRecordAdapter)

        # Simulate recording of draws and stats
        rng = np.random.RandomState(2023)
        for i in range(N):
            draw = {"a": rng.normal(), "b_interval__": rng.normal()}
            stats = [dict(tune=(i <= 5), s1=i, accepted=bool(rng.randint(0, 2)))]
            cra.record(draw, stats)

        # Check final state of the chain
        assert len(cra) == N
        # Variables b and c were calculated by the point function
        draws_a = cra.get_values("a")
        draws_b = cra.get_values("b")
        draws_c = cra.get_values("c")
        np.testing.assert_array_equal(draws_a + draws_b, draws_c)
        i = np.random.randint(0, N)
        point = cra.point(idx=i)
        assert point["a"] == draws_a[i]
        assert point["b"] == draws_b[i]
        assert point["c"] == draws_c[i]

        # Stats come in different shapes depending on the query
        s1 = cra.get_sampler_stats("s1", sampler_idx=None, burn=3, thin=2)
        assert s1.shape == (21,)
        assert s1.dtype == np.dtype("float64")
        np.testing.assert_array_equal(s1, np.arange(N)[3:None:2])

    def test_get_sampler_stats_compound(self, caplog):
        # Initialize a very simply toy model
        N = 45
        with pm.Model() as pmodel:
            a = pm.Normal("a")
            b = pm.Uniform("b")
            c = pm.Deterministic("c", a + b)
            ip = pmodel.initial_point()
            shared_a = pm.make_shared_replacements(ip, [a], pmodel)
            shared_b = pm.make_shared_replacements(ip, [b], pmodel)
            stepA = ToyStepper([a], shared_a)
            stepB = ToyStepperWithOtherStats([b], shared_b)
            run, traces = init_traces(
                backend=mcb.NumPyBackend(),
                chains=1,
                expected_length=N,
                step=pm.CompoundStep([stepA, stepB]),
                initial_point=pmodel.initial_point(),
                model=pmodel,
            )
        cra = traces[0]
        assert isinstance(cra, ChainRecordAdapter)

        # Simulate recording of draws and stats
        rng = np.random.RandomState(2023)
        for i in range(N):
            tune = i <= 5
            draw = {"a": rng.normal(), "b_interval__": rng.normal()}
            stats = [
                dict(tune=tune, s1=i, accepted=bool(rng.randint(0, 2))),
                dict(tune=tune, s2=i, accepted=bool(rng.randint(0, 2))),
            ]
            cra.record(draw, stats)

        # The 'accepted' stat was emitted by both samplers
        assert cra.get_sampler_stats("accepted", sampler_idx=None).shape == (N, 2)
        acpt_1 = cra.get_sampler_stats("accepted", sampler_idx=0, burn=3, thin=2)
        acpt_2 = cra.get_sampler_stats("accepted", sampler_idx=1, burn=3, thin=2)
        assert acpt_1.shape == (21,)  # (N-3)/2
        assert not np.array_equal(acpt_1, acpt_2)

        # s1 and s2 were sampler specific
        # they are squeezed into vectors, but warnings are logged at DEBUG level
        with caplog.at_level(logging.DEBUG, logger="pymc"):
            s1 = cra.get_sampler_stats("s1", burn=10)
            assert s1.shape == (35,)
            assert s1.dtype == np.dtype("float64")
            s2 = cra.get_sampler_stats("s2", thin=5)
            assert s2.shape == (9,)  # N/5
            assert s2.dtype == np.dtype("float64")
        assert any("'s1' was not recorded by all samplers" in r.message for r in caplog.records)

        with pytest.raises(KeyError, match="No stat"):
            cra.get_sampler_stats("notastat")


class TestMcBackendSampling:
    @pytest.mark.parametrize("discard_warmup", [False, True])
    def test_return_multitrace(self, simple_model, discard_warmup):
        with simple_model:
            mtrace = pm.sample(
                trace=mcb.NumPyBackend(),
                tune=5,
                draws=7,
                cores=1,
                chains=3,
                step=pm.Metropolis(),
                discard_tuned_samples=discard_warmup,
                return_inferencedata=False,
            )
        assert isinstance(mtrace, pm.backends.base.MultiTrace)
        tune = mtrace._straces[0].get_sampler_stats("tune")
        assert isinstance(tune, np.ndarray)
        if discard_warmup:
            assert tune.shape == (7, 3)
        else:
            assert tune.shape == (12, 3)
        pass

    @pytest.mark.parametrize("cores", [1, 3])
    def test_return_inferencedata(self, simple_model, cores):
        with simple_model:
            idata = pm.sample(
                trace=mcb.NumPyBackend(),
                tune=5,
                draws=7,
                cores=cores,
                chains=3,
                discard_tuned_samples=False,
            )
        assert isinstance(idata, arviz.InferenceData)
        assert idata.warmup_posterior.sizes["draw"] == 5
        assert idata.posterior.sizes["draw"] == 7
        pass
