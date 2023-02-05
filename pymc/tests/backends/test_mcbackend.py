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
import arviz
import numpy as np
import pytest

import pymc as pm

from pymc.backends import init_traces

try:
    import mcbackend as mcb
except ImportError:
    pytest.skip("Requires McBackend to be installed.")

from pymc.backends.mcbackend import ChainRecordAdapter, make_runmeta


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


def test_make_runmeta(simple_model):
    with simple_model:
        step = pm.DEMetropolisZ()
        dtypes = {rv.name: rv.dtype for rv in step.vars}
        shapes = {rv.name: rv.shape.eval() for rv in step.vars}
    rmeta = make_runmeta(
        var_dtypes=dtypes,
        var_shapes=shapes,
        step=step,
        model=simple_model,
    )
    assert isinstance(rmeta, mcb.RunMeta)
    assert len(rmeta.variables) == len(dtypes)
    assert len(rmeta.sample_stats) == 1 + len(step.stats_dtypes[0])
    pass


def test_init_traces(simple_model):
    with simple_model:
        step = pm.DEMetropolisZ()
        dtypes = {rv.name: rv.dtype for rv in step.vars}
        shapes = {rv.name: rv.shape.eval() for rv in step.vars}
    traces = init_traces(
        backend=mcb.NumPyBackend(),
        chains=2,
        expected_length=70,
        step=step,
        var_dtypes=dtypes,
        var_shapes=shapes,
        model=simple_model,
    )
    assert isinstance(traces, list)
    assert len(traces) == 2
    assert isinstance(traces[0], ChainRecordAdapter)
    assert isinstance(traces[0]._chain, mcb.backends.numpy.NumPyChain)
    pass


class TestMcBackendSampling:
    def test_multitrace_wrap(self, simple_model):
        with simple_model:
            mtrace = pm.sample(
                trace=mcb.NumPyBackend(),
                tune=5,
                draws=7,
                cores=1,
                chains=3,
                step=pm.Metropolis(),
                discard_tuned_samples=False,
                return_inferencedata=False,
            )
        assert isinstance(mtrace, pm.backends.base.MultiTrace)
        tune = mtrace._straces[0].get_sampler_stats("tune")
        assert isinstance(tune, np.ndarray)
        assert tune.shape == (12, 3)
        pass

    @pytest.mark.parametrize("cores", [1, 3])
    def test_simple_model(self, simple_model, cores):
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
