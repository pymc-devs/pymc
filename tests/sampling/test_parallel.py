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
import multiprocessing
import os
import platform
import sys
import sysconfig
import warnings

import cloudpickle
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from pytensor.compile.ops import as_op
from pytensor.tensor.type import TensorType

import pymc as pm
import pymc.sampling.mcmc as mcmc
import pymc.sampling.parallel as ps

from pymc.pytensorf import floatX
from pymc.step_methods import CompoundStep


def test_context():
    with pm.Model():
        pm.Normal("x")
        ctx = multiprocessing.get_context("spawn")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            pm.sample(tune=2, draws=2, chains=2, cores=2, mp_ctx=ctx)


class TestMpCtxJaxSwitch:
    def test_switches_default_away_from_fork_under_jax(self):
        if ps._initialize_multiprocessing_context(None).get_start_method() != "fork":
            pytest.skip("platform default is not fork")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ctx = ps._initialize_multiprocessing_context(None, mode="JAX")
        assert ctx.get_start_method() != "fork"
        assert not any("JAX backend" in str(x.message) for x in w)

    def test_warns_when_user_explicitly_picks_fork_under_jax(self):
        if "fork" not in multiprocessing.get_all_start_methods():
            pytest.skip("fork start method not available on this platform")
        with pytest.warns(UserWarning, match="JAX backend"):
            ctx = ps._initialize_multiprocessing_context("fork", mode="JAX")
        assert ctx.get_start_method() == "fork"

    def test_leaves_non_jax_default_alone(self):
        expected = ps._initialize_multiprocessing_context(None).get_start_method()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ctx = ps._initialize_multiprocessing_context(None, mode="NUMBA")
        assert ctx.get_start_method() == expected
        assert not any("JAX backend" in str(x.message) for x in w)

    @pytest.mark.parametrize("explicit", ["spawn", "forkserver"])
    def test_no_warn_when_user_picks_safe_method(self, explicit):
        if explicit not in multiprocessing.get_all_start_methods():
            pytest.skip(f"{explicit} start method not available on this platform")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ctx = ps._initialize_multiprocessing_context(explicit, mode="JAX")
        assert ctx.get_start_method() == explicit
        assert not any("JAX backend" in str(x.message) for x in w)


class NoUnpickle:
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        raise AttributeError("This fails")


def test_bad_unpickle():
    with pm.Model() as model:
        pm.Normal("x")

    with model:
        step = pm.NUTS()
        step.no_unpickle = NoUnpickle()
        with pytest.raises(Exception) as exc_info:
            pm.sample(
                tune=2,
                draws=2,
                mp_ctx="spawn",
                step=step,
                cores=2,
                chains=2,
                compute_convergence_checks=False,
            )
        assert "could not be unpickled" in str(exc_info.getrepr(style="short"))


at_vector = TensorType(pytensor.config.floatX, [False])


@as_op([at_vector, pt.iscalar], [at_vector])
def _crash_remote_process(a, master_pid):
    if os.getpid() != master_pid:
        sys.exit(0)
    return 2 * np.array(a)


def test_remote_pipe_closed():
    master_pid = os.getpid()
    with pm.Model():
        x = pm.Normal("x", shape=2, mu=0.1)
        at_pid = pt.as_tensor_variable(np.array(master_pid, dtype="int32"))
        pm.Normal("y", mu=_crash_remote_process(x, at_pid), shape=2)

        step = pm.Metropolis()
        with pytest.raises(ps.ParallelSamplingError, match="Chain [0-9] failed with") as ex:
            pm.sample(step=step, mp_ctx="spawn", tune=2, draws=2, cores=2, chains=2)


@pytest.mark.skip(reason="Unclear")
@pytest.mark.parametrize("mp_start_method", ["spawn", "fork"])
def test_abort(mp_start_method):
    with pm.Model() as model:
        a = pm.Normal("a", shape=1)
        b = pm.HalfNormal("b")
        step1 = pm.NUTS([model.rvs_to_values[a]])
        step2 = pm.Metropolis([model.rvs_to_values[b]])

    step = CompoundStep([step1, step2])

    # on Windows we cannot fork
    if platform.system() == "Windows" and mp_start_method == "fork":
        return
    if mp_start_method == "spawn":
        step_method_pickled = cloudpickle.dumps(step, protocol=-1)
    else:
        step_method_pickled = None

    for abort in [False, True]:
        ctx = multiprocessing.get_context(mp_start_method)
        proc = ps.ProcessAdapter(
            10,
            10,
            step,
            chain=3,
            seed=1,
            mp_ctx=ctx,
            start={"a": floatX(np.array([1.0])), "b_log__": floatX(np.array(2.0))},
            step_method_pickled=step_method_pickled,
        )
        proc.start()
        while True:
            proc.write_next()
            out = ps.ProcessAdapter.recv_draw([proc])
            if out[1]:
                break
        if abort:
            proc.abort()
        proc.join()


@pytest.mark.parametrize("mp_start_method", ["spawn", "fork"])
def test_explicit_sample(mp_start_method):
    with pm.Model() as model:
        a = pm.Normal("a", shape=1)
        b = pm.HalfNormal("b")
        step1 = pm.NUTS([model.rvs_to_values[a]])
        step2 = pm.Metropolis([model.rvs_to_values[b]])

    step = CompoundStep([step1, step2])

    # on Windows we cannot fork
    if platform.system() == "Windows" and mp_start_method == "fork":
        return
    if mp_start_method == "spawn":
        step_method_pickled = cloudpickle.dumps(step, protocol=-1)
    else:
        step_method_pickled = None

    ctx = multiprocessing.get_context(mp_start_method)
    proc = ps.ProcessAdapter(
        10,
        10,
        step,
        chain=3,
        rng=np.random.default_rng(1),
        mp_ctx=ctx,
        start={"a": floatX(np.array([1.0])), "b_log__": floatX(np.array(2.0))},
        step_method_pickled=step_method_pickled,
        blas_cores=None,
    )
    proc.start()
    while True:
        proc.write_next()
        out = ps.ProcessAdapter.recv_draw([proc])
        view = proc.shared_point_view
        for name in view:
            view[name].copy()
        if out[1]:
            break
    proc.join()


def test_iterator():
    with pm.Model() as model:
        a = pm.Normal("a", shape=1)
        b = pm.HalfNormal("b")
        step1 = pm.NUTS([model.rvs_to_values[a]])
        step2 = pm.Metropolis([model.rvs_to_values[b]])

    step = CompoundStep([step1, step2])

    start = {"a": floatX(np.array([1.0])), "b_log__": floatX(np.array(2.0))}
    sampler = ps.ParallelSampler(
        draws=10,
        tune=10,
        chains=3,
        cores=2,
        rngs=np.random.default_rng(1).spawn(3),
        start_points=[start] * 3,
        step_method=step,
        progressbar=False,
        blas_cores=None,
    )
    with sampler:
        for draw in sampler:
            pass


def test_spawn_densitydist_function():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)

        def func(x):
            return -2 * (x**2).sum()

        obs = pm.CustomDist("density_dist", logp=func, observed=np.random.randn(100))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            pm.sample(draws=10, tune=10, step=pm.Metropolis(), cores=2, mp_ctx="spawn")


def test_spawn_densitydist_bound_method():
    N = 100
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)

        def logp(x, mu):
            normal_dist = pm.Normal.dist(mu, 1, size=N)
            out = pm.logp(normal_dist, x)
            return out

        obs = pm.CustomDist("density_dist", mu, logp=logp, observed=np.random.randn(N), size=N)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*number of samples.*", UserWarning)
            pm.sample(draws=10, tune=10, step=pm.Metropolis(), cores=2, mp_ctx="spawn")


@pytest.mark.parametrize("cores", (1, 2))
def test_sampling_with_random_generator_matches(cores):
    # Regression test for https://github.com/pymc-devs/pymc/issues/7612
    kwargs = {
        "chains": 2,
        "cores": cores,
        "tune": 10,
        "draws": 10,
        "compute_convergence_checks": False,
        "progress_bar": False,
    }
    with pm.Model() as m:
        x = pm.Normal("x")

        post1 = pm.sample(random_seed=np.random.default_rng(42), **kwargs).posterior
        post2 = pm.sample(random_seed=np.random.default_rng(42), **kwargs).posterior

    assert post1.equals(post2), (post1["x"].mean().item(), post2["x"].mean().item())


@pytest.mark.skipif(
    sysconfig.get_config_var("Py_GIL_DISABLED") != 1,
    reason="requires a free-threaded (no-GIL) build",
)
def test_thread_parallel_sampling_matches_sequential(monkeypatch):
    # On a free-threaded build `mp_ctx="thread"` runs chains as threads (the path this
    # test targets); `nuts_sampler="pymc"` keeps it on PyMC's stepper instead of
    # dispatching to nutpie. The conftest `pytest_sessionfinish` hook separately guards
    # that no import re-enabled the GIL during the run.
    #
    # Spy on `_thread_sample`, recording success only *after* it returns, so the threaded
    # path is proven to have run and not silently fallen back to multiprocessing (which
    # would produce the same draws and hide a regression).
    thread_sampled = False
    real_thread_sample = mcmc._thread_sample

    def spy(**kwargs):
        nonlocal thread_sampled
        result = real_thread_sample(**kwargs)
        thread_sampled = True
        return result

    monkeypatch.setattr(mcmc, "_thread_sample", spy)

    kwargs = {
        "draws": 50,
        "tune": 50,
        "chains": 2,
        "nuts_sampler": "pymc",
        "progressbar": False,
        "compute_convergence_checks": False,
        "random_seed": 1,
    }
    with pm.Model():
        pm.Normal("x", 0.0, 1.0)
        pm.Normal("y", pm.Normal("mu", 0.0, 1.0), 1.0, shape=3)
        threaded = pm.sample(cores=2, mp_ctx="thread", **kwargs)
        sequential = pm.sample(cores=1, **kwargs)

    assert thread_sampled, "sampling did not run through the threaded path"
    # Same seed => each chain's RNG is seeded once and identically regardless of
    # execution mode, so threaded sampling reproduces sequential draws bit-for-bit.
    assert threaded.posterior.equals(sequential.posterior)
