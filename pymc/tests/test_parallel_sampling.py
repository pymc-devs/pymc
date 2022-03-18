#   Copyright 2020 The PyMC Developers
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

import aesara
import aesara.tensor as at
import cloudpickle
import numpy as np
import pytest

from aesara.compile.ops import as_op
from aesara.tensor.type import TensorType

import pymc as pm
import pymc.parallel_sampling as ps

from pymc.aesaraf import floatX


def test_context():
    with pm.Model():
        pm.Normal("x")
        ctx = multiprocessing.get_context("spawn")
        pm.sample(tune=2, draws=2, chains=2, cores=2, mp_ctx=ctx)


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


at_vector = TensorType(aesara.config.floatX, [False])


@as_op([at_vector, at.iscalar], [at_vector])
def _crash_remote_process(a, master_pid):
    if os.getpid() != master_pid:
        os.exit(0)
    return 2 * np.array(a)


def test_remote_pipe_closed():
    master_pid = os.getpid()
    with pm.Model():
        x = pm.Normal("x", shape=2, mu=0.1)
        at_pid = at.as_tensor_variable(np.array(master_pid, dtype="int32"))
        pm.Normal("y", mu=_crash_remote_process(x, at_pid), shape=2)

        step = pm.Metropolis()
        with pytest.raises(RuntimeError, match="Chain [0-9] failed"):
            pm.sample(step=step, mp_ctx="spawn", tune=2, draws=2, cores=2, chains=2)


@pytest.mark.skip(reason="Unclear")
@pytest.mark.parametrize("mp_start_method", ["spawn", "fork"])
def test_abort(mp_start_method):
    with pm.Model() as model:
        a = pm.Normal("a", shape=1)
        b = pm.HalfNormal("b")
        step1 = pm.NUTS([model.rvs_to_values[a]])
        step2 = pm.Metropolis([model.rvs_to_values[b]])

    step = pm.CompoundStep([step1, step2])

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

    step = pm.CompoundStep([step1, step2])

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
        seed=1,
        mp_ctx=ctx,
        start={"a": floatX(np.array([1.0])), "b_log__": floatX(np.array(2.0))},
        step_method_pickled=step_method_pickled,
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

    step = pm.CompoundStep([step1, step2])

    start = {"a": floatX(np.array([1.0])), "b_log__": floatX(np.array(2.0))}
    sampler = ps.ParallelSampler(10, 10, 3, 2, [2, 3, 4], [start] * 3, step, 0, False)
    with sampler:
        for draw in sampler:
            pass


def test_spawn_densitydist_function():
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)

        def func(x):
            return -2 * (x**2).sum()

        obs = pm.DensityDist("density_dist", logp=func, observed=np.random.randn(100))
        pm.sample(draws=10, tune=10, step=pm.Metropolis(), cores=2, mp_ctx="spawn")


def test_spawn_densitydist_bound_method():
    N = 100
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)

        def logp(x, mu):
            normal_dist = pm.Normal.dist(mu, 1, size=N)
            out = pm.logp(normal_dist, x)
            return out

        obs = pm.DensityDist("density_dist", mu, logp=logp, observed=np.random.randn(N), size=N)
        pm.sample(draws=10, tune=10, step=pm.Metropolis(), cores=2, mp_ctx="spawn")
