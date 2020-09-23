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

import pytest
import pymc3.parallel_sampling as ps
import pymc3 as pm
import theano
import theano.tensor as tt
import numpy as np


def test_context():
    with pm.Model():
        pm.Normal('x')
        ctx = multiprocessing.get_context('spawn')
        pm.sample(tune=2, draws=2, chains=2, cores=2, mp_ctx=ctx)


class NoUnpickle:
    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        raise AttributeError("This fails")


def test_bad_unpickle():
    with pm.Model() as model:
        pm.Normal('x')

    with model:
        step = pm.NUTS()
        step.no_unpickle = NoUnpickle()
        with pytest.raises(Exception) as exc_info:
            pm.sample(tune=2, draws=2, mp_ctx='spawn', step=step,
                      cores=2, chains=2, compute_convergence_checks=False)
        assert 'could not be unpickled' in str(exc_info.getrepr(style='short'))


tt_vector = tt.TensorType(theano.config.floatX, [False])


@theano.as_op([tt_vector, tt.iscalar], [tt_vector])
def _crash_remote_process(a, master_pid):
    if os.getpid() != master_pid:
        os.exit(0)
    return 2 * np.array(a)


def test_dill():
    with pm.Model():
        pm.Normal('x')
        pm.sample(tune=1, draws=1, chains=2, cores=2, pickle_backend="dill", mp_ctx="spawn")


def test_remote_pipe_closed():
    master_pid = os.getpid()
    with pm.Model():
        x = pm.Normal('x', shape=2, mu=0.1)
        tt_pid = tt.as_tensor_variable(np.array(master_pid, dtype='int32'))
        pm.Normal('y', mu=_crash_remote_process(x, tt_pid), shape=2)

        step = pm.Metropolis()
        with pytest.raises(RuntimeError, match="Chain [0-9] failed"):
            pm.sample(step=step, mp_ctx='spawn', tune=2, draws=2, cores=2, chains=2)


def test_abort():
    with pm.Model() as model:
        a = pm.Normal('a', shape=1)
        pm.HalfNormal('b')
        step1 = pm.NUTS([a])
        step2 = pm.Metropolis([model.b_log__])

    step = pm.CompoundStep([step1, step2])

    ctx = multiprocessing.get_context()
    proc = ps.ProcessAdapter(10, 10, step, chain=3, seed=1, mp_ctx=ctx,
                             start={'a': 1., 'b_log__': 2.},
                             step_method_pickled=None, pickle_backend='pickle')
    proc.start()
    proc.write_next()
    proc.abort()
    proc.join()


def test_explicit_sample():
    with pm.Model() as model:
        a = pm.Normal('a', shape=1)
        pm.HalfNormal('b')
        step1 = pm.NUTS([a])
        step2 = pm.Metropolis([model.b_log__])

    step = pm.CompoundStep([step1, step2])

    ctx = multiprocessing.get_context()
    proc = ps.ProcessAdapter(10, 10, step, chain=3, seed=1, mp_ctx=ctx,
                             start={'a': 1., 'b_log__': 2.},
                             step_method_pickled=None, pickle_backend='pickle')
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
        a = pm.Normal('a', shape=1)
        pm.HalfNormal('b')
        step1 = pm.NUTS([a])
        step2 = pm.Metropolis([model.b_log__])

    step = pm.CompoundStep([step1, step2])

    start = {'a': 1., 'b_log__': 2.}
    sampler = ps.ParallelSampler(10, 10, 3, 2, [2, 3, 4], [start] * 3,
                                 step, 0, False)
    with sampler:
        for draw in sampler:
            pass
