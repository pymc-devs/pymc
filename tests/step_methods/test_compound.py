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

import numpy as np
import pytensor
import pytest

import pymc as pm

from pymc.model import modelcontext
from pymc.step_methods import (
    NUTS,
    CompoundStep,
    DEMetropolis,
    HamiltonianMC,
    Metropolis,
    Slice,
)
from pymc.step_methods.compound import (
    BlockedStep,
    StatsBijection,
    flatten_steps,
    get_stats_dtypes_shapes_from_steps,
    infer_warn_stats_info,
)
from pymc.testing import fast_unstable_sampling_mode
from pymc.util import get_random_generator, get_value_vars_from_user_vars
from tests.helpers import StepMethodTester
from tests.models import simple_2model_continuous


class NonForkableStep(BlockedStep):
    """A minimal custom step method that samples but never implements ``fork``.

    Stands in for a third-party / user-defined sampler: it can drive ``pm.sample``
    (a plain random-walk Metropolis), but because it does not override ``fork`` it
    hits the ``BlockedStep.fork`` default that raises ``NotImplementedError`` --
    which is how threaded sampling detects it must fall back to multiprocessing.
    """

    name = "nonforkable"
    stats_dtypes_shapes: dict = {}

    def __init__(self, vars, model=None, rng=None):
        model = modelcontext(model)
        self.vars = get_value_vars_from_user_vars(vars, model)
        self.var_names = [v.name for v in self.vars]
        self._logp = model.compile_logp()
        self.rng = get_random_generator(rng)
        self.tune = False

    def set_rng(self, rng):
        self.rng = get_random_generator(rng, copy=False)

    def step(self, point):
        proposal = dict(point)
        for name in self.var_names:
            proposal[name] = point[name] + self.rng.normal(scale=0.3, size=np.shape(point[name]))
        if np.log(self.rng.uniform()) < float(self._logp(proposal)) - float(self._logp(point)):
            return proposal, []
        return dict(point), []


def test_fork_fallback_for_non_forkable_step(monkeypatch, caplog):
    """A step method that never implements ``fork`` must fail loudly so threaded
    sampling falls back to multiprocessing instead of running an unsafe shared
    step across threads.

    Covers: the base ``fork`` raising ``NotImplementedError``, ``CompoundStep``
    propagating it (rather than forking only some children), and ``pm.sample``
    completing via the process path when the threaded branch is forced on.
    """
    import logging

    # The step itself, and any CompoundStep containing it, refuse to fork.
    with pm.Model():
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", 0, 1)
        with pytest.raises(NotImplementedError, match="NonForkableStep"):
            NonForkableStep([x]).fork(np.random.default_rng(0))
        with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
            compound = CompoundStep([NUTS([y]), NonForkableStep([x])])
        with pytest.raises(NotImplementedError, match="NonForkableStep"):
            compound.fork(np.random.default_rng(0))

    # Force the free-threaded dispatch branch regardless of the actual build;
    # sampling must still complete by falling back to multiprocessing.
    # `sys._is_gil_enabled` only exists on Python 3.13+; create it if absent so
    # the free-threaded branch is exercised on GIL builds too.
    monkeypatch.setattr("pymc.sampling.mcmc.sys._is_gil_enabled", lambda: False, raising=False)
    with pm.Model():
        x = pm.Normal("x", 2.0, 1.0)
        step = NonForkableStep([x])
        with caplog.at_level(logging.WARNING, logger="pymc"):
            idata = pm.sample(
                step=step,
                draws=20,
                tune=20,
                chains=2,
                cores=2,
                progressbar=False,
                compute_convergence_checks=False,
                random_seed=1,
            )
    assert "does not support threaded sampling" in caplog.text
    assert idata.posterior["x"].shape == (2, 20)


def test_stepmethods_do_not_require_tune_stat():
    step_types = pm.step_methods.STEP_METHODS
    assert len(step_types) > 5
    for cls in step_types:
        assert "tune" not in cls.stats_dtypes_shapes


class TestCompoundStep:
    samplers = (Metropolis, Slice, HamiltonianMC, NUTS, DEMetropolis)

    def test_non_blocked(self):
        """Test that samplers correctly create non-blocked compound steps."""
        with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
            _, model = simple_2model_continuous()
            with model:
                for sampler in self.samplers:
                    assert isinstance(sampler(blocked=False), CompoundStep)

    def test_blocked(self):
        with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
            _, model = simple_2model_continuous()
            with model:
                for sampler in self.samplers:
                    sampler_instance = sampler(blocked=True)
                    assert not isinstance(sampler_instance, CompoundStep)
                    assert isinstance(sampler_instance, sampler)

    def test_name(self):
        with pm.Model() as m:
            c1 = pm.HalfNormal("c1")
            c2 = pm.HalfNormal("c2")

            step1 = NUTS([c1])
            step2 = Slice([c2])
            step = CompoundStep([step1, step2])
        assert step.name == "Compound[nuts, slice]"


class TestStepCompound(StepMethodTester):
    @pytest.mark.parametrize(
        "step_fn, draws",
        [
            (
                lambda C, _: CompoundStep(
                    [
                        HamiltonianMC(scaling=C, is_cov=True),
                        HamiltonianMC(scaling=C, is_cov=True, blocked=False),
                    ]
                ),
                1000,
            ),
        ],
        ids=str,
    )
    def test_step_continuous(self, step_fn, draws):
        self.step_continuous(step_fn, draws)


class TestRVsAssignmentCompound:
    def test_compound_step(self):
        with pm.Model() as m:
            c1 = pm.HalfNormal("c1")
            c2 = pm.HalfNormal("c2")

            with pytensor.config.change_flags(mode=fast_unstable_sampling_mode):
                step1 = NUTS([c1])
                step2 = NUTS([c2])
                step = CompoundStep([step1, step2])
            assert {m.rvs_to_values[c1], m.rvs_to_values[c2]} == set(step.vars)


class TestStatsMetadata:
    def test_infer_warn_stats_info(self):
        """
        Until `BlockedStep.stats_dtypes` is removed, the new `stats_dtypes_shapes`
        attributed is inferred from `stats_dtypes`, or vice versa.
        """
        # Infer new
        with pytest.warns(DeprecationWarning, match="to specify"):
            old, new = infer_warn_stats_info([{"a": int, "b": object}], {}, "bla")
        assert isinstance(old, list)
        assert len(old) == 1
        assert old[0] == {"a": int, "b": object}
        assert isinstance(new, dict)
        assert new["a"] == (int, None)
        assert new["b"] == (object, None)

        # Infer old
        old, new = infer_warn_stats_info([], {"a": (int, []), "b": (float, [2])}, "bla")
        assert isinstance(old, list)
        assert len(old) == 1
        assert old[0] == {"a": int, "b": float}
        assert isinstance(new, dict)
        assert new["a"] == (int, [])
        assert new["b"] == (float, [2])

        # Disallow specifying both (single source of truth problem)
        with pytest.raises(TypeError, match="Only one of"):
            infer_warn_stats_info([{"a": float}], {"b": (int, [])}, "bla")

    def test_stats_from_steps(self):
        with pm.Model():
            s1 = pm.NUTS(pm.Normal("n"))
            s2 = pm.Metropolis(pm.Bernoulli("b", 0.5))
            cs = CompoundStep([s1, s2])
        # Make sure that sampler initialization does not modify the
        # class-level default values of the attributes.
        assert pm.NUTS.stats_dtypes == []
        assert pm.Metropolis.stats_dtypes == []

        sds = get_stats_dtypes_shapes_from_steps([s1, s2])
        assert "sampler_0__step_size" in sds
        assert "sampler_1__accepted" in sds
        assert len(cs.stats_dtypes) == 2
        assert cs.stats_dtypes_shapes == sds


class TestStatsBijection:
    def test_flatten_steps(self):
        with pm.Model():
            a = pm.Normal("a")
            b = pm.Normal("b")
            c = pm.Normal("c")
            s1 = Metropolis([a])
            s2 = Metropolis([b])
            c1 = CompoundStep([s1, s2])
            s3 = NUTS([c])
            c2 = CompoundStep([c1, s3])
        assert flatten_steps(s1) == [s1]
        assert flatten_steps(c2) == [s1, s2, s3]
        with pytest.raises(ValueError, match="Unexpected type"):
            flatten_steps("not a step")

    def test_stats_bijection(self):
        step_stats_dtypes = [
            {"a": float, "b": int},
            {"a": float, "c": Warning},
        ]
        bij = StatsBijection(step_stats_dtypes)
        assert bij.object_stats == {"sampler_1__c": (1, "c")}
        assert bij.n_samplers == 2
        w = Warning("hmm")
        stats_l = [
            {"a": 1.5, "b": 3},
            {"a": 2.5, "c": w},
        ]
        stats_d = bij.map(stats_l)
        assert isinstance(stats_d, dict)
        assert stats_d["sampler_0__a"] == 1.5
        assert stats_d["sampler_0__b"] == 3
        assert stats_d["sampler_1__a"] == 2.5
        assert stats_d["sampler_1__c"] == w
        rev = bij.rmap(stats_d)
        assert isinstance(rev, list)
        assert len(rev) == len(stats_l)
        assert rev == stats_l
        # Also rmap incomplete dicts
        rev2 = bij.rmap({"sampler_1__a": 0})
        assert len(rev2) == 2
        assert len(rev2[0]) == 0
        assert len(rev2[1]) == 1
