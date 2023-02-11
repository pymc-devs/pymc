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

import pytensor
import pytest

import pymc as pm

from pymc.step_methods import (
    NUTS,
    CompoundStep,
    DEMetropolis,
    HamiltonianMC,
    Metropolis,
    Slice,
)
from pymc.step_methods.compound import (
    StatsBijection,
    flatten_steps,
    get_stats_dtypes_shapes_from_steps,
    infer_warn_stats_info,
)
from pymc.tests.helpers import StepMethodTester, fast_unstable_sampling_mode
from pymc.tests.models import simple_2model_continuous


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
            cs = pm.CompoundStep([s1, s2])
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
            {"a": float, "c": int},
        ]
        bij = StatsBijection(step_stats_dtypes)
        assert bij.n_samplers == 2
        stats_l = [
            dict(a=1.5, b=3),
            dict(a=2.5, c=4),
        ]
        stats_d = bij.map(stats_l)
        assert isinstance(stats_d, dict)
        assert stats_d["sampler_0__a"] == 1.5
        assert stats_d["sampler_0__b"] == 3
        assert stats_d["sampler_1__a"] == 2.5
        assert stats_d["sampler_1__c"] == 4
        rev = bij.rmap(stats_d)
        assert isinstance(rev, list)
        assert len(rev) == len(stats_l)
        assert rev == stats_l
        # Also rmap incomplete dicts
        rev2 = bij.rmap({"sampler_1__a": 0})
        assert len(rev2) == 2
        assert len(rev2[0]) == 0
        assert len(rev2[1]) == 1
