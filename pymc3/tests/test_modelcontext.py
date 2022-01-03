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

import threading

from pytest import raises

from pymc3 import Model, Normal
from pymc3.distributions.distribution import (
    _DrawValuesContext,
    _DrawValuesContextBlocker,
)
from pymc3.model import modelcontext


class TestModelContext:
    def test_thread_safety(self):
        """Regression test for issue #1552: Thread safety of model context manager

        This test creates two threads that attempt to construct two
        unrelated models at the same time.
        For repeatable testing, the two threads are syncronised such
        that thread A enters the context manager first, then B,
        then A attempts to declare a variable while B is still in the context manager.
        """
        aInCtxt, bInCtxt, aDone = (threading.Event() for _ in range(3))
        modelA = Model()
        modelB = Model()

        def make_model_a():
            with modelA:
                aInCtxt.set()
                bInCtxt.wait()
                Normal("a", 0, 1)
            aDone.set()

        def make_model_b():
            aInCtxt.wait()
            with modelB:
                bInCtxt.set()
                aDone.wait()
                Normal("b", 0, 1)

        threadA = threading.Thread(target=make_model_a)
        threadB = threading.Thread(target=make_model_b)
        threadA.start()
        threadB.start()
        threadA.join()
        threadB.join()
        # now let's see which model got which variable
        # previous to #1555, the variables would be swapped:
        # - B enters it's model context after A, but before a is declared -> a goes into B
        # - A leaves it's model context before B attempts to declare b. A's context manager
        #   takes B from the stack, such that b ends up in model A
        assert (
            list(modelA.named_vars),
            list(modelB.named_vars),
        ) == (["a"], ["b"])


def test_mixed_contexts():
    modelA = Model()
    modelB = Model()
    with raises((ValueError, TypeError)):
        modelcontext(None)
    with modelA:
        with modelB:
            assert Model.get_context() == modelB
            assert modelcontext(None) == modelB
            dvc = _DrawValuesContext()
            with dvc:
                assert Model.get_context() == modelB
                assert modelcontext(None) == modelB
                assert _DrawValuesContext.get_context() == dvc
                dvcb = _DrawValuesContextBlocker()
                with dvcb:
                    assert _DrawValuesContext.get_context() == dvcb
                    assert _DrawValuesContextBlocker.get_context() == dvcb
                assert _DrawValuesContext.get_context() == dvc
                assert _DrawValuesContextBlocker.get_context() is dvc
                assert Model.get_context() == modelB
                assert modelcontext(None) == modelB
            assert _DrawValuesContext.get_context(error_if_none=False) is None
            with raises(TypeError):
                _DrawValuesContext.get_context()
            assert Model.get_context() == modelB
            assert modelcontext(None) == modelB
        assert Model.get_context() == modelA
        assert modelcontext(None) == modelA
    assert Model.get_context(error_if_none=False) is None
    with raises(TypeError):
        Model.get_context(error_if_none=True)
    with raises((ValueError, TypeError)):
        modelcontext(None)
