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

import contextlib

from logging.handlers import BufferingHandler

import aesara
import numpy as np
import numpy.random as nr

from aesara.gradient import verify_grad as at_verify_grad
from aesara.graph.opt import in2out
from aesara.sandbox.rng_mrg import MRG_RandomStream as RandomStream

from pymc.aesaraf import at_rng, local_check_parameter_to_ninf_switch, set_at_rng


class SeededTest:
    random_seed = 20160911
    random_state = None

    @classmethod
    def setup_class(cls):
        nr.seed(cls.random_seed)

    def setup_method(self):
        nr.seed(self.random_seed)
        self.old_at_rng = at_rng()
        set_at_rng(RandomStream(self.random_seed))

    def teardown_method(self):
        set_at_rng(self.old_at_rng)

    def get_random_state(self, reset=False):
        if self.random_state is None or reset:
            self.random_state = nr.RandomState(self.random_seed)
        return self.random_state


class LoggingHandler(BufferingHandler):
    def __init__(self, matcher):
        # BufferingHandler takes a "capacity" argument
        # so as to know when to flush. As we're overriding
        # shouldFlush anyway, we can set a capacity of zero.
        # You can call flush() manually to clear out the
        # buffer.
        super().__init__(0)
        self.matcher = matcher

    def shouldFlush(self):
        return False

    def emit(self, record):
        self.buffer.append(record.__dict__)

    def matches(self, **kwargs):
        """
        Look for a saved dict whose keys/values match the supplied arguments.
        """
        for d in self.buffer:
            if self.matcher.matches(d, **kwargs):
                result = True
                break
        return result


class Matcher:

    _partial_matches = ("msg", "message")

    def matches(self, d, **kwargs):
        """
        Try to match a single dict with the supplied arguments.

        Keys whose values are strings and which are in self._partial_matches
        will be checked for partial (i.e. substring) matches. You can extend
        this scheme to (for example) do regular expression matching, etc.
        """
        result = True
        for k in kwargs:
            v = kwargs[k]
            dv = d.get(k)
            if not self.match_value(k, dv, v):
                result = False
                break
        return result

    def match_value(self, k, dv, v):
        """
        Try to match a single stored value (dv) with a supplied value (v).
        """
        if isinstance(v, type(dv)):
            result = False
        elif not isinstance(dv, str) or k not in self._partial_matches:
            result = v == dv
        else:
            result = dv.find(v) >= 0
        return result


def select_by_precision(float64, float32):
    """Helper function to choose reasonable decimal cutoffs for different floatX modes."""
    decimal = float64 if aesara.config.floatX == "float64" else float32
    return decimal


@contextlib.contextmanager
def not_raises():
    yield


def verify_grad(op, pt, n_tests=2, rng=None, *args, **kwargs):
    if rng is None:
        rng = nr.RandomState(411342)
    at_verify_grad(op, pt, n_tests, rng, *args, **kwargs)


def assert_random_state_equal(state1, state2):
    for field1, field2 in zip(state1, state2):
        if isinstance(field1, np.ndarray):
            np.testing.assert_array_equal(field1, field2)
        else:
            assert field1 == field2


# This mode can be used for tests where model compilations takes the bulk of the runtime
# AND where we don't care about posterior numerical or sampling stability (e.g., when
# all that matters are the shape of the draws or deterministic values of observed data).
# DO NOT USE UNLESS YOU HAVE A GOOD REASON TO!
fast_unstable_sampling_mode = (
    aesara.compile.mode.FAST_COMPILE
    # Remove slow rewrite phases
    .excluding("canonicalize", "specialize")
    # Include necessary rewrites for proper logp handling
    .including("remove_TransformedVariables").register(
        (in2out(local_check_parameter_to_ninf_switch), -1)
    )
)
