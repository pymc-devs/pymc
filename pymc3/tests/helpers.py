from logging.handlers import BufferingHandler
import contextlib
import numpy.random as nr
from theano.sandbox.rng_mrg import MRG_RandomStreams
from ..theanof import set_tt_rng, tt_rng
import theano


class SeededTest:
    random_seed = 20160911

    @classmethod
    def setup_class(cls):
        nr.seed(cls.random_seed)

    def setup_method(self):
        nr.seed(self.random_seed)
        self.old_tt_rng = tt_rng()
        set_tt_rng(MRG_RandomStreams(self.random_seed))

    def teardown_method(self):
        set_tt_rng(self.old_tt_rng)


class LoggingHandler(BufferingHandler):
    def __init__(self, matcher):
        # BufferingHandler takes a "capacity" argument
        # so as to know when to flush. As we're overriding
        # shouldFlush anyway, we can set a capacity of zero.
        # You can call flush() manually to clear out the
        # buffer.
        BufferingHandler.__init__(self, 0)
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

    _partial_matches = ('msg', 'message')

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
        elif not isinstance(df, str) or k not in self._partial_matches:
            result = (v == dv)
        else:
            result = dv.find(v) >= 0
        return result


def select_by_precision(float64, float32):
    """Helper function to choose reasonable decimal cutoffs for different floatX modes."""
    decimal = float64 if theano.config.floatX == "float64" else float32
    return decimal


@contextlib.contextmanager
def not_raises():
    yield
