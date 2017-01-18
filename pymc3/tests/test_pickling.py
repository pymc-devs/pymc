import unittest
import pickle
import traceback
from .models import simple_model


class TestPickling(unittest.TestCase):
    def setUp(self):
        _, self.model, _ = simple_model()

    def test_model_roundtrip(self):
        m = self.model
        for proto in range(pickle.HIGHEST_PROTOCOL+1):
            try:
                s = pickle.dumps(m, proto)
                n = pickle.loads(s)
            except Exception as ex:
                raise AssertionError(
                    "Exception while trying roundtrip with pickle protocol %d:\n"%proto +
                    ''.join(traceback.format_exc())
                )
