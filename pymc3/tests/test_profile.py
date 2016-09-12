import unittest
from .models import simple_model


class TestProfile(unittest.TestCase):
    def setUp(self):
        _, self.model, _ = simple_model()

    def test_profile_model(self):
        self.assertGreater(self.model.profile(self.model.logpt).fct_call_time, 0)

    def test_profile_variable(self):
        self.assertGreater(self.model.profile(self.model.vars[0].logpt).fct_call_time, 0)

    def test_profile_count(self):
        count = 1005
        self.assertEqual(self.model.profile(self.model.logpt, n=count).fct_callcount, count)
