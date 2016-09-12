import unittest
import numpy.random as nr


class SeededTest(unittest.TestCase):
    random_seed = 20160911

    @classmethod
    def setUpClass(cls):
        nr.seed(cls.random_seed)

    def setUp(self):
        nr.seed(self.random_seed)
