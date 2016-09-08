import unittest
import numpy.random as nr


class SeededTest(unittest.TestCase):
    random_seed = 20160907

    def setUp(self):
        nr.seed(self.random_seed)
