from numpy.testing import *
from pymc.gp import *
from .test_mean import M, x
from .test_cov import C
from numpy import *
from copy import copy

# Impose observations on the GP


class test_realization(TestCase):

    def test(self):
        for i in range(3):
            f = Realization(M, C)
            f(x)
