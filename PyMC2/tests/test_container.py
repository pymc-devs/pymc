from numpy.testing import *
from PyMC2.examples import DisasterModel as DM
from PyMC2 import Container

class test_Container(NumpyTestCase):
    def check(self):
        A = [[DM.e, DM.s], [DM.l, DM.D, 3.], 54.323]
        C = Container(A)
        print C
