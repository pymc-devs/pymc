from numpy.testing import *
from RJMCMC import *
import transd_model

from numpy.testing import *

def test_RJMCMC(NumpyTestCase):
    def check(self):
        RM = Sampler(transd_model)
        RM.sample(100,10,1)

if __name__=='__main__':
    NumpyTest().run()
