"""
The trans-dimensional example

"""
from numpy.testing import *
from pylab import *
PLOT=True

class test_Sampler(NumpyTestCase):
    def check(self):
        from pymc import Sampler
        from pymc.examples import trans_dimensional
        M = Sampler(trans_dimensional)
        M.sample(5000,0,10,verbose=False)
        M.plot()
            

if __name__ == '__main__':
    NumpyTest().run()
    
