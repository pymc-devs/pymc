"""
The DisasterSampler example.

"""
from numpy.testing import *
from pylab import *
PLOT=True


class test_Sampler(NumpyTestCase):
    def check(self):
        
        # Import modules
        from PyMC2 import Sampler
        from PyMC2.examples import DisasterModel
        
        # Instantiate samplers
        M = Sampler(DisasterModel)
        
        # Check parameter arrays
        assert_equal(len(M.parameters), 3)
        assert_equal(len(M.data),1)
        assert_array_equal(M.D.value, DisasterModel.D_array)
        
        # Sample
        M.sample(50000,10000)
        
        # Plot samples
        M.plot()

if __name__ == '__main__':
    NumpyTest().run()
    
