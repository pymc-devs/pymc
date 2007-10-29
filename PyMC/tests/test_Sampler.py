"""
The DisasterSampler example.

"""
from numpy.testing import *
from pylab import *
PLOT=True


class test_Sampler(NumpyTestCase):
    def check(self):
        
        # Import modules
        from PyMC import Sampler
        from PyMC.examples import DisasterModel
        
        # Instantiate samplers
        M = Sampler(DisasterModel)
        
        # Check stoch arrays
        assert_equal(len(M.stochs), 3)
        assert_equal(len(M.data),1)
        assert_array_equal(M.D.value, DisasterModel.D_array)
        
        # Sample
        M.sample(100000,50000,verbose=2)
        
        # Plot samples
        M.plot()

if __name__ == '__main__':
    NumpyTest().run()
    
