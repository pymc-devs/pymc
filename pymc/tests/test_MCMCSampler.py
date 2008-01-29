"""
The DisasterMCMC example.

"""
from numpy.testing import *
from pylab import *
PLOT=True


class test_MCMC(NumpyTestCase):
    def check(self):
        
        # Import modules
        from pymc import MCMC
        from pymc.examples import DisasterModel
        
        # Instantiate samplers
        M = MCMC(DisasterModel)
        
        # Check stochastic arrays
        assert_equal(len(M.stochastics), 3)
        assert_equal(len(M.data_stochastics),1)
        assert_array_equal(M.D.value, DisasterModel.D_array)
        
        # Sample
        M.sample(20000,10000,verbose=2)
        
        if PLOT:
            # Plot samples
            M.plot()

if __name__ == '__main__':
    NumpyTest().run()
    
