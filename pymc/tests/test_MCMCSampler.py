"""
The DisasterMCMC example.

"""
from numpy.testing import *

try:
    from pymc.Matplot import plot
except:
    pass

PLOT=False


class test_MCMC(TestCase):
    def test(self):
        
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
        M.sample(10000,5000,verbose=0)
        
        if PLOT:
            # Plot samples
            plot(M)

if __name__ == '__main__':
    import unittest
    unittest.main()
    
