"""
The DisasterMCMC example.

"""
from numpy.testing import *
try:
    from pymc.Matplot import plot
    PLOT=True
except:
    PLOT=False


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
            plot(M)

if __name__ == '__main__':
    NumpyTest().run()
    
