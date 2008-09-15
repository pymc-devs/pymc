"""
Test interactive sampler
"""

# TODO: Make real test case.

from pymc import MCMC
from pymc.examples import DisasterModel
import nose

def test_interactive():
    # The interactive sampler exits only after the user presses a key. 
    # In other words, the test hangs. 
    raise nose.SkipTest
    S = MCMC(DisasterModel)
    S.isample(200,100,2)
    
    
if __name__ == '__main__':
    nose.runmodule()
