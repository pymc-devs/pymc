"""Test the distributions and their random generators to check 
they are consistent with one another.
For each distribution:
1. Select parameters (params = {'alpha':.., 'beta'=...}).
2. Generate N random samples.
3. Compute likelihood for a vector of x values.
4. Compute histogram of samples.
5. Compare histogram with likelihood.
6. Assert if they fit or not.

"""
from decorators import *
from numpy.testing import *
import numpy as np
from numpy import exp


# Check Bernoulli
class test_bernoulli(NumpyTestCase):
    def check_consistency(self):
        N = 500
        params = {'p':.6}
        samples = []
        for i in range(N):
            samples.append(rbernoulli(**params))
        H = np.bincount(samples)*1./N
        l0 = exp(like(0, **params[dist]))
        l1 = exp(like(1, **params[dist]))
        assert_array_almost_equal(H, [l0,l1], 2)

    
if __name__ == "__main__":
    NumpyTest('decorators').run()
