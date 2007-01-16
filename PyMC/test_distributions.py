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
import unittest 

from numpy.testing import *
import numpy as np
from numpy import exp

# Add plots

# Check Bernoulli
class test_bernoulli(NumpyTestCase):
    def check_consistency(self):
        N = 3000
        params = {'p':.6}
        samples = []
        for i in range(N):
            samples.append(rbernoulli(**params))
        H = np.bincount(samples)*1./N
        l0 = exp(bernoulli_like(0, **params))
        l1 = exp(bernoulli_like(1, **params))
        assert_array_almost_equal(H, [l0,l1], 2)

class test_beta(NumpyTestCase):
    def check_consistency(self):
        N = 5000
        nx = 15
        params ={'alpha':3, 'beta':5}
        samples = []
        for i in range(N):
            samples.append(rbeta(**params))
        X = np.linspace(0,1,nx, endpoint=False)+.5/nx
        like = []
        for x in X:
            like.append(exp(beta_like(x, **params)))
        H,bins = np.histogram(samples, range =[0,1], bins=nx, normed=True)
        assert_array_almost_equal(H, like,1)
    
if __name__ == '__main__':
    NumpyTest().run()
