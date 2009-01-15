import pymc
import numpy as np

from numpy.testing import *

### Bivariate normal ###
"""We first draw random samples from a bivariate normal distribution
with the following parameters:
 * sigma_1 = 1.
 * sigma_2 = sqrt(2)
 * rho = .8
 * mu = [-2.,  3.].

and C = [ \sigma_1\sigma_1       \rho\sigma_1\sigma_2 ]
        [ \rho\sigma_2\sigma_1       \sigma_2\sigma_2 ]

Then, knowing the covariance matrix C and given the random samples,
we want to estimate the posterior distribution of mu. Since the prior
for mu is uniform, the mean posterior distribution is simply a bivariate
normal with the same correlation coefficient rho, but with variances
divided by sqrt(N), where N is the number of samples drawn.

We can check that the sampler works correctly by making sure that
after a while, the covariance matrix of the samples for mu tend to C/N.

"""
N=50
mu = np.array([-2.,  3.])
C = np.array([[1,  .8*np.sqrt(2)], [.8*np.sqrt(2), 2.]])
r = pymc.rmv_normal_cov(mu, C, size=50)

@pymc.stoch
def mean(value=np.array([0., 0.])):
    """The mean of the samples (mu). """
    return 0.

obs = pymc.MvNormalCov('obs', mean, C, value=r, observed=True)

# def test_AM():
S = pymc.MCMC([mean, obs])
S.use_step_method(pymc.AdaptiveMetropolis,  mean, delay=200)
S.sample(1000)
S.sample(5000)
Cs = np.cov(S.trace('mean')[:].T)
assert_array_almost_equal(Cs,  C/N, 2)


# if __name__ == '__main__':
#     import nose
#     nose.runmodule()

