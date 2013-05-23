from pymc import rnormal, Normal, MCMC, Uniform, Matplot, Metropolis
from numpy import ma, shape, ravel
from numpy.testing import *

# Generate some data with missing observations
fake_data = rnormal(0, 1, size=100)
m = ma.masked_array(fake_data)
m[[3, 6]] = ma.masked
m.fill_value = -4


class TestMissing(TestCase):

    """Unit test for missing data imputation"""

    def test_simple(self):

        # Priors
        mu = Normal('mu', mu=0, tau=0.0001)
        s = Uniform('s', lower=0, upper=100, value=10)
        tau = s ** -2

        # Likelihood with missing data
        x = Normal('x', mu=mu, tau=tau, value=m, observed=True)

        # Instantiate sampler
        M = MCMC([mu, s, tau, x])

        # Run sampler
        M.sample(10000, 5000, progress_bar=0)

        # Check length of value
        assert_equal(len(x.value), 100)
        # Check size of trace
        tr = M.trace('x')()
        assert_equal(shape(tr), (5000, 2))

        sd2 = [-2 < i < 2 for i in ravel(tr)]

        # Check for standard normal output
        assert_almost_equal(sum(sd2) / 10000., 0.95, decimal=1)
