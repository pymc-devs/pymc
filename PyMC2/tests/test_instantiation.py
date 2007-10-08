###
# Test that decorators return the appropriate object.
# David Huard
# June 21, 2007
###

###
# TODO
# ----
# Add test for dtrm
# Check discrete and binary stochs
# Test the distribution instantiators.
###

from numpy.testing import *
import PyMC2
from PyMC2 import Sampler, data, stoch, dtrm, discrete_stoch, \
    Stochastic,Deterministic
from numpy import array, log, sum, ones, concatenate, inf
from PyMC2 import uniform_like, exponential_like, poisson_like


D_array =   array([ 4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                    3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                    2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0,
                    1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                    0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                    3, 3, 1, 1, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

# Define data and stochs

@discrete_stoch
def s(value=50, length=110):
    """Change time for rate stoch."""
    return uniform_like(value, 0, length)

@stoch
def e(value=1., rate=1.):
    """Rate stoch of poisson distribution."""
    return exponential_like(value, rate)

@stoch
def l(value=.1, rate = 1.):
    """Rate stoch of poisson distribution."""
    return exponential_like(value, rate)
        
@data(discrete=True)
def D(  value = D_array,
        s = s,
        e = e,
        l = l):
    """Annual occurences of coal mining disasters."""
    return poisson_like(value[:s],e) + poisson_like(value[s:],l)

E = data(e)

@data
def F(value = D_array*.5,
        s = s,
        e = e,
        l = l):
    """Annual occurences of coal mining disasters."""
    return poisson_like(value[:s],e) + poisson_like(value[s:],l)
        
@data
@stoch
def G(value = D_array*.5,
        s = s,
        e = e,
        l = l):
    """Annual occurences of coal mining disasters."""
    return poisson_like(value[:s],e) + poisson_like(value[s:],l)
        
class test_instantiation(NumpyTestCase):
    def check_data(self):
        assert(isinstance(D, Stochastic))
        assert(D.isdata)
        assert(isinstance(E, Stochastic))
        assert(E.isdata)
        assert(isinstance(F, Stochastic))
        assert(F.isdata)
        assert(isinstance(G, Stochastic))
        assert(G.isdata)
    def check_stoch(self):
        assert(isinstance(l, Stochastic))
        assert(not l.isdata)
if __name__ == '__main__':
    NumpyTest().run()
