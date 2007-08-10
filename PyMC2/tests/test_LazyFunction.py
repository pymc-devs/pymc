from numpy.testing import *
import numpy as np
from numpy.random import random, normal
from PyMC2.PyrexLazyFunction import LazyFunction
from PyMC2 import parameter, data, node, Normal, potential, ZeroProbability

verbose = False

@parameter
def A(value=1.):
   return -10.*value

# If normcoef is positive, there will be an uncaught ZeroProbability
normcoef = 1.

# B's value is a random function of A.
@node(verbose=verbose)
def B(A=A):
   return A + normcoef * normal()

# Guarantee that initial state is OK
while B.value<0.:
   @node(verbose=verbose)
   def B(A=A):
       return A + normcoef * normal()

@data
@parameter(verbose=verbose)
def C(value = 0., B=B):
   if B<0.:
       return -np.Inf
   else:
       return 0.
       
L = C._logp
C.logp
acc = True


for i in range(1000):

    # Record last values
    last_B_value = B.value
    last_C_logp = C.logp
    
    # Propose a value
    A.value = 1. + normal()
    
    # Check the argument values
    L.refresh_argument_values()    
    assert_equal(L.argument_values['B'], B.value)
    assert_equal(L.ultimate_arg_values[0], B.value)
    
    # Accept or reject value
    acc = True
    try:
        C.logp
        print 'Accepting'
    except ZeroProbability:
        print 'Rejecting'
        acc = False
        A.value = A.last_value
        print [A.value, A.last_value], B._value.cached_args
        print [B.value, last_B_value], B._value.cached_values 
        assert_equal(B.value, last_B_value)
        
    # print L.get()
    
    # If jump was accepted:
    if acc:
        # Check the head of the cache
        assert_equal(L.cached_args[0], B.value)
        assert_equal(L.cached_values[0], C.logp)
        
    # If jump was rejected:        
    else:
        
        # Should currently be using the second entry in the cache
        assert_equal(L.cached_args[2], B.value)
        assert_equal(L.cached_values[1], C.logp)
        

