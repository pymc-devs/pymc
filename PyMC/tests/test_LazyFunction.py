from numpy.testing import *
import numpy as np
from numpy.random import random, normal
from PyMC2.LazyFunction import LazyFunction
from PyMC2 import stoch, data, dtrm, Normal, potential, ZeroProbability

verbose = False

@stoch
def A(value=1.):
    return -10.*value

# If normcoef is positive, there will be an uncaught ZeroProbability
normcoef = 1.

# B's value is a random function of A.
@dtrm(verbose=verbose)
def B(A=A):
    # print 'B computing' 
    return A + normcoef * normal()

# Guarantee that initial state is OK
while B.value<0.:
    @dtrm(verbose=verbose)
    def B(A=A):
        # print 'B computing'
        return A + normcoef * normal()

@data
@stoch(verbose=verbose)
def C(value = 0., B=[A,B]):
    if B[0]<0.:
         return -np.Inf
    else:
         return 0.
 
L = C._logp
C.logp
acc = True

class test_LazyFunction(NumpyTestCase):
    def check(self):
        for i in range(1000):

            # Record last values
            last_B_value = B.value
            last_C_logp = C.logp
    
            # Propose a value
            A.value = 1. + normal()
    
            # Check the argument values
            # L.refresh_argument_values()    
            # assert(L.argument_values['B'] is B.value)
            assert(L.ultimate_args.value[1] is B.value)
    
            # Accept or reject values
            acc = True
            try:
                C.logp
        
                # Make sure A's value and last value occupy correct places in B's
                # cached arguments
                cur_frame = B._value.frame_queue[1]
                assert(B._value.cached_args[cur_frame] is A.value)
                assert(B._value.cached_args[1-cur_frame] is A.last_value)
                assert(B._value.ultimate_args.value[0] is A.value)

            except ZeroProbability:
    
                acc = False
        
                # Reject jump
                A.value = A.last_value
        
                # Make sure A's value and last value occupy correct places in B's
                # cached arguments        
                cur_frame = B._value.frame_queue[1]
                assert(B._value.cached_args[1-cur_frame] is A.value)
                assert(B._value.cached_args[cur_frame] is A.last_value)
                assert(B._value.ultimate_args.value[0] is A.value)
                assert(B.value is last_B_value)
    
    
            # Check C's cache
            cur_frame = L.frame_queue[1]
    
            # If jump was accepted:
            if acc:
                # B's value should be at the head of C's cache
                assert(L.cached_args[cur_frame*3+1] is B.value)
                assert(L.cached_values[cur_frame] is C.logp)
        
            # If jump was rejected:        
            else:
        
                # B's value should be at the back of C's cache.
                assert(L.cached_args[(1-cur_frame)*3+1] is B.value)
                assert(L.cached_values[1-cur_frame] is C.logp)
        
            assert(L.ultimate_args.value[1] is B.value)
        

if __name__ == '__main__':
    NumpyTest().run()