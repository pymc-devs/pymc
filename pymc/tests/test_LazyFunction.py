from numpy.testing import *
import numpy as np
from numpy.random import random, normal
from pymc.LazyFunction import LazyFunction
from pymc import stochastic, data, deterministic, Normal, potential, ZeroProbability

verbose = False

@stochastic
def A(value=1.):
    return -10.*value

# If normcoef is positive, there will be an uncaught ZeroProbability
normcoef = 1.

# B's value is a random function of A.
@deterministic(verbose=verbose)
def B(A=A):
    # print 'B computing' 
    return A + normcoef * normal()

# Guarantee that initial state is OK
while B.value<0.:
    @deterministic(verbose=verbose)
    def B(A=A):
        # print 'B computing'
        return A + normcoef * normal()

@data
@stochastic(verbose=verbose)
def C(value = 0., B=[A,B]):
    if B[0]<0.:
         return -np.Inf
    else:
         return 0.
 
L = C._logp
C.logp
acc = True

for i in range(1000):

    # Record last values
    last_B_value = B.value
    last_A_count = A.counter.get_count()
    # print B._value.get_cached_counts(), B._value.get_frame_queue(), B._value.get_ultimate_arg_counter()[A]
    # print A.counter.count, B._value.get_cached_counts()
    last_C_logp = C.logp

    # Propose a value
    # print A.counter.get_count(),
    A.value = 1. + normal()
    # print A.counter.get_count(),
    B.value
    # print last_A_count, A.counter.get_count(), B._value.get_cached_counts()    

    # Check the argument values
    # L.refresh_argument_values()    
    # assert(L.argument_values['B'] is B.value)
    assert(C in L.ultimate_args)

    # Accept or reject values
    acc = True
    try:
        C.logp

        # Make sure A's value and last value occupy correct places in B's
        # cached arguments
        cur_frame = B._value.get_frame_queue()[1]
        assert(B._value.get_cached_counts()[0,cur_frame] == A.counter.get_count())
        assert(B._value.get_cached_counts()[0,1-cur_frame] == last_A_count)
        assert(B._value.ultimate_args[0] is A)
        # print

    except ZeroProbability:

        acc = False

        # Reject jump
        A.revert()
        # print 'reverting!', A.counter.get_count()

        # Make sure A's value and last value occupy correct places in B's
        # cached arguments        
        cur_frame = B._value.get_frame_queue()[1]
        assert(B._value.get_cached_counts()[0,1-cur_frame] == A.counter.get_count())
        # assert(B._value.cached_args[cur_frame] is A.last_value)
        # assert(B._value.ultimate_args.value[0] is A.value)
        assert(B.value is last_B_value)


    # Check C's cache
    cur_frame = L.get_frame_queue()[1]

    # If jump was accepted:
    if acc:
        # C's value should be at the head of C's cache
        assert(L.get_cached_counts()[1,cur_frame] == C.counter.get_count())
        assert(L.cached_values[cur_frame] is C.logp)

    # If jump was rejected:        
    else:

        # B's value should be at the back of C's cache.
        assert(L.get_cached_counts()[1,1-cur_frame] == C.counter.get_count())
        assert(L.cached_values[1-cur_frame] is C.logp)

    # assert(L.ultimate_args.value[1] is C.value)


# class test_LazyFunction(TestCase):
#     def test(self):
#         for i in range(1000):
# 
#             # Record last values
#             last_B_value = B.value
#             last_C_logp = C.logp
# 
#             # Propose a value
#             A.value = 1. + normal()
# 
#             # Check the argument values
#             # L.refresh_argument_values()    
#             # assert(L.argument_values['B'] is B.value)
#             assert(C in L.ultimate_args)
# 
#             # Accept or reject values
#             acc = True
#             try:
#                 C.logp
# 
#                 # Make sure A's value and last value occupy correct places in B's
#                 # cached arguments
#                 cur_frame = B._value.get_frame_queue()[1]
#                 assert(B._value.get_cached_counts()[0,cur_frame] == A.counter.count[0])
#                 assert(B._value.get_cached_counts()[0,1-cur_frame] == A.counter.count[0]-1)
#                 assert(B._value.ultimate_args[0] is A)
# 
#             except ZeroProbability:
# 
#                 acc = False
# 
#                 # Reject jump
#                 A.revert()
# 
#                 # Make sure A's value and last value occupy correct places in B's
#                 # cached arguments        
#                 cur_frame = B._value.get_frame_queue()[1]
#                 assert(B._value.get_cached_counts()[0,1-cur_frame] == A.counter.count[0])
#                 # assert(B._value.cached_args[cur_frame] is A.last_value)
#                 # assert(B._value.ultimate_args.value[0] is A.value)
#                 assert(B.value is last_B_value)
# 
# 
#             # Check C's cache
#             cur_frame = L.get_frame_queue()[1]
# 
#             # If jump was accepted:
#             if acc:
#                 # C's value should be at the head of C's cache
#                 assert(L.get_cached_counts()[1,cur_frame] == C.counter.count[0])
#                 assert(L.cached_values[cur_frame] is C.logp)
# 
#             # If jump was rejected:        
#             else:
# 
#                 # B's value should be at the back of C's cache.
#                 assert(L.get_cached_counts()[1,1-cur_frame] == C.counter.count[0])
#                 assert(L.cached_values[1-cur_frame] is C.logp)
# 
#             # assert(L.ultimate_args.value[1] is C.value)


if __name__ == '__main__':
    import unittest
    unittest.main()
