from numpy.testing import *
import numpy as np
from numpy.random import normal
from pymc import stochastic, observed, deterministic, ZeroProbability

verbose = False

np.random.seed(1)


class test_LazyFunction(TestCase):

    def test_cached(self):

        @stochastic
        def A(value=1.):
            return -10. * value

        # If normcoef is positive, there will be an uncaught ZeroProbability
        normcoef = -1.

        # B's value is a random function of A.
        @deterministic(verbose=verbose)
        def B(A=A):
            return A + normcoef * normal()

        # Guarantee that initial state is OK
        while B.value < 0.:
            @deterministic(verbose=verbose)
            def B(A=A):
                return A + normcoef * normal()

        @observed(verbose=verbose)
        def C(value=0., B=(A, B)):
            if B[0] < 0.:
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
            last_C_logp = C.logp

            # Propose a value
            A.value = 1. + normal()

            # Check the argument values
            assert(C in L.ultimate_args)
            a_loc = L.ultimate_args.index(A)
            c_loc = L.ultimate_args.index(C)

            # Accept or reject values
            acc = True
            try:
                C.logp

                # Make sure A's value and last value occupy correct places in B's
                # cached arguments
                cur_frame = B._value.get_frame_queue()[1]
                assert(
                    B._value.get_cached_counts()[a_loc,
                                                 cur_frame] == A.counter.get_count())
                assert(
                    B._value.get_cached_counts()[a_loc,
                                                 1 - cur_frame] == last_A_count)
                assert(B._value.ultimate_args[a_loc] is A)

            except ZeroProbability:

                acc = False

                # Reject jump
                A.revert()

                # Make sure A's value and last value occupy correct places in B's
                # cached arguments
                cur_frame = B._value.get_frame_queue()[1]
                assert(
                    B._value.get_cached_counts()[a_loc,
                                                 1 - cur_frame] == A.counter.get_count())
                assert(B.value is last_B_value)

            # Check C's cache
            cur_frame = L.get_frame_queue()[1]

            # If jump was accepted:
            if acc:
                # C's value should be at the head of C's cache
                assert_equal(
                    L.get_cached_counts()[c_loc,
                                          cur_frame],
                    C.counter.get_count())
                assert(L.cached_values[cur_frame] is C.logp)

            # If jump was rejected:
            else:

                # B's value should be at the back of C's cache.
                assert_equal(
                    L.get_cached_counts()[c_loc,
                                          1 - cur_frame],
                    C.counter.get_count())
                assert(L.cached_values[1 - cur_frame] is C.logp)

if __name__ == '__main__':
    import unittest
    unittest.main()
