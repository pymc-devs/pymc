from numpy.testing import *
from numpy import *
from pymc.examples import DisasterModel as DM
from pymc import Container

class test_Container(TestCase):
    def test(self):

# Test set container:

        S = [DM.e, DM.s, DM.l]
        R = Container(S)

        for item in R:
            assert(item in S)

        for item in R.value:
            val_in_S = False
            for S_item in S:
                if S_item.value is item:
                    val_in_S = True

            assert(val_in_S)

        # Test list/dictionary container:

        A = [[DM.e, DM.s], [DM.l, DM.D, 3.], 54.323]
        C = Container(A)

        for i in range(2):
            for j in range(2):
                assert(C[i][j] is A[i][j])
                assert(C.value[i][j] is A[i][j].value)

            assert(C[1][2] is A[1][2])
            assert(C.value[1][2] is A[1][2])

            assert(C[2] is A[2])
            assert(C.value[2] is A[2])

        # Test array container:

        B = ndarray((3,3),dtype=object)
        B[0,:] = DM.e
        B[1,:] = 1.
        B[2,:] = A
        D = Container(B)


        for i in range(2):
            assert(D[0,i] is DM.e)
            assert(D.value[0,i] is DM.e.value)

            assert(D[1,i] is 1.)
            assert(D.value[1,i] is 1.)

        P = D[2,:]
        Q = D.value[2,:]

        for i in range(2):
            for j in range(2):
                assert(P[i][j] is A[i][j])
                assert(Q[i][j] is A[i][j].value)

            assert(P[1][2] is A[1][2])
            assert(Q[1][2] is A[1][2])

            assert(P[2] is A[2])
            assert(Q[2] is A[2])


if __name__ == '__main__':
    import unittest
    unittest.main()
